# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open.

Skinny shapes (m <= 32) are memory-bound, so the BF16 weights are stored losslessly in
12 bits: a byte with sign + 7 mantissa bits, and a 4-bit code for the exponent relative to
a per-tensor base (a 16-binade window, placed to cover the most weights). The few weights
outside the window decode to a fixed placeholder and are corrected exactly in FP32 when
their output is written. The kernel decodes in registers straight into mma.sync A
fragments. Each CTA owns a 128-row block and a K-slice; the K-slices of a row block form
a thread-block cluster and reduce through distributed shared memory. The router (FP32)
runs as a BF16 hi + lo pair concatenated along K, with FP32 accumulation. Larger m goes
to cuBLAS.
"""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor  # bfloat16 [2304, 8192]
    o: torch.Tensor         # bfloat16 [8192, 1024]
    pooler: torch.Tensor    # bfloat16 [4096, 8192]
    unpooler: torch.Tensor  # bfloat16 [8192, 4096]
    router: torch.Tensor    # float32 [256, 8192]


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor  # bfloat16 [m, k]
    config: Config


CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace cg = cooperative_groups;

constexpr int ROWS = 128;  // output rows per CTA (8 row tiles of 16)
constexpr int KUNIT = 64;  // k per unit
constexpr int PF = 4;      // units in flight per warp
constexpr int WSTRIDE = 32;

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}

__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

__device__ __forceinline__ void ldmatrix_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(s));
}

__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Four 12-bit weights -> two registers of BF16 pairs. S: per byte sign<<7 | mantissa.
// C: per byte the exponent code (0..15); base4: the base exponent in every byte.
__device__ __forceinline__ void decode4(uint32_t S, uint32_t C, uint32_t base4, uint32_t& r0, uint32_t& r1) {
  uint32_t T = C + base4;
  uint32_t H = (S & 0x80808080u) | ((T >> 1) & 0x7F7F7F7Fu);
  uint32_t L = (S & 0x7F7F7F7Fu) | ((T << 7) & 0x80808080u);
  r0 = __byte_perm(L, H, 0x5140);
  r1 = __byte_perm(L, H, 0x7362);
}

template <bool OUTF32>
__device__ __forceinline__ void store_out(void* OUT, size_t o, float v) {
  if constexpr (OUTF32) reinterpret_cast<float*>(OUT)[o] = v;
  else reinterpret_cast<__nv_bfloat16*>(OUT)[o] = __float2bfloat16_rn(v);
}

constexpr int CAP = 16;      // exception slots per (row block, k unit)
constexpr int EXC_MAX = 1024;

__device__ unsigned long long g_ts[1024][8];
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define TS(i) do { if ((dbg & 64) && tid == 0) g_ts[blockIdx.x][i] = gtime(); } while (0)

__device__ __forceinline__ unsigned ld_acquire(const unsigned* p) {
  unsigned v;
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];\n" : "=r"(v) : "l"(p));
  return v;
}

// Grid: (N / 128) row blocks x S K-slices, slice-fastest. Warp w handles row tile w % 8
// and every WPR-th unit of the slice starting at w / 8.
// Reduction over slices: S == 1 stores directly; C == S > 1 reduces in a cluster through
// distributed shared memory; C == 0 stores partials to WS and, after a per-row-block
// barrier (every CTA is resident: one per SM), each slice sums its share of the rows.
template <int NT, bool COMP, bool OUTF32, int WPR>
__global__ void __launch_bounds__(256 * WPR, 1)
skinny_kernel(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, void* __restrict__ OUT,
              float* __restrict__ WS, unsigned* __restrict__ BAR, const int2* __restrict__ SLOTS, int m, int N,
              int KU, int Kx, int S, int C, uint32_t base0, uint32_t base1, int dbg) {
  constexpr int THREADS = 256 * WPR;
  constexpr int MP = NT * 8;
  constexpr int WBW = COMP ? 1536 : 2048;
  constexpr int NV = COMP ? 3 : 4;
  constexpr size_t USTRIDE = 8 * WBW;
  constexpr int SPT = 2;  // exception slots per thread (n * CAP <= SPT * THREADS)
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_nexc;
  __shared__ int2 s_exc[EXC_MAX];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int rt = warp & 7, half = warp >> 3;
  const int rb = blockIdx.x / S, sl = blockIdx.x % S;
  const int ku0 = sl * KU / S, n = (sl + 1) * KU / S - ku0;
  const int nw = n > half ? (n - half + WPR - 1) / WPR : 0;
  if (tid == 0) s_nexc = 0;
  TS(0);

  // x's slice first: compute waits on it.
  for (int c = tid; c < n * MP * 8; c += THREADS) {
    int ul = c / (MP * 8), r = (c >> 3) % MP, ch = c & 7;
    bool valid = r < m;
    int kx = ((ku0 + ul) * KUNIT) % Kx + ch * 8;
    cp_async16(smem + (ul * MP + r) * 128 + ((ch ^ (r & 7)) << 4), X + (size_t)(valid ? r : 0) * Kx + kx,
               valid ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;\n" ::);

  const uint8_t* wp = W + ((size_t)(rb * KU + ku0) * 8 + rt) * WBW + lane * 16;
  uint4 buf[PF][NV];
#pragma unroll
  for (int p = 0; p < PF; p++)
    if (p < nw)
#pragma unroll
      for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + (half + p * WPR) * USTRIDE + v * 512);

  // This slice's exception slots; not consumed until after the main loop.
  int2 my_slot[SPT];
  const int2* slots = SLOTS + (size_t)(rb * KU + ku0) * CAP;
#pragma unroll
  for (int i = 0; i < SPT; i++) {
    const int j = tid + i * THREADS;
    my_slot[i] = j < n * CAP ? slots[j] : make_int2(-1, 0);
  }

  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  TS(1);

  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

  for (int i = 0; i < nw; i += PF) {
#pragma unroll
    for (int p = 0; p < PF; p++) {
      const int iu = i + p;
      if (iu < nw) {
        const int u = half + iu * WPR;
        const uint8_t* sX = smem + u * MP * 128;
        uint32_t b[NT][8];
#pragma unroll
        for (int c = 0; c < NT * 2; c++) {
          int q = 4 * c + (lane >> 3);
          int row = (q >> 3) * 8 + (lane & 7);
          uint32_t r0, r1, r2, r3;
          ldmatrix_x4(r0, r1, r2, r3, sX + row * 128 + (((q & 7) ^ (row & 7)) << 4));
          int q0 = 4 * c;
          b[q0 >> 3][(q0 & 7) + 0] = r0;
          b[q0 >> 3][(q0 & 7) + 1] = r1;
          b[q0 >> 3][(q0 & 7) + 2] = r2;
          b[q0 >> 3][(q0 & 7) + 3] = r3;
        }
        if constexpr (COMP) {
          const uint32_t base4 = ((ku0 + u) * KUNIT < Kx) ? base0 : base1;
          uint32_t sw[8] = {buf[p][0].x, buf[p][0].y, buf[p][0].z, buf[p][0].w,
                            buf[p][1].x, buf[p][1].y, buf[p][1].z, buf[p][1].w};
          uint32_t cw[4] = {buf[p][2].x, buf[p][2].y, buf[p][2].z, buf[p][2].w};
#pragma unroll
          for (int j = 0; j < 4; j++) {
            uint32_t a[4];
            decode4(sw[2 * j], cw[j] & 0x0F0F0F0Fu, base4, a[0], a[1]);
            decode4(sw[2 * j + 1], (cw[j] >> 4) & 0x0F0F0F0Fu, base4, a[2], a[3]);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma16816(acc[nt], a, b[nt][2 * j], b[nt][2 * j + 1]);
          }
        } else {
#pragma unroll
          for (int j = 0; j < 4; j++) {
            uint32_t a[4] = {buf[p][j].x, buf[p][j].y, buf[p][j].z, buf[p][j].w};
#pragma unroll
            for (int nt = 0; nt < NT; nt++) mma16816(acc[nt], a, b[nt][2 * j], b[nt][2 * j + 1]);
          }
        }
        if (iu + PF < nw)
#pragma unroll
          for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + (u + PF * WPR) * USTRIDE + v * 512);
      }
    }
  }

  if (dbg & 64) {
    __syncthreads();
    TS(2);
  }
  if (dbg & 1) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < NT; i++) t += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    if (t == 12345.f) WS[0] = t + my_slot[0].y;
    return;
  }

  // Compact the occupied exception slots.
#pragma unroll
  for (int i = 0; i < SPT; i++)
    if (my_slot[i].x >= 0) s_exc[atomicAdd(&s_nexc, 1)] = my_slot[i];

  // This CTA's partial, [MP][ROWS] in FP32, after the x slice; the same offset in every
  // CTA (slices differ in length by at most one unit).
  float* part = reinterpret_cast<float*>(smem + ((KU + S - 1) / S) * MP * 128);
#pragma unroll
  for (int h = 0; h < WPR; h++) {
    if (half == h) {
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        int t0 = nt * 8 + 2 * t4, r = rt * 16 + g;
        float* p0 = part + t0 * ROWS + r;
        if (h == 0) {
          p0[0] = acc[nt][0]; p0[ROWS] = acc[nt][1]; p0[8] = acc[nt][2]; p0[ROWS + 8] = acc[nt][3];
        } else {
          p0[0] += acc[nt][0]; p0[ROWS] += acc[nt][1]; p0[8] += acc[nt][2]; p0[ROWS + 8] += acc[nt][3];
        }
      }
    }
    __syncthreads();
  }
  TS(3);

  // Correct the placeholders: + x * (w - placeholder), exact products summed in FP32.
  // Token-major, so a warp's lanes take different exceptions.
  const int ne = s_nexc;
  for (int wi = tid; wi < ne * m; wi += THREADS) {
    const int t = wi / ne, e = wi - t * ne;
    const int2 ent = s_exc[e];
    const int r = ent.x & 127, kk = ent.x >> 7;
    const bool hi = kk >= Kx;
    const float wv = __int_as_float(ent.y);
    const float fake = __uint_as_float((hi ? (base1 & 0xFF) : (base0 & 0xFF)) << 23);
    const int kl = kk - ku0 * KUNIT, ul = kl >> 6, kin = kl & 63;
    const float xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
        smem + (ul * MP + t) * 128 + (((kin >> 3) ^ (t & 7)) << 4) + (kin & 7) * 2));
    atomicAdd(part + t * ROWS + r, xv * wv - xv * fake);
  }
  __syncthreads();
  TS(4);

  constexpr int IT = (ROWS * MP + THREADS - 1) / THREADS;
  if (S == 1) {
#pragma unroll
    for (int i = 0; i < IT; i++) {
      const int idx = tid + i * THREADS;
      if (idx < ROWS * m) store_out<OUTF32>(OUT, (size_t)(idx / ROWS) * N + rb * ROWS + idx % ROWS, part[idx]);
    }
    TS(7);
    return;
  }

  if (C > 1) {
    // One cluster per row block: rank r sums rows [r*128/C, (r+1)*128/C) over the cluster.
    cg::cluster_group cluster = cg::this_cluster();
    cluster.sync();
    TS(5);
    const int r0 = sl * ROWS / C, nrows = (sl + 1) * ROWS / C - r0, nitems = nrows * m;
    float v[IT];
#pragma unroll
    for (int i = 0; i < IT; i++) {
      const int idx = tid + i * THREADS;
      v[i] = 0.f;
      if (idx < nitems) {
        const int r = r0 + idx % nrows, t = idx / nrows;
        for (int q = 0; q < C; q++) v[i] += cluster.map_shared_rank(part, q)[t * ROWS + r];
      }
    }
#pragma unroll
    for (int i = 0; i < IT; i++) {
      const int idx = tid + i * THREADS;
      if (idx < nitems)
        store_out<OUTF32>(OUT, (size_t)(idx / nrows) * N + rb * ROWS + r0 + idx % nrows, v[i]);
    }
    TS(6);
    cluster.sync();  // peers read this CTA's shared memory until here
    TS(7);
    return;
  }

  // Partials through global memory, then a barrier over this row block's S slices.
  float* pw = WS + (size_t)blockIdx.x * (ROWS * 32);
#pragma unroll
  for (int i = 0; i < IT; i++) {
    const int idx = tid + i * THREADS;
    if (idx < ROWS * m) __stcg(pw + idx, part[idx]);
  }
  __syncthreads();
  if (tid == 0) {
    unsigned* cnt = BAR + 2 * rb;
    unsigned* gen = cnt + 1;
    const unsigned g0 = ld_acquire(gen);
    __threadfence();
    if (atomicAdd(cnt, 1u) == (unsigned)S - 1) {
      *cnt = 0;
      __threadfence();
      atomicAdd(gen, 1u);
    } else {
      while (ld_acquire(gen) == g0) {
      }
    }
    __threadfence();
  }
  __syncthreads();
  TS(5);
  const int r0 = sl * ROWS / S, nrows = (sl + 1) * ROWS / S - r0, nitems = nrows * m;
  int L = 1;  // lanes per output, splitting the S partials
  while (L < 32 && 2 * L <= S && nitems * 2 * L <= THREADS) L *= 2;
  const float* wsb = WS + (size_t)rb * S * (ROWS * 32);
  for (int base = 0; base < nitems; base += THREADS / L) {
    const int item = base + tid / L, sub = tid % L;
    float v = 0.f;
    int r = 0, t = 0;
    if (item < nitems) {
      r = r0 + item % nrows;
      t = item / nrows;
#pragma unroll 8
      for (int q = sub; q < S; q += L) v += __ldcg(wsb + (size_t)q * (ROWS * 32) + t * ROWS + r);
    }
    for (int o = L / 2; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if (item < nitems && sub == 0) store_out<OUTF32>(OUT, (size_t)t * N + rb * ROWS + r, v);
  }
  TS(7);
}

torch::Tensor get_ts(int64_t n) {
  auto out = torch::empty({n, 8}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(out.data_ptr(), g_ts, n * 8 * sizeof(unsigned long long));
  return out;
}

// ---------------------------------------------------------------------------------------
// m <= 8: no cross-CTA reduction. A CTA owns TPC row tiles (16 rows each) over all of K;
// its 8 warps split (tile, K-slice), each streaming its weight fragments through a private
// cp.async ring, with x's B fragments read through L1. Warps reduce in shared memory.
constexpr int CAPT = 64;    // exception slots per 16-row tile

template <bool COMP, int GD>
__global__ void __launch_bounds__(256, 2)
gemv_kernel(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ OUT,
            const int2* __restrict__ TSLOTS, int m, int N, int KU, int Kx, int TPC, uint32_t base0, int dbg) {
  constexpr int WBW = COMP ? 1536 : 2048;
  constexpr int NV = COMP ? 3 : 4;
  extern __shared__ __align__(128) uint8_t smem[];
  float* red = reinterpret_cast<float*>(smem + 8 * GD * WBW);  // [8 warps][16 rows][8 tokens]

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int WSL = 8 / TPC;                 // K-slices per tile
  const int tl = warp % TPC, ks = warp / TPC;
  const int tile = blockIdx.x * TPC + tl;  // global 16-row tile
  const int rb = tile >> 3, rt = tile & 7;
  const int ku0 = ks * KU / WSL, nu = (ks + 1) * KU / WSL - ku0;

  // Exception slots for this CTA's tiles; consumed at the end.
  int2 my_slot = make_int2(-1, 0);
  if (tid < TPC * CAPT) my_slot = TSLOTS[(size_t)blockIdx.x * TPC * CAPT + tid];

  uint8_t* ring = smem + warp * GD * WBW + lane * 16;
  const uint8_t* wp = W + ((size_t)(rb * KU + ku0) * 8 + rt) * WBW + lane * 16;
  constexpr size_t USTRIDE = 8 * WBW;
#pragma unroll
  for (int p = 0; p < GD - 1; p++) {
    if (p < nu)
#pragma unroll
      for (int v = 0; v < NV; v++) cp_async16(ring + p * WBW + v * 512, wp + p * USTRIDE + v * 512, 16);
    asm volatile("cp.async.commit_group;\n" ::);
  }

  const uint32_t* x32 = reinterpret_cast<const uint32_t*>(X);
  const bool tok = g < m;
  const size_t xrow = (size_t)(tok ? g : 0) * (Kx / 2);
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < nu; i++) {
    const int kb = (ku0 + i) * KUNIT;
    uint32_t b[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int k = (kb + j * 16 + 2 * t4) % Kx;
      b[2 * j] = tok ? __ldg(x32 + xrow + k / 2) : 0u;
      b[2 * j + 1] = tok ? __ldg(x32 + xrow + (k + 8) / 2) : 0u;
    }
    asm volatile("cp.async.wait_group %0;\n" ::"n"(GD - 2));
    const uint8_t* sp = ring + (i % GD) * WBW;
    uint4 q[NV];
#pragma unroll
    for (int v = 0; v < NV; v++) q[v] = *reinterpret_cast<const uint4*>(sp + v * 512);
    {
      const int un = i + GD - 1;
      if (un < nu)
#pragma unroll
        for (int v = 0; v < NV; v++)
          cp_async16(ring + (un % GD) * WBW + v * 512, wp + un * USTRIDE + v * 512, 16);
      asm volatile("cp.async.commit_group;\n" ::);
    }
    if constexpr (COMP) {
      uint32_t sw[8] = {q[0].x, q[0].y, q[0].z, q[0].w, q[1].x, q[1].y, q[1].z, q[1].w};
      uint32_t cw[4] = {q[2].x, q[2].y, q[2].z, q[2].w};
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t a[4];
        decode4(sw[2 * j], cw[j] & 0x0F0F0F0Fu, base0, a[0], a[1]);
        decode4(sw[2 * j + 1], (cw[j] >> 4) & 0x0F0F0F0Fu, base0, a[2], a[3]);
        mma16816(acc, a, b[2 * j], b[2 * j + 1]);
      }
    } else {
      // Raw layout: per k16 step j, 16 bytes per lane at j * 512.
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t a[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
        mma16816(acc, a, b[2 * j], b[2 * j + 1]);
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::);

  // Partials: red[warp][row][token]; C fragment rows g, g+8, tokens 2*t4, 2*t4+1.
  float* rw = red + warp * 128;
  rw[g * 8 + 2 * t4] = acc[0];
  rw[g * 8 + 2 * t4 + 1] = acc[1];
  rw[(g + 8) * 8 + 2 * t4] = acc[2];
  rw[(g + 8) * 8 + 2 * t4 + 1] = acc[3];
  __syncthreads();
  // Exceptions: + x * (w - placeholder) into the first slice's partial of their tile.
  if (my_slot.x >= 0) {
    const int etl = tid / CAPT, r = my_slot.x & 15, k = my_slot.x >> 4;
    const float wv = __int_as_float(my_slot.y);
    const float fake = __uint_as_float((base0 & 0xFF) << 23);
    for (int t = 0; t < m; t++) {
      const float xv = __bfloat162float(X[(size_t)t * Kx + k]);
      atomicAdd(red + etl * 128 + r * 8 + t, xv * wv - xv * fake);
    }
  }
  __syncthreads();
  // One output per thread: (tile, row, token).
  const int ot = tid / 128, orow = (tid >> 3) & 15, otok = tid & 7;
  if (ot < TPC && otok < m) {
    float v = 0.f;
    for (int s = 0; s < WSL; s++) v += red[(s * TPC + ot) * 128 + orow * 8 + otok];
    OUT[(size_t)otok * N + (blockIdx.x * TPC + ot) * 16 + orow] = __float2bfloat16_rn(v);
  }
  if (TPC > 2) {
    const int ot2 = ot + 2;
    if (ot2 < TPC && otok < m) {
      float v = 0.f;
      for (int s = 0; s < WSL; s++) v += red[(s * TPC + ot2) * 128 + orow * 8 + otok];
      OUT[(size_t)otok * N + (blockIdx.x * TPC + ot2) * 16 + orow] = __float2bfloat16_rn(v);
    }
  }
}

template <bool COMP, int GD>
void launch_gemv_t(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, int m, int N, int KU,
                   int Kx, int TPC, uint32_t base0, int dbg, int smem) {
  auto kern = gemv_kernel<COMP, GD>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kern<<<N / (16 * TPC), 256, smem, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<uint8_t>(), (const __nv_bfloat16*)X.data_ptr(), (__nv_bfloat16*)OUT.data_ptr(),
      (const int2*)TSLOTS.data_ptr<int>(), m, N, KU, Kx, TPC, base0, dbg);
}

void launch_gemv(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, int64_t m, int64_t N,
                 int64_t KU, int64_t Kx, int64_t TPC, int64_t base0, bool comp, int64_t dbg) {
  const int WBW = comp ? 1536 : 2048;
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  const int gd = (8 * 8 * WBW + 8 * 128 * 4 <= optin) ? 8 : 4;
  const int smem = 8 * gd * WBW + 8 * 128 * 4;
  if (comp) {
    if (gd == 8) launch_gemv_t<true, 8>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
    else launch_gemv_t<true, 4>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
  } else {
    if (gd == 8) launch_gemv_t<false, 8>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
    else launch_gemv_t<false, 4>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
  }
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gemv launch failed: ", cudaGetErrorString(err));
}

constexpr int WPR = 2;

template <int NT, bool COMP, bool OUTF32>
int smem_bytes(int KU, int S) {
  int nmax = (KU + S - 1) / S;
  return nmax * NT * 8 * 128 + NT * 8 * ROWS * 4;
}

template <int NT, bool COMP, bool OUTF32>
void launch_t(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor BAR,
              torch::Tensor SLOTS, int m, int N, int KU, int Kx, int64_t base0, int64_t base1,
              int S, int C, int dbg) {
  auto kern = skinny_kernel<NT, COMP, OUTF32, WPR>;
  const int smem = smem_bytes<NT, COMP, OUTF32>(KU, S);
  static int optin = 0;
  if (!optin) cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  TORCH_CHECK(smem <= optin, "x slice does not fit in shared memory");
  cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / ROWS) * S);
  cfg.blockDim = dim3(256 * WPR);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C > 1 ? C : 1;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kern, (const uint8_t*)W.data_ptr<uint8_t>(),
      (const __nv_bfloat16*)X.data_ptr(), OUT.data_ptr(), WS.data_ptr<float>(), (unsigned*)BAR.data_ptr<int>(),
      (const int2*)SLOTS.data_ptr<int>(), m, N, KU, Kx, S, C,
      (uint32_t)base0, (uint32_t)base1, dbg);
  TORCH_CHECK(err == cudaSuccess, "launch failed: ", cudaGetErrorString(err));
}

template <int NT, bool COMP, bool OUTF32>
int64_t clusters_t(int KU, int S, int C, int N) {
  auto kern = skinny_kernel<NT, COMP, OUTF32, WPR>;
  const int smem = smem_bytes<NT, COMP, OUTF32>(KU, S);
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  if (smem > optin) return 0;
  cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaGetLastError();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / ROWS) * S);
  cfg.blockDim = dim3(256 * WPR);
  cfg.dynamicSmemBytes = smem;
  cudaLaunchAttribute at[1];
  at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = C;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  int n = 0;
  if (cudaOccupancyMaxActiveClusters(&n, kern, &cfg) != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return n;
}

#define DISPATCH(m, comp, f32, CALL)                                                   \
  do {                                                                                 \
    int nt_ = (m) <= 8 ? 1 : ((m) <= 16 ? 2 : 4);                                      \
    if (comp) {                                                                        \
      if (f32) { if (nt_ == 1) CALL(1, true, true); else if (nt_ == 2) CALL(2, true, true); else CALL(4, true, true); } \
      else { if (nt_ == 1) CALL(1, true, false); else if (nt_ == 2) CALL(2, true, false); else CALL(4, true, false); } \
    } else {                                                                           \
      if (f32) { if (nt_ == 1) CALL(1, false, true); else if (nt_ == 2) CALL(2, false, true); else CALL(4, false, true); } \
      else { if (nt_ == 1) CALL(1, false, false); else if (nt_ == 2) CALL(2, false, false); else CALL(4, false, false); } \
    }                                                                                  \
  } while (0)

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor BAR,
            torch::Tensor SLOTS, int64_t m, int64_t N, int64_t KU, int64_t Kx,
            int64_t base0, int64_t base1, bool comp, int64_t S, int64_t C, int64_t dbg) {
  bool f32 = OUT.scalar_type() == torch::kFloat32;
#define CALL(NTV, CP, F) launch_t<NTV, CP, F>(W, X, OUT, WS, BAR, SLOTS, m, N, KU, Kx, base0, base1, S, C, dbg)
  DISPATCH(m, comp, f32, CALL);
#undef CALL
}

int64_t max_clusters(int64_t m, bool comp, bool f32, int64_t KU, int64_t S, int64_t C, int64_t N) {
  int64_t r = 0;
#define CALL(NTV, CP, F) r = clusters_t<NTV, CP, F>(KU, S, C, N)
  DISPATCH(m, comp, f32, CALL);
#undef CALL
  return r;
}
"""

CPP_SRC = """
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor BAR,
            torch::Tensor SLOTS, int64_t m, int64_t N, int64_t KU, int64_t Kx,
            int64_t base0, int64_t base1, bool comp, int64_t S, int64_t C, int64_t dbg);
int64_t max_clusters(int64_t m, bool comp, bool f32, int64_t KU, int64_t S, int64_t C, int64_t N);
torch::Tensor get_ts(int64_t n);
void launch_gemv(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, int64_t m, int64_t N,
                 int64_t KU, int64_t Kx, int64_t TPC, int64_t base0, bool comp, int64_t dbg);
"""

ROWS, KUNIT = 128, 64
MAX_M = 32
WSTRIDE = 32

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="okc_bf16_proj_v5",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["launch", "max_clusters", "get_ts", "launch_gemv"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
            verbose=False,
        )
    return _ext


def _frag_view(w: torch.Tensor) -> torch.Tensor:
    """[N, K] -> [rb, w, h, g, ku, j, khalf, t4, p] (row = ((rb*8+w)*2+h)*8+g, k = (((ku*4+j)*2+khalf)*4+t4)*2+p)."""
    n, k = w.shape
    return w.reshape(n // ROWS, 8, 2, 8, k // KUNIT, 4, 2, 4, 2)


def _pack_raw(w: torch.Tensor) -> torch.Tensor:
    """BF16 in mma fragment order: per (rb, ku, warp), per k16 step, 16 bytes per lane."""
    v = _frag_view(w.view(torch.int16))
    return v.permute(0, 4, 1, 5, 3, 7, 6, 2, 8).contiguous().view(torch.uint8).flatten()


def _pack_comp(w: torch.Tensor):
    """12-bit lossless packing: (packed bytes, base exponent, per-row exceptions (col, FP32 bits))."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    hist = torch.bincount(e.flatten(), minlength=256).cpu()
    # The 16-exponent window [base, base + 15] that covers the most weights (base >= 1).
    cover = torch.stack([hist[b:b + 16].sum() for b in range(1, 241)])
    base = int(cover.argmax()) + 1
    exc = (e < base) | (e > base + 15)
    code = torch.where(exc, 0, e - base)
    s = torch.where(exc, 0, ((bits >> 8) & 0x80) | (bits & 0x7F))

    sv = _frag_view(s.to(torch.uint8))
    sv = sv.reshape(n // ROWS, 8, 2, 8, k // KUNIT, 2, 2, 2, 4, 2)  # j -> (jhalf, jlow)
    s_packed = sv.permute(0, 4, 1, 5, 3, 8, 6, 7, 2, 9).reshape(n // ROWS, k // KUNIT, 8, 1024)
    cv = _frag_view(code.to(torch.uint8))
    cbyte = cv[:, :, :, :, :, :, 0] | (cv[:, :, :, :, :, :, 1] << 4)  # [rb, w, h, g, ku, j, t4, p]
    c_packed = cbyte.permute(0, 4, 1, 3, 6, 5, 2, 7).reshape(n // ROWS, k // KUNIT, 8, 512)
    packed = torch.cat([s_packed, c_packed], dim=-1).contiguous().flatten()

    idx = exc.nonzero()  # row-major, so already sorted by row
    rows, cols = idx[:, 0], idx[:, 1]
    return packed, base, rows, cols, w[rows, cols].float().view(torch.int32)


CAP = 16  # exception slots per (row block, k unit); must match the kernel


def _slots(n: int, ku: int, rows, cols, vals, dev):
    """Exceptions in fixed slots per (row block, k unit): (row % 128 | k << 7, FP32 bits), or
    x = -1 when empty. None if some unit has more than CAP."""
    slots = torch.full(((n // ROWS) * ku, CAP, 2), -1, dtype=torch.int32, device=dev)
    if rows.numel():
        key = (rows // ROWS) * ku + cols // KUNIT
        order = torch.argsort(key, stable=True)
        rows, cols, vals, key = rows[order], cols[order], vals[order], key[order]
        counts = torch.bincount(key, minlength=(n // ROWS) * ku)
        if int(counts.max()) > CAP:
            return None
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(key.numel(), device=dev) - start[key]
        slots[key, pos, 0] = ((rows % ROWS) | (cols << 7)).int()
        slots[key, pos, 1] = vals
    return slots.flatten()


CAPT = 64  # exception slots per 16-row tile (gemv kernel)
GEMV_TPC = {"qkv_gate": 1, "o": 4, "pooler": 1, "unpooler": 1}


def _tile_slots(n: int, rows, cols, vals, dev):
    """Exceptions in fixed slots per 16-row tile: (row % 16 | k << 4, FP32 bits), x = -1 when
    empty. None if a tile has more than CAPT."""
    slots = torch.full((n // 16, CAPT, 2), -1, dtype=torch.int32, device=dev)
    if rows.numel():
        key = rows // 16
        counts = torch.bincount(key, minlength=n // 16)
        if int(counts.max()) > CAPT:
            return None
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(key.numel(), device=dev) - start[key]  # rows are sorted (row-major nonzero)
        slots[key, pos, 0] = ((rows % 16) | (cols << 4)).int()
        slots[key, pos, 1] = vals
    return slots.flatten()


def _layer_state(w: torch.Tensor, comp: bool):
    n, k = w.shape
    if comp:
        packed, base, rows, cols, vals = _pack_comp(w)
        slots = _slots(n, k // KUNIT, rows, cols, vals, w.device)
        tslots = _tile_slots(n, rows, cols, vals, w.device)
        if slots is not None and tslots is not None:
            return {"w": packed, "base": base * 0x01010101, "base1": base * 0x01010101, "slots": slots,
                    "tslots": tslots, "ku": k // KUNIT, "kx": k, "comp": True, "nexc": int(rows.numel())}
    # Uncompressed (also the fallback if exceptions cluster beyond the slots).
    empty = torch.zeros(0, dtype=torch.int64, device=w.device)
    return {"w": _pack_raw(w), "base": 0x01010101, "base1": 0x01010101,
            "slots": _slots(n, k // KUNIT, empty, empty, empty, w.device),
            "tslots": _tile_slots(n, empty, empty, empty, w.device), "ku": k // KUNIT, "kx": k,
            "comp": False, "nexc": 0}


def _router_state(r: torch.Tensor, comp: bool):
    """FP32 [N, K] -> BF16 hi + lo concatenated along K (FP32 accumulation of both)."""
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    n, k = r.shape
    if comp:
        pa, ba, ra, ca, va = _pack_comp(hi)
        pb, bb, rb_, cb, vb = _pack_comp(lo)
        slots = _slots(n, 2 * k // KUNIT, torch.cat([ra, rb_]), torch.cat([ca, cb + k]), torch.cat([va, vb]),
                       r.device)
        if slots is not None:
            packed = torch.cat([pa.view(n // ROWS, k // KUNIT, -1), pb.view(n // ROWS, k // KUNIT, -1)], 1)
            return {"w": packed.flatten(), "base": ba * 0x01010101, "base1": bb * 0x01010101, "slots": slots,
                    "ku": 2 * k // KUNIT, "kx": k, "comp": True, "nexc": int(ra.numel() + rb_.numel())}
    st = _layer_state(torch.cat([hi, lo], 1), False)
    st["kx"] = k
    return st


COMPRESS = {"qkv_gate": True, "o": True, "pooler": True, "unpooler": True}
# Layers our kernel beats cuBLAS on so far; the rest stay on cuBLAS.
OURS = {"router"}
GEMV = {"qkv_gate", "o", "pooler", "unpooler"}
ROUTER_COMPRESS = False


def _plan(state, layer: str, m: int, n: int):
    """(S, C): K-slices per row block, and how they reduce: S == 1 none, C == S a cluster,
    C == 0 through global memory behind a barrier. One CTA per SM, all resident at once.
    None if nothing fits (then cuBLAS)."""
    key = (layer, m, n)
    if key in state["plans"]:
        return state["plans"][key]
    p = state["packs"][layer]
    nrb, ku = n // ROWS, p["ku"]
    mp = 8 if m <= 8 else (16 if m <= 16 else 32)
    s = max(1, min(ku, state["sms"] // nrb))
    plan = None
    if -(-ku // s) * mp * 128 + mp * ROWS * 4 <= state["smem"]:
        plan = (s, 1) if s == 1 else (s, 0)
        if 1 < s <= 16 and _module().max_clusters(m, p["comp"], layer == "router", ku, s, s, n) * s >= nrb * s:
            plan = (s, s)
    state["plans"][key] = plan
    return plan


def prepare(weights: Weights):
    dev = weights["qkv_gate"].device
    _module()
    state = {"raw": {name: w.contiguous() for name, w in weights.items() if name != "router"}}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    packs = {name: _layer_state(weights[name].contiguous(), COMPRESS[name]) for name in COMPRESS}
    packs["router"] = _router_state(r, ROUTER_COMPRESS)
    state["packs"] = packs
    props = torch.cuda.get_device_properties(dev)
    state["ws"] = torch.zeros(props.multi_processor_count * ROWS * 32, dtype=torch.float32, device=dev)
    state["bar"] = torch.zeros(2 * 64, dtype=torch.int32, device=dev)
    state["sms"] = props.multi_processor_count
    state["smem"] = props.shared_memory_per_block_optin - 1024
    state["plans"] = {}
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state, plan=None, dbg=0) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    m, n, layer = cfg["m"], cfg["n"], cfg["layer"]
    if m > MAX_M:
        if layer == "router":
            acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
            return acc
        return torch.mm(x, state["raw"][layer][:n].t())
    p = state["packs"][layer]
    if m <= 8 and layer in GEMV and plan is None and "tslots" in p:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _module().launch_gemv(p["w"], x.contiguous(), out, p["tslots"], m, n, p["ku"], p["kx"], GEMV_TPC[layer],
                              p["base"], p["comp"], dbg)
        return out
    plan = plan or (_plan(state, layer, m, n) if layer in OURS else None)
    if plan is None:
        if layer == "router":
            acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
            acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
            return acc
        return torch.mm(x, state["raw"][layer][:n].t())
    s, c = plan
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().launch(p["w"], x.contiguous(), out, state["ws"], state["bar"], p["slots"], m, n, p["ku"],
                     p["kx"], p["base"], p["base1"], p["comp"], s, c, dbg)
    return out


# ---------------------------------------------------------------- diagnostics (check runs only)
import time as _time

_STREAM_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void stream_read(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  uint32_t acc = 0;
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    uint4 v;
    asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i));
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x12345678u) out[0] = acc;
}
void stream(torch::Tensor buf, int64_t nbytes, torch::Tensor out, int64_t grid, int64_t threads) {
  stream_read<<<grid, threads, 0, at::cuda::getCurrentCUDAStream()>>>((const uint4*)buf.data_ptr(), nbytes / 16, (uint32_t*)out.data_ptr());
}
"""


def _gt(fn, flush, reps=25):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    ts = []
    for _ in range(reps):
        flush.max()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def _diag(weights, state):
    t0 = _time.time()
    flush = torch.empty(300 << 20, dtype=torch.uint8, device="cuda")
    st = load_inline(name="okc_stream_probe", cpp_sources="void stream(torch::Tensor buf, int64_t nbytes, torch::Tensor out, int64_t grid, int64_t threads);",
                     cuda_sources=_STREAM_SRC, functions=["stream"], extra_cuda_cflags=["-O3"])
    out = torch.zeros(4, dtype=torch.int32, device="cuda")
    big = torch.empty(80 << 20, dtype=torch.uint8, device="cuda")
    print("empty graph", f"{_gt(lambda: None, flush):.2f}")
    for mb in ():
        row = []
        for grid, thr in ((132, 1024), (264, 512), (528, 512), (1056, 256)):
            row.append(f"{_gt(lambda: st.stream(big, mb << 20, out, grid, thr), flush):6.2f}")
        print(f"stream {mb:3d}MB", *row)
    for layer in ("qkv_gate", "o", "pooler", "unpooler"):
        w = weights[layer]
        k = w.shape[1]
        for m in (1, 8):
            n = 1024 if (layer == "unpooler" and m <= 8) else w.shape[0]
            x = torch.randn(m, k, device="cuda").to(torch.bfloat16)
            d = {"x": x, "config": {"m": m, "n": n, "k": k, "layer": layer}}
            res = [f"cublas {_gt(lambda: torch.mm(x, w[:n].t()), flush):6.2f}"]
            pl = _plan(state, layer, m, n)
            if pl:
                res.append(f"split{pl} {_gt(lambda: kernel_fn(d, state, plan=pl), flush):6.2f}")
            orig = GEMV_TPC[layer]
            for tpc in (1, 2, 4):
                if (n // 16) % tpc:
                    continue
                GEMV_TPC[layer] = tpc
                res.append(f"gemv{tpc} {_gt(lambda: kernel_fn(d, state), flush):6.2f}")
            GEMV_TPC[layer] = orig
            print(f"{layer:8s} m={m}", " | ".join(res), flush=True)
    print(f"diag took {_time.time() - t0:.1f}s")


_orig_prepare = prepare


def prepare(weights):
    state = _orig_prepare(weights)
    _diag(weights, state)
    return state
