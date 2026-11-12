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

__device__ unsigned long long g_ts[1024][10];
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
// Phase stamps (dbg & 64): [0] globaltimer at start, [1..8] clock64 per phase, [9] globaltimer at end.
#define TS(i) do { if ((dbg & 64) && tid == 0) { g_ts[blockIdx.x][(i) + 1] = clock64(); \
                    if ((i) == 0) g_ts[blockIdx.x][0] = gtime(); if ((i) == 7) g_ts[blockIdx.x][9] = gtime(); } } while (0)

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
  // Tokens beyond 32 run as further 32-token chunks along blockIdx.y.
  {
    const int ch = blockIdx.y;
    X += (size_t)ch * 32 * Kx;
    OUT = reinterpret_cast<uint8_t*>(OUT) + (size_t)ch * 32 * N * (OUTF32 ? 4 : 2);
    WS += (size_t)ch * gridDim.x * (ROWS * 32);
    BAR += ch * 128;
    m = min(32, m - ch * 32);
  }
  // The reduction's counter line is cold after an L2 flush: start fetching it now.
  if (tid == 0 && S > 1 && C < 2) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(BAR + 2 * rb));
  TS(0);

  if (tid == 0) s_nexc = 0;

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

  // This slice's exception slots: issued now, consumed after the main loop.
  int2 my_slot[SPT];
  {
    const int2* slots = SLOTS + (size_t)(rb * KU + ku0) * CAP;
#pragma unroll
    for (int i = 0; i < SPT; i++) my_slot[i] = slots[min(tid + i * THREADS, n * CAP - 1)];
  }

  asm volatile("cp.async.wait_group 0;\n" ::);
  // This CTA's partial, [MP][ROWS] in FP32, after the x slice; the same offset in every
  // CTA (slices differ in length by at most one unit). It starts as the exception
  // corrections, + x * (w - placeholder), while the first weight units are in flight.
  float* part = reinterpret_cast<float*>(smem + ((KU + S - 1) / S) * MP * 128);
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

#pragma unroll
  for (int h = 0; h < WPR; h++) {
    if (half == h) {
#pragma unroll
      for (int nt = 0; nt < NT; nt++) {
        float* p0 = part + (nt * 8 + 2 * t4) * ROWS + rt * 16 + g;
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
  // Exceptions: compact the occupied slots (one shared atomic per warp), then
  // + x * (w - placeholder) in FP32, token-major so a warp's lanes take different exceptions.
  if (!(dbg & 4)) {
#pragma unroll
    for (int i = 0; i < SPT; i++) {
      const bool valid = tid + i * THREADS < n * CAP && my_slot[i].x >= 0;
      const unsigned mask = __ballot_sync(0xffffffffu, valid);
      int base = 0;
      if (lane == 0 && mask) base = atomicAdd(&s_nexc, __popc(mask));
      base = __shfl_sync(0xffffffffu, base, 0);
      if (valid) s_exc[base + __popc(mask & ((1u << lane) - 1))] = my_slot[i];
    }
    __syncthreads();
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
    if (ne) __syncthreads();
  }
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

  if (C < 0) {
    // Last arriver: every slice publishes its partial; the last to finish sums them all.
    __shared__ int s_last;
    float4* pw4 = reinterpret_cast<float4*>(WS + (size_t)blockIdx.x * (ROWS * 32));
    const float4* part4 = reinterpret_cast<const float4*>(part);
    if (!(dbg & 16))
      for (int idx = tid; idx < ROWS * m / 4; idx += THREADS) __stcg(pw4 + idx, part4[idx]);
    __syncthreads();
    TS(5);
    if (tid == 0) {
      __threadfence();
      unsigned* cnt = BAR + 2 * rb;
      s_last = atomicAdd(cnt, 1u) == (unsigned)S - 1;
      if (s_last) *cnt = 0;
      __threadfence();
    }
    __syncthreads();
    TS(6);
    if (!s_last) {
      TS(7);
      return;
    }
    // Sum all S partials (own included, from L2) in a fixed order; L lanes split the slices.
    const float4* wsb = reinterpret_cast<const float4*>(WS + (size_t)rb * S * (ROWS * 32));
    const int items = (ROWS / 4) * m;
    int L = 1;
    while (L < 32 && 2 * L <= S && items * 2 * L <= THREADS) L *= 2;
    for (int base = 0; base < items; base += THREADS / L) {
      const int item = base + tid / L, sub = tid % L;
      float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
      if (item < items) {
#pragma unroll 8
        for (int q = sub; q < S; q += L) {
          const float4 a = __ldcg(wsb + (size_t)q * (ROWS * 8) + item);
          v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
        }
      }
      for (int o = L / 2; o; o >>= 1) {
        v.x += __shfl_xor_sync(0xffffffffu, v.x, o);
        v.y += __shfl_xor_sync(0xffffffffu, v.y, o);
        v.z += __shfl_xor_sync(0xffffffffu, v.z, o);
        v.w += __shfl_xor_sync(0xffffffffu, v.w, o);
      }
      if (item < items && sub == 0) {
        const int t = item / (ROWS / 4), r = (item % (ROWS / 4)) * 4;
        const size_t o = (size_t)t * N + rb * ROWS + r;
        store_out<OUTF32>(OUT, o, v.x);
        store_out<OUTF32>(OUT, o + 1, v.y);
        store_out<OUTF32>(OUT, o + 2, v.z);
        store_out<OUTF32>(OUT, o + 3, v.w);
      }
    }
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
  auto out = torch::empty({n, 10}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(out.data_ptr(), g_ts, n * 10 * sizeof(unsigned long long));
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
  // m == 1: x's row in shared memory after the partials.
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(smem + 8 * GD * WBW + 8 * 128 * 4);

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int WSL = 8 / TPC;                 // K-slices per tile
  const int tl = warp % TPC, ks = warp / TPC;
  const int tile = blockIdx.x * TPC + tl;  // global 16-row tile
  const int rb = tile >> 3, rt = tile & 7;
  const int ku0 = ks * KU / WSL, nu = (ks + 1) * KU / WSL - ku0;

  if (m == 1) {
    for (int c = tid; c < Kx / 8; c += 256) cp_async16(sx + c * 8, X + c * 8, 16);
    asm volatile("cp.async.commit_group;\n" ::);
  }
  // Exception slots for this CTA's tiles; consumed at the end.
  int2 my_slot = TSLOTS[(size_t)blockIdx.x * TPC * CAPT + min(tid, TPC * CAPT - 1)];

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

  const uint32_t* x32 = reinterpret_cast<const uint32_t*>(m == 1 ? sx : X);
  const bool tok = g < m;
  const size_t xrow = (size_t)(tok ? g : 0) * (Kx / 2);
  if (m == 1) {
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
  }
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  for (int i = 0; i < nu; i++) {
    const int kb = (ku0 + i) * KUNIT + 2 * t4;
    uint32_t b[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      b[2 * j] = tok ? x32[xrow + (kb + j * 16) / 2] : 0u;
      b[2 * j + 1] = tok ? x32[xrow + (kb + j * 16 + 8) / 2] : 0u;
    }
    // The x group (m == 1) was committed first and is already complete.
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
  if (tid < TPC * CAPT && my_slot.x >= 0) {
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
  // Two CTAs per SM: ring depth 6 when it fits twice, else 4.
  const int xs = m == 1 ? Kx * 2 : 0;
  const int gd = (2 * (8 * 6 * WBW + 8 * 128 * 4 + xs) <= optin * 2 && 8 * 6 * WBW + 8 * 128 * 4 + xs <= 113 * 1024) ? 6 : 4;
  const int smem = 8 * gd * WBW + 8 * 128 * 4 + xs;
  if (comp) {
    if (gd == 6) launch_gemv_t<true, 6>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
    else launch_gemv_t<true, 4>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
  } else {
    if (gd == 6) launch_gemv_t<false, 6>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
    else launch_gemv_t<false, 4>(W, X, OUT, TSLOTS, m, N, KU, Kx, TPC, base0, dbg, smem);
  }
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "gemv launch failed: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------------------
// m <= 8, hybrid schedule. One 16-warp CTA per SM owns F whole 16-row tiles over all of K
// and reduces them in shared memory. Tiles left over after G * F are split into pieces
// that every CTA does first; each warp adds its piece into an FP32 workspace and the last
// contributor finalizes that tile mid-kernel, hidden behind the full tiles. Weights stream
// through per-warp cp.async rings; x sits in shared memory.
template <bool COMP, int GD, bool RED_IN_RING>
__global__ void __launch_bounds__(512, 1)
hgemv_kernel(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ OUT,
             const int2* __restrict__ TSLOTS, float* __restrict__ WS, int* __restrict__ CNT, int m, int N, int KU,
             int Kx, int F, int Tdp, uint32_t base0, int dbg) {
  constexpr int NW = 16;
  constexpr int WBW = COMP ? 1536 : 2048;
  constexpr int NV = COMP ? 3 : 4;
  extern __shared__ __align__(128) uint8_t smem[];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int XS = Kx + 8;  // padded x row (elements): rows land on different banks
  uint8_t* ring = smem + warp * GD * WBW + lane * 16;
  // Per-warp tile partials get their own buffer, or reuse the ring after the loop when x
  // leaves no room for both (RED_IN_RING).
  const bool red_in_ring = RED_IN_RING;
  float* scr = reinterpret_cast<float*>(smem + NW * GD * WBW);           // [NW][128] finalize scratch
  float* red = red_in_ring ? reinterpret_cast<float*>(smem) : scr + NW * 128;  // [F][NW][128]
  __nv_bfloat16* sx = reinterpret_cast<__nv_bfloat16*>(scr + NW * 128 + (red_in_ring ? 0 : F * NW * 128));

  const int T = N / 16, G = gridDim.x, c = blockIdx.x;
  const long long UL = (long long)(T - Tdp) * KU;
  const int p0 = (int)(c * UL / G), p1 = (int)((c + 1) * UL / G);
  const int wp0 = p0 + (p1 - p0) * warp / NW, nP = p0 + (p1 - p0) * (warp + 1) / NW - wp0;
  const int fu0 = F * KU * warp / NW, nF = F * KU * (warp + 1) / NW - fu0;
  const int nU = nP + nF;
  // Walks this warp's units in order, (tile, ku): pieces first (leftover tiles), then full tiles.
  struct It {
    int i, tile, ku;
  };
  const int ftile0 = c * F + fu0 / KU, fku0 = fu0 % KU;
  auto it_begin = [&]() { return nP > 0 ? It{0, Tdp + wp0 / KU, wp0 % KU} : It{0, ftile0, fku0}; };
  auto it_next = [&](It& u) {
    if (++u.i == nP) {
      u.tile = ftile0;
      u.ku = fku0;
    } else if (++u.ku == KU) {
      u.ku = 0;
      u.tile++;
    }
  };
  auto it_ptr = [&](const It& u) {
    return W + ((size_t)((u.tile >> 3) * KU + u.ku) * 8 + (u.tile & 7)) * WBW + lane * 16;
  };

  for (int cidx = tid; cidx < m * (Kx / 8); cidx += 512) {
    const int r = cidx / (Kx / 8), q = cidx % (Kx / 8);
    cp_async16(sx + r * XS + q * 8, X + (size_t)r * Kx + q * 8, 16);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  It nx = it_begin();  // next unit to fetch
#pragma unroll
  for (int p = 0; p < GD - 1; p++) {
    if (p < nU) {
      const uint8_t* src = it_ptr(nx);
#pragma unroll
      for (int v = 0; v < NV; v++) cp_async16(ring + p * WBW + v * 512, src + v * 512, 16);
      it_next(nx);
    }
    asm volatile("cp.async.commit_group;\n" ::);
  }
  // Exception slots of the full tiles; consumed at the end.
  const int2 my_slot = TSLOTS[(size_t)c * F * CAPT + min(tid, max(F * CAPT - 1, 0))];
  for (int i = tid; i < XS / 2; i += 512) reinterpret_cast<uint32_t*>(sx + m * XS)[i] = 0u;  // zero row
  if (!red_in_ring)
    for (int i = tid; i < F * NW * 128; i += 512) red[i] = 0.f;
  asm volatile("cp.async.wait_group %0;\n" ::"n"(GD - 1));  // x
  __syncthreads();

  // Lanes past the last token read the zero row.
  const uint32_t* xr = reinterpret_cast<const uint32_t*>(sx + (g < m ? g : m) * XS) + t4;
  float acc[4] = {0.f, 0.f, 0.f, 0.f};
  float hold[8];
  int htile[2] = {-1, -1}, nfull = 0;
  int seg_start = 0;
  It cu = it_begin();  // current unit
  for (int i = 0; i < nU; i++) {
    const uint32_t* xk = xr + cu.ku * (KUNIT / 2);
    uint32_t b[8];
#pragma unroll
    for (int j = 0; j < 4; j++) {
      b[2 * j] = xk[j * 8];
      b[2 * j + 1] = xk[j * 8 + 4];
    }
    asm volatile("cp.async.wait_group %0;\n" ::"n"(GD - 2));
    const uint8_t* sp = ring + (i % GD) * WBW;
    uint4 q[NV];
#pragma unroll
    for (int v = 0; v < NV; v++) q[v] = *reinterpret_cast<const uint4*>(sp + v * 512);
    {
      const int un = i + GD - 1;
      if (un < nU) {
        const uint8_t* src = it_ptr(nx);
#pragma unroll
        for (int v = 0; v < NV; v++) cp_async16(ring + (un % GD) * WBW + v * 512, src + v * 512, 16);
        it_next(nx);
      }
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
#pragma unroll
      for (int j = 0; j < 4; j++) {
        uint32_t a[4] = {q[j].x, q[j].y, q[j].z, q[j].w};
        mma16816(acc, a, b[2 * j], b[2 * j + 1]);
      }
    }
    const int tile = cu.tile;
    it_next(cu);
    const bool seg_end = (i + 1 == nU) || (i + 1 == nP) || cu.tile != tile;
    if (!seg_end) continue;
    if (i < nP) {
      // Piece of a leftover tile: add into the workspace, then count its units.
      float* wt = WS + (size_t)tile * 128;
      if (2 * t4 < m) {
        atomicAdd(wt + g * 8 + 2 * t4, acc[0]);
        atomicAdd(wt + (g + 8) * 8 + 2 * t4, acc[2]);
      }
      if (2 * t4 + 1 < m) {
        atomicAdd(wt + g * 8 + 2 * t4 + 1, acc[1]);
        atomicAdd(wt + (g + 8) * 8 + 2 * t4 + 1, acc[3]);
      }
      __threadfence();
      __syncwarp();
      int last = 0;
      if (lane == 0) last = atomicAdd(CNT + tile, i + 1 - seg_start) + (i + 1 - seg_start) == KU;
      last = __shfl_sync(0xffffffffu, last, 0);
      if (last) {
        // Finalize the tile: workspace + exceptions -> BF16; reset for the next call.
        __threadfence();
        float* sc = scr + warp * 128;
        for (int e = lane; e < 128; e += 32) {
          sc[e] = __ldcg(wt + e);
          __stcg(wt + e, 0.f);
        }
        __syncwarp();
        for (int e = lane; e < CAPT; e += 32) {
          const int2 ent = TSLOTS[(size_t)tile * CAPT + e];
          if (ent.x >= 0) {
            const int r = ent.x & 15, k = ent.x >> 4;
            const float wv = __int_as_float(ent.y), fake = __uint_as_float((base0 & 0xFF) << 23);
            for (int t = 0; t < m; t++) {
              const float xv = __bfloat162float(sx[t * XS + k]);
              atomicAdd(sc + r * 8 + t, xv * wv - xv * fake);
            }
          }
        }
        __syncwarp();
        for (int e = lane; e < 16 * m; e += 32) {
          const int r = e / m, t = e % m;
          OUT[(size_t)t * N + tile * 16 + r] = __float2bfloat16_rn(sc[r * 8 + t]);
        }
        if (lane == 0) CNT[tile] = 0;
      }
    } else if (!red_in_ring) {
      float* rw = red + ((tile - c * F) * NW + warp) * 128;
      rw[g * 8 + 2 * t4] = acc[0];
      rw[g * 8 + 2 * t4 + 1] = acc[1];
      rw[(g + 8) * 8 + 2 * t4] = acc[2];
      rw[(g + 8) * 8 + 2 * t4 + 1] = acc[3];
    } else {
      // Full-tile segment: hold it until the ring is free (a warp spans at most two tiles).
      if (nfull == 0) { hold[0] = acc[0]; hold[1] = acc[1]; hold[2] = acc[2]; hold[3] = acc[3]; htile[0] = tile; }
      else { hold[4] = acc[0]; hold[5] = acc[1]; hold[6] = acc[2]; hold[7] = acc[3]; htile[1] = tile; }
      nfull++;
    }
    acc[0] = acc[1] = acc[2] = acc[3] = 0.f;
    seg_start = i + 1;
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  if (red_in_ring) {
    for (int i = tid; i < F * NW * 128; i += 512) red[i] = 0.f;
    __syncthreads();
  }
  for (int h = 0; h < nfull; h++) {
    float* rw = red + ((htile[h] - c * F) * NW + warp) * 128;
    rw[g * 8 + 2 * t4] = hold[4 * h];
    rw[g * 8 + 2 * t4 + 1] = hold[4 * h + 1];
    rw[(g + 8) * 8 + 2 * t4] = hold[4 * h + 2];
    rw[(g + 8) * 8 + 2 * t4 + 1] = hold[4 * h + 3];
  }
  __syncthreads();
  // Full tiles: exceptions into warp 0's slot, then one output per thread.
  if (tid < F * CAPT && my_slot.x >= 0) {
    const int f = tid / CAPT, r = my_slot.x & 15, k = my_slot.x >> 4;
    const float wv = __int_as_float(my_slot.y), fake = __uint_as_float((base0 & 0xFF) << 23);
    for (int t = 0; t < m; t++) {
      const float xv = __bfloat162float(sx[t * XS + k]);
      atomicAdd(red + f * NW * 128 + r * 8 + t, xv * wv - xv * fake);
    }
  }
  __syncthreads();
  for (int o = tid; o < F * 16 * 8; o += 512) {
    const int f = o / 128, r = (o >> 3) & 15, t = o & 7;
    if (t < m) {
      float v = 0.f;
#pragma unroll
      for (int w = 0; w < NW; w++) v += red[(f * NW + w) * 128 + r * 8 + t];
      OUT[(size_t)t * N + (c * F + f) * 16 + r] = __float2bfloat16_rn(v);
    }
  }
}

template <bool COMP, int GD, bool RIR>
void launch_hgemv_t(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, torch::Tensor WS,
                    torch::Tensor CNT, int m, int N, int KU, int Kx, int G, int F, int Tdp, uint32_t base0, int dbg,
                    int smem) {
  auto kern = hgemv_kernel<COMP, GD, RIR>;
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  kern<<<G, 512, smem, at::cuda::getCurrentCUDAStream()>>>(
      W.data_ptr<uint8_t>(), (const __nv_bfloat16*)X.data_ptr(), (__nv_bfloat16*)OUT.data_ptr(),
      (const int2*)TSLOTS.data_ptr<int>(), WS.data_ptr<float>(), CNT.data_ptr<int>(), m, N, KU, Kx, F, Tdp, base0,
      dbg);
}

int64_t hgemv_smem(int64_t gd, bool comp, int64_t F, int64_t m, int64_t Kx) {
  return 16 * gd * (comp ? 1536 : 2048) + (F + 1) * 16 * 128 * 4 + (m + 1) * (Kx + 8) * 2;
}

void launch_hgemv(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, torch::Tensor WS,
                  torch::Tensor CNT, int64_t m, int64_t N, int64_t KU, int64_t Kx, int64_t G, int64_t F, int64_t Tdp,
                  int64_t base0, bool comp, int64_t gd, int64_t dbg) {
  const int smem = hgemv_smem(gd, comp, F, m, Kx);
#define HL(C, D) launch_hgemv_t<C, D, false>(W, X, OUT, TSLOTS, WS, CNT, m, N, KU, Kx, G, F, Tdp, (uint32_t)base0, dbg, smem)
  if (comp) { if (gd == 6) HL(true, 6); else if (gd == 4) HL(true, 4); else HL(true, 3); }
  else { if (gd == 6) HL(false, 6); else if (gd == 4) HL(false, 4); else HL(false, 3); }
#undef HL
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "hgemv launch failed: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------------------
// Router (FP32 out, raw BF16 hi | lo weights along K'). A CTA owns one 16-row tile and a
// K-slice; its 16 warps split the slice's units and reduce in shared memory; the S slices
// of a tile reduce through the workspace, the last arriver summing them in a fixed order.
// Tokens beyond 32 run as 32-token chunks along blockIdx.y.
template <int NT>
__global__ void __launch_bounds__(512, 1)
router_kernel(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, float* __restrict__ OUT,
              float* __restrict__ WS, unsigned* __restrict__ BAR, int m, int N, int KU, int Kx, int S) {
  constexpr int NW = 16, MP = NT * 8, WBW = 2048, NV = 4;
  constexpr size_t USTRIDE = 8 * WBW;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int tile = blockIdx.x / S, sl = blockIdx.x % S;
  const int ch = blockIdx.y;
  X += (size_t)ch * 32 * Kx;
  OUT += (size_t)ch * 32 * N;
  m = min(32, m - ch * 32);
  const int T = N / 16;
  float* wsb = WS + ((size_t)ch * T * S + (size_t)tile * S) * (16 * 32);
  unsigned* cnt = BAR + ch * 512 + tile;
  if (tid == 0) asm volatile("prefetch.global.L2 [%0];\n" ::"l"(cnt));
  const int ku0 = sl * KU / S, n = (sl + 1) * KU / S - ku0;
  const int nw = n > warp ? (n - warp + NW - 1) / NW : 0;

  for (int c = tid; c < n * MP * 8; c += 512) {
    int ul = c / (MP * 8), r = (c >> 3) % MP, chn = c & 7;
    bool valid = r < m;
    int kx = ((ku0 + ul) * KUNIT) % Kx + chn * 8;
    cp_async16(smem + (ul * MP + r) * 128 + ((chn ^ (r & 7)) << 4), X + (size_t)(valid ? r : 0) * Kx + kx,
               valid ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;\n" ::);
  const uint8_t* wp = W + ((size_t)((tile >> 3) * KU + ku0) * 8 + (tile & 7)) * WBW + lane * 16;
  uint4 buf[PF][NV];
#pragma unroll
  for (int p = 0; p < PF; p++)
    if (p < nw)
#pragma unroll
      for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + (warp + p * NW) * USTRIDE + v * 512);
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();

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
        const int u = warp + iu * NW;
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
#pragma unroll
        for (int j = 0; j < 4; j++) {
          uint32_t a[4] = {buf[p][j].x, buf[p][j].y, buf[p][j].z, buf[p][j].w};
#pragma unroll
          for (int nt = 0; nt < NT; nt++) mma16816(acc[nt], a, b[nt][2 * j], b[nt][2 * j + 1]);
        }
        if (iu + PF < nw)
#pragma unroll
          for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + (u + PF * NW) * USTRIDE + v * 512);
      }
    }
  }
  // Warps -> red[warp][MP][16]; then one CTA partial [m][16].
  __syncthreads();  // x slice no longer needed
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    float* p0 = red + (warp * MP + nt * 8 + 2 * t4) * 16 + g;
    p0[0] = acc[nt][0];
    p0[16] = acc[nt][1];
    p0[8] = acc[nt][2];
    p0[16 + 8] = acc[nt][3];
  }
  __syncthreads();
  float v = 0.f;
  const int t = tid >> 4, r = tid & 15;  // one (token, row) per thread, m <= 32
  if (t < m) {
#pragma unroll
    for (int w = 0; w < NW; w++) v += red[(w * MP + t) * 16 + r];
  }
  if (S == 1) {
    if (t < m) OUT[(size_t)t * N + tile * 16 + r] = v;
    return;
  }
  if (t < m) __stcg(wsb + (size_t)sl * (16 * 32) + tid, v);
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = atomicAdd(cnt, 1u) == (unsigned)S - 1;
    if (s_last) *cnt = 0;
    __threadfence();
  }
  __syncthreads();
  if (!s_last || t >= m) return;
  float sum = 0.f;
#pragma unroll 8
  for (int q = 0; q < S; q++) sum += __ldcg(wsb + (size_t)q * (16 * 32) + tid);
  OUT[(size_t)t * N + tile * 16 + r] = sum;
}

int64_t router_smem(int64_t m, int64_t KU, int64_t S) {
  const int MP = m <= 8 ? 8 : (m <= 16 ? 16 : 32);
  const int x = (KU + S - 1) / S * MP * 128, red = 16 * MP * 16 * 4;
  return x > red ? x : red;
}

void launch_router(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor BAR,
                   int64_t m, int64_t N, int64_t KU, int64_t Kx, int64_t S) {
  const int smem = router_smem(m, KU, S);
  dim3 grid((N / 16) * S, (m + 31) / 32);
  auto stream = at::cuda::getCurrentCUDAStream();
#define RL(NTV)                                                                                          \
  do {                                                                                                   \
    cudaFuncSetAttribute(router_kernel<NTV>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);         \
    router_kernel<NTV><<<grid, 512, smem, stream>>>(W.data_ptr<uint8_t>(), (const __nv_bfloat16*)X.data_ptr(), \
        OUT.data_ptr<float>(), WS.data_ptr<float>(), (unsigned*)BAR.data_ptr<int>(), m, N, KU, Kx, S);   \
  } while (0)
  if (m <= 8) RL(1); else if (m <= 16) RL(2); else RL(4);
#undef RL
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "router launch failed: ", cudaGetErrorString(err));
}

// ---------------------------------------------------------------------------------------
// m <= 32, data-parallel with x streamed in K-chunks. A CTA owns F whole 16-row tiles over
// all of K; warp w works on tile w % F. x moves through three shared-memory chunk buffers
// (one barrier per chunk) and every warp takes F units of each chunk, so all F tiles reuse
// each x chunk. No cross-CTA reduction: warps of a tile reduce in shared memory at the end.
constexpr int DCK = 16;  // units per x chunk

template <int NT, bool COMP>
__global__ void __launch_bounds__(512, 1)
dgemv_kernel(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, __nv_bfloat16* __restrict__ OUT,
             const int2* __restrict__ TSLOTS, int m, int N, int KU, int Kx, int F, uint32_t base0) {
  constexpr int NW = 16, MP = NT * 8;
  constexpr int WBW = COMP ? 1536 : 2048;
  constexpr int NV = COMP ? 3 : 4;
  constexpr size_t USTRIDE = 8 * WBW;
  constexpr int XCB = DCK * MP * 128;  // bytes per x chunk
  extern __shared__ __align__(128) uint8_t smem[];
  float* corr = reinterpret_cast<float*>(smem + 3 * XCB);  // [F][16][MP] exception corrections
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int tl = warp % F, wsub = warp / F, WT = NW / F;  // WT warps per tile
  const int tile = blockIdx.x * F + tl;
  const int nch = KU / DCK, nu = KU / WT;                   // units per warp: F per chunk
  const uint8_t* wbase = W + ((size_t)((tile >> 3) * KU) * 8 + (tile & 7)) * WBW + lane * 16;
  // Unit i of this warp: chunk i / F, ku = chunk * DCK + wsub + (i % F) * WT.
  auto unit_ku = [&](int i) { return (i / F) * DCK + wsub + (i % F) * WT; };

  auto load_chunk = [&](int ch) {
    uint8_t* dst = smem + (ch % 3) * XCB;
    for (int c = tid; c < DCK * MP * 8; c += 512) {
      int ul = c / (MP * 8), r = (c >> 3) % MP, chn = c & 7;
      bool valid = r < m;
      const __nv_bfloat16* src = X + (size_t)(valid ? r : 0) * Kx + (ch * DCK + ul) * KUNIT + chn * 8;
      cp_async16(dst + (ul * MP + r) * 128 + ((chn ^ (r & 7)) << 4), src, valid ? 16 : 0);
    }
  };
  load_chunk(0);
  asm volatile("cp.async.commit_group;\n" ::);
  if (nch > 1) load_chunk(1);
  asm volatile("cp.async.commit_group;\n" ::);

  uint4 buf[PF][NV];
#pragma unroll
  for (int p = 0; p < PF; p++)
    if (p < nu)
#pragma unroll
      for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wbase + unit_ku(p) * USTRIDE + v * 512);
  // Exception slots of this CTA's tiles (one per thread), used while their x chunk is resident.
  const int2 my_slot = TSLOTS[(size_t)blockIdx.x * F * CAPT + min(tid, F * CAPT - 1)];
  for (int i = tid; i < F * 16 * MP; i += 512) corr[i] = 0.f;

  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

  for (int ch = 0; ch < nch; ch++) {
    asm volatile("cp.async.wait_group 1;\n" ::);
    __syncthreads();
    if (ch + 2 < nch) load_chunk(ch + 2);
    asm volatile("cp.async.commit_group;\n" ::);
    const uint8_t* xc = smem + (ch % 3) * XCB;
    if (tid < F * CAPT && my_slot.x >= 0 && (my_slot.x >> 4) / (DCK * KUNIT) == ch) {
      const int f = tid / CAPT, r = my_slot.x & 15, kl = (my_slot.x >> 4) - ch * DCK * KUNIT;
      const int ul = kl >> 6, kin = kl & 63;
      const float wv = __int_as_float(my_slot.y), fake = __uint_as_float((base0 & 0xFF) << 23);
      for (int t = 0; t < m; t++) {
        const float xv = __bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(
            xc + (ul * MP + t) * 128 + (((kin >> 3) ^ (t & 7)) << 4) + (kin & 7) * 2));
        atomicAdd(corr + (f * 16 + r) * MP + t, xv * wv - xv * fake);
      }
    }
    // F units per warp per chunk; the register ring spans chunks (PF is a multiple of F or not).
    for (int j = 0; j < F; j++) {
      const int i = ch * F + j;
      const int p = i % PF;
      const int ul = wsub + j * WT;
      const uint8_t* sX = xc + ul * MP * 128;
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
      uint4 q4[NV];
#pragma unroll
      for (int v = 0; v < NV; v++) {
        // Register ring slot p (runtime): select through a small unrolled switch.
#pragma unroll
        for (int pp = 0; pp < PF; pp++)
          if (pp == p) q4[v] = buf[pp][v];
      }
      if (i + PF < nu) {
        const uint8_t* src = wbase + unit_ku(i + PF) * USTRIDE;
#pragma unroll
        for (int pp = 0; pp < PF; pp++)
          if (pp == p)
#pragma unroll
            for (int v = 0; v < NV; v++) buf[pp][v] = ldg_stream(src + v * 512);
      }
      if constexpr (COMP) {
        uint32_t sw[8] = {q4[0].x, q4[0].y, q4[0].z, q4[0].w, q4[1].x, q4[1].y, q4[1].z, q4[1].w};
        uint32_t cw[4] = {q4[2].x, q4[2].y, q4[2].z, q4[2].w};
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          uint32_t a[4];
          decode4(sw[2 * jj], cw[jj] & 0x0F0F0F0Fu, base0, a[0], a[1]);
          decode4(sw[2 * jj + 1], (cw[jj] >> 4) & 0x0F0F0F0Fu, base0, a[2], a[3]);
#pragma unroll
          for (int nt = 0; nt < NT; nt++) mma16816(acc[nt], a, b[nt][2 * jj], b[nt][2 * jj + 1]);
        }
      } else {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          uint32_t a[4] = {q4[jj].x, q4[jj].y, q4[jj].z, q4[jj].w};
#pragma unroll
          for (int nt = 0; nt < NT; nt++) mma16816(acc[nt], a, b[nt][2 * jj], b[nt][2 * jj + 1]);
        }
      }
    }
  }
  asm volatile("cp.async.wait_group 0;\n" ::);
  __syncthreads();
  // red[warp][MP][16] over the x buffers.
  float* red = reinterpret_cast<float*>(smem);
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    float* p0 = red + (warp * MP + nt * 8 + 2 * t4) * 16 + g;
    p0[0] = acc[nt][0];
    p0[16] = acc[nt][1];
    p0[8] = acc[nt][2];
    p0[16 + 8] = acc[nt][3];
  }
  __syncthreads();
  for (int o = tid; o < F * 16 * m; o += 512) {
    const int f = o / (16 * m), rem = o % (16 * m), t = rem / 16, r = rem % 16;
    float v = corr[(f * 16 + r) * MP + t];
    for (int s2 = 0; s2 < WT; s2++) v += red[((f + s2 * F) * MP + t) * 16 + r];
    OUT[(size_t)t * N + (blockIdx.x * F + f) * 16 + r] = __float2bfloat16_rn(v);
  }
}

int64_t dgemv_smem(int64_t m, int64_t F) {
  const int MP = m <= 8 ? 8 : (m <= 16 ? 16 : 32);
  const int x = 3 * DCK * MP * 128, red = 16 * MP * 16 * 4;
  return (x > red ? x : red) + F * 16 * MP * 4;
}

void launch_dgemv(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, int64_t m, int64_t N,
                  int64_t KU, int64_t Kx, int64_t F, int64_t base0, bool comp) {
  const int smem = dgemv_smem(m, F);
  const int grid = N / (16 * F);
  auto stream = at::cuda::getCurrentCUDAStream();
#define DL(NTV, C)                                                                                        \
  do {                                                                                                    \
    cudaFuncSetAttribute(dgemv_kernel<NTV, C>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);        \
    dgemv_kernel<NTV, C><<<grid, 512, smem, stream>>>(W.data_ptr<uint8_t>(), (const __nv_bfloat16*)X.data_ptr(), \
        (__nv_bfloat16*)OUT.data_ptr(), (const int2*)TSLOTS.data_ptr<int>(), m, N, KU, Kx, F, (uint32_t)base0); \
  } while (0)
  if (comp) { if (m <= 8) DL(1, true); else if (m <= 16) DL(2, true); else DL(4, true); }
  else { if (m <= 8) DL(1, false); else if (m <= 16) DL(2, false); else DL(4, false); }
#undef DL
  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "dgemv launch failed: ", cudaGetErrorString(err));
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
  cfg.gridDim = dim3((N / ROWS) * S, (m + 31) / 32);
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
int64_t clusters_t(int KU, int S, int C, int N, int m) {
  auto kern = skinny_kernel<NT, COMP, OUTF32, WPR>;
  const int smem = smem_bytes<NT, COMP, OUTF32>(KU, S);
  int optin = 0;
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  if (smem > optin) return 0;
  cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  cudaGetLastError();
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((N / ROWS) * S, (m + 31) / 32);
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
#define CALL(NTV, CP, F) r = clusters_t<NTV, CP, F>(KU, S, C, N, m)
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
void launch_hgemv(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, torch::Tensor WS,
                  torch::Tensor CNT, int64_t m, int64_t N, int64_t KU, int64_t Kx, int64_t G, int64_t F, int64_t Tdp,
                  int64_t base0, bool comp, int64_t gd, int64_t dbg);
int64_t hgemv_smem(int64_t gd, bool comp, int64_t F, int64_t m, int64_t Kx);
void launch_router(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor BAR,
                   int64_t m, int64_t N, int64_t KU, int64_t Kx, int64_t S);
int64_t router_smem(int64_t m, int64_t KU, int64_t S);
void launch_dgemv(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor TSLOTS, int64_t m, int64_t N,
                  int64_t KU, int64_t Kx, int64_t F, int64_t base0, bool comp);
int64_t dgemv_smem(int64_t m, int64_t F);
"""

ROWS, KUNIT = 128, 64
MAX_M = 32
ROUTER_KERNEL_MAX_M = 32    # the dedicated router kernel; the split kernel above
ROUTER_MAX_CTAS = 1024
# Shapes where cuBLAS runs faster on W stored transposed ([K, N]).
CUBLAS_WT = {("qkv_gate", 8), ("qkv_gate", 16), ("qkv_gate", 32), ("qkv_gate", 128), ("pooler", 8),
             ("pooler", 16), ("pooler", 32), ("pooler", 128), ("pooler", 256), ("unpooler", 8), ("unpooler", 16),
             ("unpooler", 32), ("unpooler", 128), ("o", 256)}
WSTRIDE = 32

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="okc_bf16_proj_v8",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["launch", "max_clusters", "get_ts", "launch_gemv", "launch_hgemv", "hgemv_smem", "launch_router",
                       "router_smem", "launch_dgemv", "dgemv_smem"],
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
GEMV = set()
HGEMV = {("qkv_gate", 1), ("pooler", 1), ("unpooler", 1)}
DGEMV = set()
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
    chunks = -(-m // 32)
    if chunks > 8 or nrb * chunks > state["sms"]:
        return None
    s = max(1, min(ku, state["sms"] // (nrb * chunks)))
    plan = None
    if -(-ku // s) * mp * 128 + mp * ROWS * 4 <= state["smem"]:
        plan = (s, 1) if s == 1 else ((s, -1) if s <= 16 and (s - 1) * ROWS * m * 4 <= (128 << 10) else (s, 0))
        if 1 < s <= 16 and _module().max_clusters(m, p["comp"], layer == "router", ku, s, s, n) * s >= nrb * s * chunks:
            plan = (s, s)
    state["plans"][key] = plan
    return plan




# ---------------------------------------------------------------------------------------
# v2 compressed GEMV (fixed-base decode, in-register exception patches), m <= 32.
CUDA_SRC_Z = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#define UNIT_BYTES 1536

struct Params {
  const uint8_t* W;
  const __nv_bfloat16* X;
  void* OUT;
  const uint4* slots;  // [T][S][NW][32][2] per-lane exception patches (4 per lane)
  float4* ws;          // split-K partials
  int* sem;            // split-K counters, one per tile
  int m, N, KU, KUh, Kx, S;
  float scale0, scale1;
};

__device__ __forceinline__ uint4 ldg_w(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
// One k16 step (8 weights per lane) into an A fragment, fixed exponent base.
__device__ __forceinline__ void decode(uint32_t L0, uint32_t L1, uint32_t nw, uint32_t* a) {
  const uint32_t lowc = 0x07070707u, basec = 0x38383838u, sgn = 0x80808080u;
  const uint32_t h03 = ((nw << 4) & sgn) | (nw & lowc) | basec;
  const uint32_t h47 = (nw & sgn) | ((nw >> 4) & lowc) | basec;
  a[0] = __byte_perm(L0, h03, 0x5140);
  a[1] = __byte_perm(L0, h03, 0x7362);
  a[2] = __byte_perm(L1, h47, 0x5140);
  a[3] = __byte_perm(L1, h47, 0x7362);
}

template <int NW, int U, int NB, bool RT, int MINB>
__global__ void __launch_bounds__(NW * 32, MINB) gemvz(const Params p) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, tq = lane & 3;
  const int tile = blockIdx.x, sidx = blockIdx.y, S = p.S;
  const int kus = p.KU / S, kuw = kus / NW;
  const int u0 = sidx * kus + warp * kuw;   // this warp's first unit (contiguous range)
  const int m = p.m;
  const uint8_t* wp = p.W + ((size_t)tile * p.KU + u0) * UNIT_BYTES + lane * 16;
  // Exception slots: issued first (tiny), parked in shared memory once the weights are in flight.
  constexpr int SH16 = NW * (NB * 32 > 64 ? NB * 32 : 64);
  __shared__ uint4 shbuf[SH16];
  uint4 (*sslot)[32][2] = (uint4 (*)[32][2])shbuf;
  const uint4* sp = p.slots + (((size_t)tile * S + sidx) * NW + warp) * 64;
  const uint4 sl0 = sp[lane * 2], sl1 = sp[lane * 2 + 1];
  const __nv_bfloat16* xr[NB];
#pragma unroll
  for (int nb = 0; nb < NB; nb++) {
    const int tok = min(nb * 8 + g, m - 1);
    xr[nb] = p.X + (size_t)tok * p.Kx + tq * 8;
  }
  float acc[NB][4], acc2[NB][4];
#pragma unroll
  for (int nb = 0; nb < NB; nb++)
#pragma unroll
    for (int r = 0; r < 4; r++) { acc[nb][r] = 0.f; acc2[nb][r] = 0.f; }

  uint4 a[U][3];
#pragma unroll
  for (int v = 0; v < U; v++) {
    const uint8_t* q = wp + (size_t)min(v, kuw - 1) * UNIT_BYTES;
    a[v][0] = ldg_w(q); a[v][1] = ldg_w(q + 512); a[v][2] = ldg_w(q + 1024);
  }
  sslot[warp][lane][0] = sl0;
  sslot[warp][lane][1] = sl1;
  uint32_t emask;
  {
    uint32_t bm = 0;
    const uint4 e0 = sl0, e1 = sl1;
    const uint32_t ms[4] = {e0.x, e0.z, e1.x, e1.z};
#pragma unroll
    for (int q = 0; q < 4; q++)
      if (ms[q] & 0x2000) bm |= 1u << ((ms[q] & 0xff) - u0);
    emask = __reduce_or_sync(0xffffffffu, bm);
  }
  for (int i = 0; i < kuw; i += U) {
    // Next batch's weights, in flight while this batch is decoded.
    uint4 na[U][3];
#pragma unroll
    for (int v = 0; v < U; v++) {
      const uint8_t* q = wp + (size_t)min(i + U + v, kuw - 1) * UNIT_BYTES;
      na[v][0] = ldg_w(q); na[v][1] = ldg_w(q + 512); na[v][2] = ldg_w(q + 1024);
    }
#pragma unroll
    for (int v = 0; v < U; v++) {
      if (i + v >= kuw) break;
      const int u = u0 + i + v;
      const bool lo_half = RT && u >= p.KUh;
      const int xk = (RT && u >= p.KUh ? u - p.KUh : u) * 64;
      uint4 xv[NB][2];
#pragma unroll
      for (int nb = 0; nb < NB; nb++) {
        xv[nb][0] = __ldg((const uint4*)(xr[nb] + xk));
        xv[nb][1] = __ldg((const uint4*)(xr[nb] + xk + 32));
      }
      const bool patch = (emask >> (i + v)) & 1;
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const int pp = s >> 1, h = s & 1;
        const uint4 lw = a[v][pp];
        const uint32_t nw = s == 0 ? a[v][2].x : (s == 1 ? a[v][2].y : (s == 2 ? a[v][2].z : a[v][2].w));
        uint32_t f[4];
        decode(h ? lw.z : lw.x, h ? lw.w : lw.y, nw, f);
        if (patch) {
          const uint4 e0 = sslot[warp][lane][0], e1 = sslot[warp][lane][1];
          const uint32_t ms[4] = {e0.x, e0.z, e1.x, e1.z}, evs[4] = {e0.y, e0.w, e1.y, e1.w};
#pragma unroll
          for (int q = 0; q < 4; q++) {
            if ((ms[q] & 0x23ff) == (0x2000u | ((uint32_t)s << 8) | (uint32_t)u)) {
              const uint32_t sh = (ms[q] >> 12) & 1 ? 16 : 0, rr = (ms[q] >> 10) & 3;
              const uint32_t keep = ~(0xFFFFu << sh), put = evs[q] << sh;
              f[0] = rr == 0 ? ((f[0] & keep) | put) : f[0];
              f[1] = rr == 1 ? ((f[1] & keep) | put) : f[1];
              f[2] = rr == 2 ? ((f[2] & keep) | put) : f[2];
              f[3] = rr == 3 ? ((f[3] & keep) | put) : f[3];
            }
          }
        }
#pragma unroll
        for (int nb = 0; nb < NB; nb++) {
          const uint4 x4 = xv[nb][pp];
          if (RT) {
            if (lo_half) mma(acc2[nb], f, h ? x4.z : x4.x, h ? x4.w : x4.y);
            else mma(acc[nb], f, h ? x4.z : x4.x, h ? x4.w : x4.y);
          } else {
            mma(h ? acc2[nb] : acc[nb], f, h ? x4.z : x4.x, h ? x4.w : x4.y);
          }
        }
      }
    }
#pragma unroll
    for (int v = 0; v < U; v++) { a[v][0] = na[v][0]; a[v][1] = na[v][1]; a[v][2] = na[v][2]; }
  }

  // Warps add their sums through shared memory.
  __syncthreads();  // slot table dead: the buffer now holds partial sums
  float4 (*red)[NB * 32] = (float4 (*)[NB * 32])shbuf;
  __shared__ int s_last;
  const float s0 = p.scale0, s1 = RT ? p.scale1 : p.scale0;
#pragma unroll
  for (int nb = 0; nb < NB; nb++)
    red[warp][nb * 32 + lane] = make_float4(acc[nb][0] * s0 + acc2[nb][0] * s1, acc[nb][1] * s0 + acc2[nb][1] * s1,
                                            acc[nb][2] * s0 + acc2[nb][2] * s1, acc[nb][3] * s0 + acc2[nb][3] * s1);
  __syncthreads();
  float4 r = make_float4(0.f, 0.f, 0.f, 0.f);
  const bool active = tid < NB * 32;
  if (active) {
#pragma unroll
    for (int wi = 0; wi < NW; wi++) {
      const float4 v = red[wi][tid];
      r.x += v.x; r.y += v.y; r.z += v.z; r.w += v.w;
    }
  }
  if (S > 1) {
    if (active) p.ws[((size_t)tile * S + sidx) * (NB * 32) + tid] = r;
    __threadfence();
    __syncthreads();
    if (tid == 0) s_last = atomicAdd(&p.sem[tile], 1) == S - 1;
    __syncthreads();
    if (!s_last) return;
    __threadfence();
    if (active) {
      r = make_float4(0.f, 0.f, 0.f, 0.f);
      for (int s = 0; s < S; s++) {
        const float4 v = __ldcg(p.ws + ((size_t)tile * S + s) * (NB * 32) + tid);
        r.x += v.x; r.y += v.y; r.z += v.z; r.w += v.w;
      }
    }
    if (tid == 0) p.sem[tile] = 0;
  }
  if (active) {
    const int nb = tid >> 5;
    const float rr[4] = {r.x, r.y, r.z, r.w};
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int tok = nb * 8 + 2 * tq + (q & 1);
      const int row = tile * 16 + g + ((q >> 1) << 3);
      if (tok < m) {
        if (RT) ((float*)p.OUT)[(size_t)tok * p.N + row] = rr[q];
        else ((__nv_bfloat16*)p.OUT)[(size_t)tok * p.N + row] = __float2bfloat16(rr[q]);
      }
    }
  }
}

template <int NW, int U, int NB, bool RT, int MINB>
static void go(const Params& p, cudaStream_t st) {
  gemvz<NW, U, NB, RT, MINB><<<dim3(p.N / 16, p.S), NW * 32, 0, st>>>(p);
}

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor SLOTS, torch::Tensor WS,
            torch::Tensor SEM, int64_t m, int64_t N, int64_t KU, int64_t KUh, int64_t Kx, double scale0,
            double scale1, int64_t S, int64_t NW, int64_t U, int64_t MB) {
  Params p;
  p.W = W.data_ptr<uint8_t>();
  p.X = (const __nv_bfloat16*)X.data_ptr();
  p.OUT = OUT.data_ptr();
  p.slots = (const uint4*)SLOTS.data_ptr<int>();
  p.ws = (float4*)WS.data_ptr<float>();
  p.sem = SEM.data_ptr<int>();
  p.m = m; p.N = N; p.KU = KU; p.KUh = KUh; p.Kx = Kx; p.S = S;
  p.scale0 = (float)scale0; p.scale1 = (float)scale1;
  const int NB = m <= 8 ? 1 : (m <= 16 ? 2 : 4);
  const bool rt = OUT.scalar_type() == torch::kFloat32;
  TORCH_CHECK((KU / S) % NW == 0 && m <= 32, "bad config");
  auto st = at::cuda::getCurrentCUDAStream();
#define G(NW_, U_, NB_, MB_) \
  if (NW == NW_ && U == U_ && NB == NB_ && MB == MB_) { if (rt) go<NW_, U_, NB_, true, MB_>(p, st); else go<NW_, U_, NB_, false, MB_>(p, st); return; }
  G(4, 2, 1, 0) G(4, 2, 1, 8) G(4, 2, 1, 12) G(8, 2, 1, 0) G(8, 2, 1, 4) G(8, 2, 1, 6) G(16, 2, 1, 0) G(16, 2, 1, 2) G(16, 2, 1, 3)
  G(8, 4, 1, 0) G(8, 4, 1, 3) G(4, 4, 1, 0) G(4, 4, 1, 6)
  G(4, 2, 2, 0) G(8, 2, 2, 0) G(8, 2, 2, 3) G(16, 2, 2, 0) G(4, 2, 4, 0) G(8, 2, 4, 0) G(8, 2, 4, 2) G(16, 2, 4, 0)
  TORCH_CHECK(false, "unsupported config");
}
"""

CPP_SRC_Z = """
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor SLOTS, torch::Tensor WS,
            torch::Tensor SEM, int64_t m, int64_t N, int64_t KU, int64_t KUh, int64_t Kx, double scale0,
            double scale1, int64_t S, int64_t NW, int64_t U, int64_t MB);
"""

_extz = None


def _modz():
    global _extz
    if _extz is None:
        _extz = load_inline(name="okc_gemvz_hy_v1", cpp_sources=CPP_SRC_Z, cuda_sources=CUDA_SRC_Z, functions=["launch"],
                            extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a"], verbose=False)
    return _extz


FIXED_BASE = 56


def _pack(w: torch.Tensor):
    """BF16 [N, K] -> (packed uint8 [T, KU, 1536], base, exception rows, cols, scaled BF16 bits)."""
    n, k = w.shape
    b = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    lo = b & 0xFF
    e1 = (b >> 8) & 0x7F
    s = b >> 15
    hist = torch.bincount(e1.flatten(), minlength=128).cpu()
    cover = torch.stack([hist[B:B + 8].sum() for B in range(1, 121)])
    base = int(cover.argmax()) + 1
    exc = (e1 < base) | (e1 > base + 7)
    nib = torch.where(exc, 0, (s << 3) | (e1 - base))
    lo = torch.where(exc, 0, lo)
    T, KU = n // 16, k // 64
    lv = lo.view(T, 2, 8, KU, 2, 4, 2, 2, 2).permute(0, 3, 4, 2, 5, 6, 7, 1, 8).reshape(T, KU, 1024)
    nv = nib.view(T, 2, 8, KU, 2, 4, 2, 2, 2).permute(0, 3, 2, 5, 4, 6, 7, 1, 8)
    nb = (nv[:, :, :, :, :, :, 0] | (nv[:, :, :, :, :, :, 1] << 4)).reshape(T, KU, 512)
    packed = torch.cat([lv, nb], -1).to(torch.uint8).contiguous()
    idx = exc.nonzero()
    rows, cols = idx[:, 0], idx[:, 1]
    val = (w[rows, cols].double() * 4.0 ** (FIXED_BASE - base)).to(torch.bfloat16)
    assert torch.equal(val.double() / 4.0 ** (FIXED_BASE - base), w[rows, cols].double()), "inexact scaling"
    vbits = val.view(torch.int16).to(torch.int32) & 0xFFFF
    return packed, base, rows, cols, vbits


def _zlayer(w: torch.Tensor):
    n, k = w.shape
    packed, base, rows, cols, vbits = _pack(w)
    sc = 4.0 ** (base - FIXED_BASE)
    return {"w": packed.flatten(), "scale0": sc, "scale1": sc, "ku": k // 64, "kuh": 0, "kx": k,
            "rows": rows, "kcols": cols, "vbits": vbits, "slots": {}}


def _zrouter(r: torch.Tensor):
    n, k = r.shape
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    ph, bh, rh, ch, vh = _pack(hi)
    pl, bl, rl, cl, vl = _pack(lo)
    packed = torch.cat([ph, pl], 1).contiguous()
    return {"w": packed.flatten(), "scale0": 4.0 ** (bh - FIXED_BASE), "scale1": 4.0 ** (bl - FIXED_BASE),
            "ku": 2 * k // 64, "kuh": k // 64, "kx": k, "rows": torch.cat([rh, rl]),
            "kcols": torch.cat([ch, cl + k]), "vbits": torch.cat([vh, vl]), "slots": {}}


def _slot_table(p, n: int, S: int, NW: int):
    """Per-lane exception patches: int32 [T, S, NW, 32, 8], four per lane, each
    (unit | step << 8 | reg << 10 | half << 12 | valid << 13, scaled BF16 bits)."""
    key = (n, S, NW)
    if key in p["slots"]:
        return p["slots"][key]
    rows, kc, vb = p["rows"], p["kcols"], p["vbits"]
    keep = rows < n
    rows, kc, vb = rows[keep], kc[keep], vb[keep]
    T = n // 16
    kus = p["ku"] // S
    kuw = kus // NW
    tile = rows // 16
    r16 = rows % 16
    rh, g = r16 // 8, r16 % 8
    u = kc // 64
    c = kc % 64
    pp, tq, h, cc2, cc1 = c // 32, (c // 8) % 4, (c // 4) % 2, (c // 2) % 2, c % 2
    lane = g * 4 + tq
    step = pp * 2 + h
    reg = rh + 2 * cc2
    sidx = u // kus
    warp = (u - sidx * kus) // kuw
    lin = ((tile * S + sidx) * NW + warp) * 32 + lane
    order = torch.argsort(lin, stable=True)
    lin, u, step, reg, cc1, vb = lin[order], u[order], step[order], reg[order], cc1[order], vb[order]
    counts = torch.bincount(lin, minlength=T * S * NW * 32)
    assert counts.numel() == 0 or int(counts.max()) <= 4, "more than 4 exceptions in one lane"
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(lin.numel(), device=lin.device) - start[lin]
    tab = torch.zeros((T * S * NW * 32, 8), dtype=torch.int32, device=rows.device)
    meta = (u | (step << 8) | (reg << 10) | (cc1 << 12) | (1 << 13)).to(torch.int32)
    tab[lin, pos * 2] = meta
    tab[lin, pos * 2 + 1] = vb.to(torch.int32)
    p["slots"][key] = tab.contiguous()
    return tab



# (layer, m bucket) -> (S, NW, U, MINB) for the v2 GEMV; everything else takes the original path.
ZCFG = {
    ("qkv_gate", 1): (1, 8, 2, 0), ("unpooler", 1): (1, 16, 2, 0),
    ("qkv_gate", 8): (1, 8, 2, 0), ("pooler", 8): (1, 8, 2, 0), ("unpooler", 8): (1, 16, 2, 0),
}


def _zbucket(m: int) -> int:
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))


def prepare(weights: Weights):
    dev = weights["qkv_gate"].device
    _module()
    state = {"raw": {name: w.contiguous() for name, w in weights.items() if name != "router"}}
    state["raw_t"] = {name: w.t().contiguous() for name, w in state["raw"].items()}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["router_cat"] = torch.cat([hi, lo], 1).contiguous()
    try:
        import deep_gemm
        state["deep_gemm"] = deep_gemm if torch.cuda.get_device_capability(dev)[0] == 9 else None
    except ImportError:
        state["deep_gemm"] = None
    packs = {name: _layer_state(weights[name].contiguous(), COMPRESS[name]) for name in COMPRESS}
    packs["router"] = _router_state(r, ROUTER_COMPRESS)
    state["packs"] = packs
    props = torch.cuda.get_device_properties(dev)
    state["ws"] = torch.zeros(props.multi_processor_count * ROWS * 32, dtype=torch.float32, device=dev)
    state["bar"] = torch.zeros(2 * 64 * 8, dtype=torch.int32, device=dev)
    state["sms"] = props.multi_processor_count
    state["smem"] = props.shared_memory_per_block_optin - 1024
    state["plans"] = {}
    state["hws"] = torch.zeros(8192 // 16 * 128, dtype=torch.float32, device=dev)
    state["hcnt"] = torch.zeros(8192 // 16, dtype=torch.int32, device=dev)
    state["rws"] = torch.zeros(ROUTER_MAX_CTAS * 16 * 32, dtype=torch.float32, device=dev)
    state["rbar"] = torch.zeros(8 * 512, dtype=torch.int32, device=dev)
    _modz()
    state["zpacks"] = {name: _zlayer(weights[name]) for name in ("qkv_gate", "o", "pooler", "unpooler")}
    for (layer, mb), (S, NW, U, MB) in ZCFG.items():
        n = 1024 if layer == "unpooler" and mb <= 8 else weights[layer].shape[0]
        _slot_table(state["zpacks"][layer], n, S, NW)
    state["zws"] = torch.zeros(8192 * 64 * 4, dtype=torch.float32, device=dev)
    state["zsem"] = torch.zeros(8192 // 16, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    return state


def _hplan(state, layer: str, m: int, n: int):
    """(G, F, Tdp, ring depth) for the hybrid kernel: G CTAs (one per SM) each own F whole
    tiles; the T - G * F leftover tiles are split across all of them. None if x does not fit."""
    key = ("h", layer, m, n)
    if key in state["plans"]:
        return state["plans"][key]
    p = state["packs"][layer]
    T, sms = n // 16, state["sms"]
    # Whole tiles only: F tiles per CTA, the fewest that fit one CTA per SM (a 16-warp CTA
    # can pull about twice its share, so idle SMs cost less than splitting tiles).
    F = -(-T // sms)
    while T % F:
        F += 1
    G = T // F
    plan = None
    for gd in (6, 4, 3):
        if _module().hgemv_smem(gd, p["comp"], F, m, p["kx"]) <= state["smem"]:
            plan = (G, F, G * F, gd)
            break
    state["plans"][key] = plan
    return plan


def _rplan(state, m: int, n: int):
    """K-slices per router tile: fill the SMs (one CTA each) with x's slice in shared memory."""
    key = ("r", m, n)
    if key not in state["plans"]:
        # No CTA waits on another (the last arriver sums), so more than one wave is fine.
        T, ku, chunks = n // 16, state["packs"]["router"]["ku"], -(-m // 32)
        s = max(1, state["sms"] // (T * chunks))
        while s <= ku and _module().router_smem(m, ku, s) > state["smem"]:
            s += 1
        state["plans"][key] = s if s <= ku and T * chunks * s <= ROUTER_MAX_CTAS else None
    return state["plans"][key]


def _router_big(x, state):
    """m > 32: one BF16 GEMM over K' = [x | x] against [hi | lo], FP32 accumulation and output."""
    m = x.shape[0]
    if state.get("deep_gemm") is not None:
        out = torch.empty((m, state["router_cat"].shape[0]), dtype=torch.float32, device=x.device)
        state["deep_gemm"].bf16_gemm_nt(torch.cat([x, x], 1), state["router_cat"], out)
        return out
    acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
    acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
    return acc


def _cublas(x, state, layer: str, m: int, n: int):
    if (layer, m) in CUBLAS_WT:
        return torch.mm(x, state["raw_t"][layer][:, :n])
    return torch.mm(x, state["raw"][layer][:n].t())


def _dplan(state, layer: str, m: int, n: int):
    """Tiles per CTA for the chunked data-parallel kernel: the fewest that fit one CTA per SM."""
    key = ("d", layer, m, n)
    if key not in state["plans"]:
        T, ku = n // 16, state["packs"][layer]["ku"]
        F = next((f for f in (1, 2, 4, 8, 16) if T % f == 0 and T // f <= state["sms"]), None)
        ok = F and ku % 16 == 0 and _module().dgemv_smem(m, F) <= state["smem"]
        state["plans"][key] = F if ok else None
    return state["plans"][key]


def kernel_fn(data: Inputs, state, plan=None, dbg=0, hplan=None, rs=None, df=None) -> torch.Tensor:
    x = data["x"]
    cfg = data["config"]
    m, n, layer = cfg["m"], cfg["n"], cfg["layer"]
    zc = ZCFG.get((layer, _zbucket(m))) if m <= 32 else None
    if zc is not None:
        S, NW, U, MB = zc
        p = state["zpacks"][layer]
        tab = _slot_table(p, n, S, NW)
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _modz().launch(p["w"], x.contiguous(), out, tab, state["zws"], state["zsem"], m, n, p["ku"], p["kuh"], p["kx"],
                       p["scale0"], p["scale1"], S, NW, U, MB)
        return out
    if m > MAX_M:
        return _router_big(x, state) if layer == "router" else _cublas(x, state, layer, m, n)
    p = state["packs"][layer]
    if layer == "router" and plan is None and m <= ROUTER_KERNEL_MAX_M and not p["comp"]:
        rs = rs or _rplan(state, m, n)
        if rs:
            assert (n // 16) * -(-m // 32) * rs <= ROUTER_MAX_CTAS, "router plan exceeds its workspace"
            out = torch.empty((m, n), dtype=torch.float32, device=x.device)
            _module().launch_router(p["w"], x.contiguous(), out, state["rws"], state["rbar"], m, n, p["ku"], p["kx"],
                                    rs)
            return out
    if m <= 32 and layer in DGEMV and plan is None and hplan is None and "tslots" in p:
        F = df or _dplan(state, layer, m, n)
        if F:
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _module().launch_dgemv(p["w"], x.contiguous(), out, p["tslots"], m, n, p["ku"], p["kx"], F, p["base"],
                                   p["comp"])
            return out
    if m <= 8 and (layer, m) in HGEMV and plan is None and "tslots" in p:
        hp = hplan or _hplan(state, layer, m, n)
        if hp is not None:
            G, F, Tdp, gd = hp
            out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
            _module().launch_hgemv(p["w"], x.contiguous(), out, p["tslots"], state["hws"], state["hcnt"], m, n,
                                   p["ku"], p["kx"], G, F, Tdp, p["base"], p["comp"], gd, dbg)
            return out
    if m <= 8 and layer in GEMV and plan is None and "tslots" in p:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
        _module().launch_gemv(p["w"], x.contiguous(), out, p["tslots"], m, n, p["ku"], p["kx"], GEMV_TPC[layer],
                              p["base"], p["comp"], dbg)
        return out
    plan = plan or (_plan(state, layer, m, n) if layer in OURS else None)
    if plan is None:
        if layer == "router":
            return _router_big(x, state)
        return _cublas(x, state, layer, m, n)
    s, c = plan
    assert (n // ROWS) * s * -(-m // 32) <= state["sms"], "split plan exceeds its workspace"
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().launch(p["w"], x.contiguous(), out, state["ws"], state["bar"], p["slots"], m, n, p["ku"],
                     p["kx"], p["base"], p["base1"], p["comp"], s, c, dbg)
    return out
