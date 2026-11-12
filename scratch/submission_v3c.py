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

// Grid: (N / 128) row blocks x S K-slices, slice-fastest; clusters of C consecutive slices.
// Warp w handles row tile w % 8 and every WPR-th unit of the slice starting at w / 8.
template <int NT, bool COMP, bool OUTF32, int WPR>
__global__ void __launch_bounds__(256 * WPR, 1)
skinny_kernel(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, void* __restrict__ OUT,
              float* __restrict__ WS, int* __restrict__ CNT, const int* __restrict__ EPTR,
              const int2* __restrict__ EXC, int m, int N, int KU, int Kx, int S, int C, uint32_t base0,
              uint32_t base1, int dbg) {
  constexpr int THREADS = 256 * WPR;
  constexpr int MP = NT * 8;
  constexpr int WBW = COMP ? 1536 : 2048;
  constexpr int NV = COMP ? 3 : 4;
  constexpr size_t USTRIDE = 8 * WBW;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;
  __shared__ int2 s_exc[256 * WPR];

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int rt = warp & 7, half = warp >> 3;
  const int rb = blockIdx.x / S, sl = blockIdx.x % S;
  const int ku0 = sl * KU / S, n = (sl + 1) * KU / S - ku0;
  const int nw = n > half ? (n - half + WPR - 1) / WPR : 0;

  // This slice's out-of-window weights: (row in block | k << 7, FP32 bits), fetched early.
  const int e0 = EPTR[rb * KU + ku0], ne = EPTR[rb * KU + ku0 + n] - e0;
  int2 my_exc = make_int2(0, 0);
  if (tid < ne) my_exc = EXC[e0 + tid];

  const uint8_t* wp = W + ((size_t)(rb * KU + ku0) * 8 + rt) * WBW + lane * 16;
  uint4 buf[PF][NV];
#pragma unroll
  for (int p = 0; p < PF; p++)
    if (p < nw)
#pragma unroll
      for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + (half + p * WPR) * USTRIDE + v * 512);

  for (int c = tid; c < n * MP * 8; c += THREADS) {
    int ul = c / (MP * 8), r = (c >> 3) % MP, ch = c & 7;
    bool valid = r < m;
    int kx = ((ku0 + ul) * KUNIT) % Kx + ch * 8;
    cp_async16(smem + (ul * MP + r) * 128 + ((ch ^ (r & 7)) << 4), X + (size_t)(valid ? r : 0) * Kx + kx,
               valid ? 16 : 0);
  }
  asm volatile("cp.async.commit_group;\ncp.async.wait_group 0;\n" ::);
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

  if (dbg & 1) {
    float t = 0.f;
#pragma unroll
    for (int i = 0; i < NT; i++) t += acc[i][0] + acc[i][1] + acc[i][2] + acc[i][3];
    if (t == 12345.f) WS[0] = t;
    return;
  }

  if (tid < ne) s_exc[tid] = my_exc;
  // This CTA's partial, [MP][ROWS] in FP32, after the x slice.
  // Same offset in every CTA of the cluster (slices differ in length by at most one unit).
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

  // Correct the placeholders: + x * (w - placeholder), exact products summed in FP32.
  // One (exception, token) pair per thread.
  const int nwork = (dbg & 4) ? 0 : ne * m;
  for (int wi = tid; wi < nwork; wi += THREADS) {
    const int t = wi / ne, e = wi - t * ne;  // lanes take different exceptions: no bank pile-up
    const int2 ent = (e < THREADS) ? s_exc[e] : EXC[e0 + e];
    const int r = ent.x & 127, kk = ent.x >> 7;
    const bool hi = kk >= Kx;
    const float wv = __int_as_float(ent.y);
    const float fake = __uint_as_float((hi ? (base1 & 0xFF) : (base0 & 0xFF)) << 23);
    const int kl = kk - ku0 * KUNIT, ul = kl >> 6, kin = kl & 63;
    const __nv_bfloat16 xb = (dbg & 32) ? __float2bfloat16(1.f) : *reinterpret_cast<const __nv_bfloat16*>(
        smem + (ul * MP + t) * 128 + ((((kin >> 3) ^ (t & 7))) << 4) + (kin & 7) * 2);
    const float xv = __bfloat162float(xb);
    if (dbg & 16) part[t * ROWS + r] += xv * wv - xv * fake;
    else atomicAdd(part + t * ROWS + r, xv * wv - xv * fake);
  }
  if (ne) __syncthreads();

  // Cluster reduction: rank r sums rows [r*128/C, (r+1)*128/C) over every CTA in the cluster.
  cg::cluster_group cluster = cg::this_cluster();
  if (C > 1) cluster.sync();
  const int rank = sl % C;
  const int r0 = rank * ROWS / C, nrows = (rank + 1) * ROWS / C - r0;
  const bool single = (S == C);
  const int nitems = nrows * m;
  // Each thread's outputs: issue every load before any store so the round trips overlap.
  constexpr int IT = (ROWS * MP + THREADS - 1) / THREADS;
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
  if (single) {
#pragma unroll
    for (int i = 0; i < IT; i++) {
      const int idx = tid + i * THREADS;
      if (idx < nitems) {
        const int r = r0 + idx % nrows, t = idx / nrows;
        store_out<OUTF32>(OUT, (size_t)t * N + rb * ROWS + r, v[i]);
      }
    }
  } else {
    // Several clusters per row block: the last to arrive at this rank's rows finishes them.
#pragma unroll
    for (int i = 0; i < IT; i++) {
      const int idx = tid + i * THREADS;
      if (idx < nitems) {
        const int r = r0 + idx % nrows, t = idx / nrows;
        atomicAdd(WS + (size_t)(rb * ROWS + r) * WSTRIDE + t, v[i]);
      }
    }
    __syncthreads();
    if (tid == 0) {
      __threadfence();
      s_last = (atomicAdd(CNT + rb * 16 + rank, 1) == S / C - 1);
    }
    __syncthreads();
    if (s_last) {
      __threadfence();
#pragma unroll
      for (int i = 0; i < IT; i++) {
        const int idx = tid + i * THREADS;
        if (idx < nitems) {
          const int r = r0 + idx % nrows, t = idx / nrows;
          v[i] = __ldcg(WS + (size_t)(rb * ROWS + r) * WSTRIDE + t);
        }
      }
#pragma unroll
      for (int i = 0; i < IT; i++) {
        const int idx = tid + i * THREADS;
        if (idx < nitems) {
          const int r = r0 + idx % nrows, t = idx / nrows;
          __stcg(WS + (size_t)(rb * ROWS + r) * WSTRIDE + t, 0.f);
          store_out<OUTF32>(OUT, (size_t)t * N + rb * ROWS + r, v[i]);
        }
      }
      if (tid == 0) CNT[rb * 16 + rank] = 0;
    }
  }
  if (C > 1) cluster.sync();  // peers read this CTA's shared memory until here
}

constexpr int WPR = 2;

template <int NT, bool COMP, bool OUTF32>
int smem_bytes(int KU, int S) {
  int nmax = (KU + S - 1) / S;
  return nmax * NT * 8 * 128 + NT * 8 * ROWS * 4;
}

template <int NT, bool COMP, bool OUTF32>
void launch_t(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor CNT,
              torch::Tensor EPTR, torch::Tensor EXC, int m, int N, int KU, int Kx, int64_t base0, int64_t base1,
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
  at[0].val.clusterDim.x = C;
  at[0].val.clusterDim.y = 1;
  at[0].val.clusterDim.z = 1;
  cfg.attrs = at;
  cfg.numAttrs = 1;
  cudaError_t err = cudaLaunchKernelEx(&cfg, kern, (const uint8_t*)W.data_ptr<uint8_t>(),
      (const __nv_bfloat16*)X.data_ptr(), OUT.data_ptr(), WS.data_ptr<float>(), CNT.data_ptr<int>(),
      (const int*)EPTR.data_ptr<int>(), (const int2*)EXC.data_ptr<int>(), m, N, KU, Kx, S, C,
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

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor CNT,
            torch::Tensor EPTR, torch::Tensor EXC, int64_t m, int64_t N, int64_t KU, int64_t Kx,
            int64_t base0, int64_t base1, bool comp, int64_t S, int64_t C, int64_t dbg) {
  bool f32 = OUT.scalar_type() == torch::kFloat32;
#define CALL(NTV, CP, F) launch_t<NTV, CP, F>(W, X, OUT, WS, CNT, EPTR, EXC, m, N, KU, Kx, base0, base1, S, C, dbg)
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
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor CNT,
            torch::Tensor EPTR, torch::Tensor EXC, int64_t m, int64_t N, int64_t KU, int64_t Kx,
            int64_t base0, int64_t base1, bool comp, int64_t S, int64_t C, int64_t dbg);
int64_t max_clusters(int64_t m, bool comp, bool f32, int64_t KU, int64_t S, int64_t C, int64_t N);
"""

ROWS, KUNIT = 128, 64
MAX_M = 32
WSTRIDE = 32

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="okc_bf16_proj_v3b",
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["launch", "max_clusters"],
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


def _csr(n: int, ku: int, rows, cols, vals, dev):
    """Exceptions grouped by (row block, k unit): entries (row % 128 | k << 7, FP32 bits)."""
    ptr = torch.zeros((n // ROWS) * ku + 1, dtype=torch.int32, device=dev)
    if not rows.numel():
        return ptr, torch.zeros((1, 2), dtype=torch.int32, device=dev)
    key = (rows // ROWS) * ku + cols // KUNIT
    order = torch.argsort(key * (2 * ku * KUNIT) + cols, stable=True)
    rows, cols, vals, key = rows[order], cols[order], vals[order], key[order]
    ptr[1:] = torch.cumsum(torch.bincount(key, minlength=(n // ROWS) * ku), 0)
    ent = torch.stack([((rows % ROWS) | (cols << 7)).int(), vals], dim=1).contiguous()
    return ptr, ent


def _layer_state(w: torch.Tensor, comp: bool):
    n, k = w.shape
    if comp:
        packed, base, rows, cols, vals = _pack_comp(w)
    else:
        packed, base = _pack_raw(w), 1
        rows = cols = vals = torch.zeros(0, dtype=torch.int64, device=w.device)
    ptr, ent = _csr(n, k // KUNIT, rows, cols, vals, w.device)
    return {"w": packed, "base": base * 0x01010101, "base1": base * 0x01010101, "ptr": ptr, "exc": ent,
            "ku": k // KUNIT, "kx": k, "comp": comp, "nexc": int(rows.numel())}


def _router_state(r: torch.Tensor, comp: bool):
    """FP32 [N, K] -> BF16 hi + lo concatenated along K (FP32 accumulation of both)."""
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    n, k = r.shape
    if not comp:
        st = _layer_state(torch.cat([hi, lo], 1), False)
        st["kx"] = k
        return st
    pa, ba, ra, ca, va = _pack_comp(hi)
    pb, bb, rb_, cb, vb = _pack_comp(lo)
    packed = torch.cat([pa.view(n // ROWS, k // KUNIT, -1), pb.view(n // ROWS, k // KUNIT, -1)], 1).flatten()
    rows, cols, vals = torch.cat([ra, rb_]), torch.cat([ca, cb + k]), torch.cat([va, vb])
    ptr, ent = _csr(n, 2 * k // KUNIT, rows, cols, vals, r.device)
    return {"w": packed, "base": ba * 0x01010101, "base1": bb * 0x01010101, "ptr": ptr, "exc": ent,
            "ku": 2 * k // KUNIT, "kx": k, "comp": True, "nexc": int(rows.numel())}


COMPRESS = {"qkv_gate": True, "o": True, "pooler": True, "unpooler": True}
ROUTER_COMPRESS = False


def _plan(state, layer: str, m: int, n: int):
    """(S, C): K-slices per row block and cluster size. Fill the SMs once, keep x's slice in
    shared memory, and keep every cluster co-resident."""
    key = (layer, m, n)
    if key in state["plans"]:
        return state["plans"][key]
    p = state["packs"][layer]
    nrb, ku = n // ROWS, p["ku"]
    mp = 8 if m <= 8 else (16 if m <= 16 else 32)
    s = max(1, min(ku, state["sms"] // nrb))
    while s < ku and -(-ku // s) * mp * 128 + mp * ROWS * 4 > state["smem"]:
        s += 1
    if s > 16:
        s -= s % 16
    ext = _module()
    best = (s, 1)  # clusters of one reduce through global atomics instead
    for c in range(min(s, 16), 1, -1):
        if s % c == 0 and ext.max_clusters(m, p["comp"], layer == "router", ku, s, c, n) * c >= nrb * s:
            best = (s, c)
            break
    state["plans"][key] = best
    return best


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
    state["ws"] = torch.zeros(8192 * WSTRIDE, dtype=torch.float32, device=dev)
    state["cnt"] = torch.zeros(64 * 16, dtype=torch.int32, device=dev)
    props = torch.cuda.get_device_properties(dev)
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
    s, c = plan or _plan(state, layer, m, n)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().launch(p["w"], x.contiguous(), out, state["ws"], state["cnt"], p["ptr"], p["exc"], m, n, p["ku"],
                     p["kx"], p["base"], p["base1"], p["comp"], s, c, dbg)
    return out
