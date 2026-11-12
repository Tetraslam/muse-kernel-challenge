# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open, memory-bound skinny GEMMs on H200.

Weights are stored losslessly in 12 bits per BF16 value: the raw low byte (exponent bit 0
and the 7 mantissa bits) plus a nibble holding the sign and a 3-bit code for the upper 7
exponent bits relative to a per-tensor base (an 8-value window, 16 binades). Weights outside
the window (about 1e-4 of them) decode to a placeholder and get an exact FP32 correction.
The kernel decodes in registers straight into mma.sync A fragments (W rows as M), with x as
the B operand loaded straight from global memory; the weight's K order is permuted so each
lane's B fragment for two k16 steps is a single 16-byte load. CTAs own whole 16-row tiles
over a K range; K ranges of one tile group are a thread-block cluster reducing through
distributed shared memory. The router (FP32) runs as its BF16 hi + lo pair, both packed
the same way and concatenated along K, with FP32 accumulation.
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
    x: torch.Tensor
    config: Config


CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cooperative_groups.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>
namespace cg = cooperative_groups;

#define UNIT_BYTES 1536
__device__ unsigned long long g_ts[1024 * 8];
__device__ __forceinline__ unsigned long long gtime() { unsigned long long t; asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t)); return t; }
#define TS(i) if ((dbg & 512) && threadIdx.x == 0) g_ts[blockIdx.x * 8 + (i)] = gtime();

__device__ __forceinline__ uint4 ldg_w(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldg_x(const __nv_bfloat16* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MT, int TPW>
struct Stage {
  uint4 w[TPW][3];
  uint4 x[MT][2];
};

template <int MT, int TPW, int D, bool F32OUT>
__global__ void __launch_bounds__(512, 1)
skinny(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, void* __restrict__ OUT,
       const int* __restrict__ eoff, const int2* __restrict__ edat, int m, int N, int T, int KU, int KUh, int Kx,
       uint32_t B0, uint32_t B1, int F, int KS, int S, int dbg) {
  extern __shared__ float red[];  // [F*16][NP]
  constexpr int NP = MT * 8;
  TS(0)
  if (dbg & 64) return;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int rank = blockIdx.x % S;
  const int t0 = (blockIdx.x / S) * F;
  const int nt = min(F, T - t0);
  const int tg = warp / KS, ks = warp % KS;
  const int ku_per = KU / S;
  const int per = ku_per / KS, rem = ku_per % KS;
  const int wu0 = rank * ku_per + ks * per + min(ks, rem);
  const int nu = (dbg & 4) ? 0 : per + (ks < rem ? 1 : 0);
  int tile[TPW];
#pragma unroll
  for (int q = 0; q < TPW; q++) tile[q] = min(t0 + tg * TPW + q, T - 1);

  Stage<MT, TPW> st[D];
  auto load = [&](Stage<MT, TPW>& s, int u) {
#pragma unroll
    for (int q = 0; q < TPW; q++) {
      const uint8_t* p = W + ((size_t)tile[q] * KU + u) * UNIT_BYTES + lane * 16;
      s.w[q][0] = ldg_w(p);
      s.w[q][1] = ldg_w(p + 512);
      s.w[q][2] = ldg_w(p + 1024);
    }
    const int xk = (u < KUh ? u : u - KUh) * 64 + tq * 8;
#pragma unroll
    for (int j = 0; j < MT; j++) {
      const int tok = j * 8 + g;
      if (tok < m) {
        const __nv_bfloat16* px = X + (size_t)tok * Kx + xk;
        s.x[j][0] = ldg_x(px);
        s.x[j][1] = ldg_x(px + 32);
      } else {
        s.x[j][0] = make_uint4(0, 0, 0, 0);
        s.x[j][1] = make_uint4(0, 0, 0, 0);
      }
    }
  };

#pragma unroll
  for (int d = 0; d < D; d++)
    if (d < nu) load(st[d], wu0 + d);
  TS(6)

  if (!(dbg & 32)) {
  for (int i = threadIdx.x; i < F * 16 * NP; i += blockDim.x) red[i] = 0.f;
  TS(7)
  __syncthreads();
  }
  if (rank == 0 && !(dbg & 1)) {
    const int e0 = eoff[t0], e1 = eoff[t0 + nt];
    for (int i = e0 * m + threadIdx.x; i < e1 * m; i += blockDim.x) {
      const int e = i / m, tok = i - e * m;
      const int2 ent = edat[e];
      const int row = (ent.x & 0xFFFF) - t0 * 16;  // row within the CTA's tiles
      const int col = ent.x >> 16;
      const float xv = __bfloat162float(X[(size_t)tok * Kx + col]);
      atomicAdd(&red[row * NP + tok], __int_as_float(ent.y) * xv);
    }
  }

  TS(1)
  float acc[TPW][MT][4];
#pragma unroll
  for (int q = 0; q < TPW; q++)
#pragma unroll
    for (int j = 0; j < MT; j++)
#pragma unroll
      for (int r = 0; r < 4; r++) acc[q][j][r] = 0.f;

  for (int i = 0; i < nu; i += D) {
#pragma unroll
    for (int d = 0; d < D; d++) {
      if (i + d < nu) {
        const int u = wu0 + i + d;
        const uint32_t B4 = u < KUh ? B0 : B1;
        if (dbg & 2) {
#pragma unroll
          for (int q = 0; q < TPW; q++) acc[q][0][0] += __int_as_float(st[d].w[q][0].x ^ st[d].w[q][1].y ^ st[d].w[q][2].z);
        } else
#pragma unroll
        for (int p = 0; p < 2; p++) {
#pragma unroll
          for (int h = 0; h < 2; h++) {
            uint32_t xb0[MT], xb1[MT];
#pragma unroll
            for (int j = 0; j < MT; j++) {
              xb0[j] = h ? st[d].x[j][p].z : st[d].x[j][p].x;
              xb1[j] = h ? st[d].x[j][p].w : st[d].x[j][p].y;
            }
#pragma unroll
            for (int q = 0; q < TPW; q++) {
              const uint4 lw = st[d].w[q][p];
              const uint32_t L0 = h ? lw.z : lw.x, L1 = h ? lw.w : lw.y;
              const uint4 nwv = st[d].w[q][2];
              const uint32_t nw = (p * 2 + h) == 0 ? nwv.x : ((p * 2 + h) == 1 ? nwv.y : ((p * 2 + h) == 2 ? nwv.z : nwv.w));
              const uint32_t h03 = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B4);
              const uint32_t h47 = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B4);
              const uint32_t a0 = __byte_perm(L0, h03, 0x5140);
              const uint32_t a1 = __byte_perm(L0, h03, 0x7362);
              const uint32_t a2 = __byte_perm(L1, h47, 0x5140);
              const uint32_t a3 = __byte_perm(L1, h47, 0x7362);
#pragma unroll
              for (int j = 0; j < MT; j++) mma(acc[q][j], a0, a1, a2, a3, xb0[j], xb1[j]);
            }
          }
        }
        if (i + d + D < nu) load(st[d], u + D);
        if (i + d == 0) { TS(2) }
      }
    }
  }

  TS(3)
#pragma unroll
  for (int q = 0; q < TPW; q++) {
    const int tl = tg * TPW + q;
    if (tl < nt) {
#pragma unroll
      for (int j = 0; j < MT; j++) {
        const int tok = j * 8 + tq * 2;
        float* r0 = red + (tl * 16 + g) * NP + tok;
        float* r1 = r0 + 8 * NP;
        if (KS == 1) {
          r0[0] += acc[q][j][0]; r0[1] += acc[q][j][1]; r1[0] += acc[q][j][2]; r1[1] += acc[q][j][3];
        } else {
          atomicAdd(r0, acc[q][j][0]); atomicAdd(r0 + 1, acc[q][j][1]);
          atomicAdd(r1, acc[q][j][2]); atomicAdd(r1 + 1, acc[q][j][3]);
        }
      }
    }
  }
  __syncthreads();

  TS(4)
  const int rows = nt * 16;
  const int rbase = t0 * 16;
  if (dbg & 16) return;
  if (S == 1) {
    for (int i = threadIdx.x; i < rows * m; i += blockDim.x) {
      const int tok = i / rows, r = i - tok * rows;
      const float v = red[r * NP + tok];
      if (F32OUT) ((float*)OUT)[(size_t)tok * N + rbase + r] = v;
      else ((__nv_bfloat16*)OUT)[(size_t)tok * N + rbase + r] = __float2bfloat16(v);
    }
    TS(5)
  } else {
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    const int R = rows / S;
    const int r0 = rank * R;
    for (int i = threadIdx.x; i < R * m; i += blockDim.x) {
      const int tok = i / R, r = r0 + (i - tok * R);
      float v = 0.f;
      for (int s = 0; s < S; s++) v += cl.map_shared_rank(red, s)[r * NP + tok];
      if (F32OUT) ((float*)OUT)[(size_t)tok * N + rbase + r] = v;
      else ((__nv_bfloat16*)OUT)[(size_t)tok * N + rbase + r] = __float2bfloat16(v);
    }
    cl.sync();
    TS(5)
  }
}
torch::Tensor get_ts() {
  auto t = torch::empty({1024 * 8}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(t.data_ptr(), g_ts, sizeof(g_ts));
  return t;
}
void clear_ts() { static unsigned long long z[1024 * 8]; cudaMemcpyToSymbol(g_ts, z, sizeof(z)); }

template <int MT, int TPW, int D, bool F32>
static void go(const uint8_t* W, const __nv_bfloat16* X, void* O, const int* eo, const int2* ed, int m, int N, int T,
               int KU, int KUh, int Kx, uint32_t B0, uint32_t B1, int F, int KS, int S, int dbg, cudaStream_t st) {
  auto kern = skinny<MT, TPW, D, F32>;
  int smem = F * 16 * MT * 8 * 4;
  static int set = 0;
  if (!set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    set = 1;
  }
  int groups = (T + F - 1) / F;
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(groups * S);
  cfg.blockDim = dim3(512);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = S;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = (S > 1 || (dbg & 8)) ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, W, X, (void*)O, eo, ed, m, N, T, KU, KUh, Kx, B0, B1, F, KS, S, dbg);
}

#define DISPATCH_D(MT, TPW, F32)                                                                                   \
  if (D == 2) go<MT, TPW, 2, F32>(Wp, Xp, Op, eo, ed, m, N, T, KU, KUh, Kx, B0, B1, F, KS, S, dbg, st);              \
  else if (D == 3) go<MT, TPW, 3, F32>(Wp, Xp, Op, eo, ed, m, N, T, KU, KUh, Kx, B0, B1, F, KS, S, dbg, st);         \
  else go<MT, TPW, 4, F32>(Wp, Xp, Op, eo, ed, m, N, T, KU, KUh, Kx, B0, B1, F, KS, S, dbg, st);

#define DISPATCH_TPW(MT, F32) \
  if (TPW == 1) { DISPATCH_D(MT, 1, F32) } else { DISPATCH_D(MT, 2, F32) }

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EOFF, torch::Tensor EDAT, int64_t m,
            int64_t N, int64_t KU, int64_t KUh, int64_t Kx, int64_t B0, int64_t B1, int64_t F, int64_t TPW,
            int64_t KS, int64_t S, int64_t D, int64_t dbg) {
  auto st = at::cuda::getCurrentCUDAStream();
  const uint8_t* Wp = W.data_ptr<uint8_t>();
  const __nv_bfloat16* Xp = (const __nv_bfloat16*)X.data_ptr();
  void* Op = OUT.data_ptr();
  const int* eo = EOFF.data_ptr<int>();
  const int2* ed = (const int2*)EDAT.data_ptr<int>();
  int T = N / 16;
  int MT = (m + 7) / 8;
  bool f32 = OUT.scalar_type() == torch::kFloat32;
  if (f32) {
    if (MT <= 1) { DISPATCH_TPW(1, true) } else if (MT <= 2) { DISPATCH_TPW(2, true) } else { DISPATCH_TPW(4, true) }
  } else {
    if (MT <= 1) { DISPATCH_TPW(1, false) } else if (MT <= 2) { DISPATCH_TPW(2, false) } else { DISPATCH_TPW(4, false) }
  }
}
"""

CPP_SRC = """
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EOFF, torch::Tensor EDAT, int64_t m,
            int64_t N, int64_t KU, int64_t KUh, int64_t Kx, int64_t B0, int64_t B1, int64_t F, int64_t TPW,
            int64_t KS, int64_t S, int64_t D, int64_t dbg);
torch::Tensor get_ts();
void clear_ts();
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        maj, mnr = torch.cuda.get_device_capability()
        arch = f"{maj}{mnr}" + ("a" if maj == 9 else "")
        _ext = load_inline(name=f"okc_skinny_v2a_{arch}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                           functions=["launch", "get_ts", "clear_ts"],
                           extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo"],
                           verbose=False)
    return _ext


def _pack(w: torch.Tensor):
    """BF16 [N, K] -> (packed uint8 [T, KU, 1536], base, exception rows, cols, FP32 deltas)."""
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
    ph = torch.tensor([base << 8], dtype=torch.int32).to(torch.int16).view(torch.bfloat16).float().item()
    delta = w[rows, cols].float() - ph
    return packed, base, rows, cols, delta


def _exc_tables(T: int, rows, cols, delta, dev):
    """Per-tile exception lists: offsets [T+1], entries (row in tile | col << 16, FP32 bits)."""
    order = torch.argsort(rows * 65536 + cols)
    rows, cols, delta = rows[order], cols[order], delta[order]
    counts = torch.bincount(rows // 16, minlength=T)
    eoff = torch.zeros(T + 1, dtype=torch.int32, device=dev)
    eoff[1:] = torch.cumsum(counts, 0).to(torch.int32)
    ent = torch.stack([(rows | (cols << 16)).to(torch.int32), delta.view(torch.int32)], 1).contiguous()
    if ent.numel() == 0:
        ent = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    return eoff, ent


def _layer(w: torch.Tensor):
    n, k = w.shape
    packed, base, rows, cols, delta = _pack(w)
    eoff, ent = _exc_tables(n // 16, rows, cols, delta, w.device)
    return {"w": packed.flatten(), "B0": base * 0x01010101, "B1": base * 0x01010101, "ku": k // 64, "kuh": k // 64,
            "kx": k, "eoff": eoff, "edat": ent, "nexc": int(rows.numel())}


def _router(r: torch.Tensor):
    n, k = r.shape
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    ph, bh, rh, ch, dh = _pack(hi)
    pl, bl, rl, cl, dl = _pack(lo)
    packed = torch.cat([ph, pl], 1).contiguous()
    eoff, ent = _exc_tables(n // 16, torch.cat([rh, rl]), torch.cat([ch, cl]), torch.cat([dh, dl]), r.device)
    return {"w": packed.flatten(), "B0": bh * 0x01010101, "B1": bl * 0x01010101, "ku": 2 * k // 64, "kuh": k // 64,
            "kx": k, "eoff": eoff, "edat": ent, "nexc": int(rh.numel() + rl.numel())}


def _bucket(m: int) -> int:
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))


# (F tiles per CTA, TPW tiles per warp, KS warps along K, S cluster K split, D prefetch depth)
CFG = {
    ("qkv_gate", 1): (2, 1, 8, 1, 4), ("qkv_gate", 8): (2, 1, 8, 1, 3),
    ("qkv_gate", 16): (8, 1, 2, 8, 3), ("qkv_gate", 32): (8, 1, 2, 8, 2),
    ("o", 1): (4, 1, 4, 1, 4), ("o", 8): (4, 1, 4, 1, 3), ("o", 16): (4, 1, 4, 1, 3), ("o", 32): (4, 1, 4, 1, 2),
    ("pooler", 1): (2, 1, 8, 1, 4), ("pooler", 8): (2, 1, 8, 1, 3),
    ("pooler", 16): (8, 1, 2, 4, 3), ("pooler", 32): (8, 1, 2, 4, 2),
    ("unpooler", 1): (1, 1, 16, 2, 4), ("unpooler", 8): (1, 1, 16, 2, 3),
    ("unpooler", 16): (4, 1, 4, 1, 3), ("unpooler", 32): (8, 1, 2, 2, 2),
    ("router", 1): (1, 1, 16, 8, 4), ("router", 8): (1, 1, 16, 8, 3),
    ("router", 16): (1, 1, 16, 8, 3), ("router", 32): (1, 1, 16, 8, 2),
}
MAX_M = 32


def prepare(weights: Weights):
    _module()
    state = {"raw_t": {}, "raw": {}}
    for name in ("qkv_gate", "o", "pooler", "unpooler"):
        w = weights[name].contiguous()
        state["raw"][name] = w
        state["raw_t"][name] = w.t().contiguous()
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["packs"] = {name: _layer(weights[name]) for name in ("qkv_gate", "o", "pooler", "unpooler")}
    state["packs"]["router"] = _router(r)
    torch.cuda.synchronize()
    return state


def _fallback(x, state, layer, n):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return torch.mm(x, state["raw_t"][layer][:, :n])


def kernel_fn(data: Inputs, state, cfg=None, dbg=0) -> torch.Tensor:
    x = data["x"]
    c = data["config"]
    m, n, layer = c["m"], c["n"], c["layer"]
    if m > MAX_M:
        return _fallback(x, state, layer, n)
    p = state["packs"][layer]
    F, TPW, KS, S, D = cfg or CFG[(layer, _bucket(m))]
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().launch(p["w"], x, out, p["eoff"], p["edat"], m, n, p["ku"], p["kuh"], p["kx"], p["B0"], p["B1"],
                     F, TPW, KS, S, D, dbg)
    return out
