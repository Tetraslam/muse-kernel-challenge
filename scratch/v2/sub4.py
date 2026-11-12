# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open, memory-bound skinny GEMMs on H200.

Weights are stored losslessly in 12 bits per BF16 value: the raw low byte (exponent bit 0
and the 7 mantissa bits) plus a nibble holding the sign and a 3-bit code for the upper 7
exponent bits relative to a per-tensor base (an 8-value window, 16 binades). Weights outside
the window (about 1e-4 of them) decode to a placeholder and get an exact FP32 correction.
The kernel decodes in registers straight into mma.sync A fragments (W rows as M), with x as
the B operand loaded straight from global memory; the weight's K order is permuted so each
lane's B fragment for two k16 steps is a single 16-byte load. Each warp owns one 16-row
tile over a K range; warps of a CTA split K, and the K ranges of a tile group form a
thread-block cluster reducing through distributed shared memory. The router (FP32) runs as
its BF16 hi + lo pair, both packed the same way and concatenated along K, with FP32
accumulation.
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
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define TS(i) if ((p.dbg & 512) && threadIdx.x == 0) g_ts[blockIdx.x * 8 + (i)] = gtime();

struct Params {
  const uint8_t* W;
  const __nv_bfloat16* X;
  void* OUT;
  const int2* exc;   // [grid][cap]: (row in CTA | x col << 16, FP32 delta bits)
  int m, N, T, KU, KUh, Kx;
  uint32_t B0, B1;
  int F, KS, S, cap, dbg;
};

__device__ __forceinline__ uint4 ldg_w(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ uint4 ldg_x(const __nv_bfloat16* p) {
  uint4 v;
  asm volatile("ld.global.nc.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma(float* c, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                    uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <int MT>
struct Stage {
  uint4 w[3];
  uint4 x[MT][2];
};

template <int MT, int NW, int D, int PJ, bool F32OUT>
__global__ void __launch_bounds__(NW * 32, 1) skinny(const Params p) {
  extern __shared__ float red[];  // [F*16][NP]
  constexpr int NP = MT * 8;
  constexpr int NT = NW * 32;
  TS(0)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int rank = blockIdx.x % p.S;
  const int t0 = (blockIdx.x / p.S) * p.F;
  const int nt = min(p.F, p.T - t0);
  const int tl = warp / p.KS, ks = warp - tl * p.KS;
  const bool live = tl < nt;
  const int tile = t0 + min(tl, nt - 1);
  const int ku_per = p.KU / p.S;
  const int per = ku_per / p.KS, rem = ku_per - per * p.KS;
  const int u0 = rank * ku_per + ks * per + min(ks, rem);
  const int nu = live ? per + (ks < rem ? 1 : 0) : 0;
  const uint8_t* wbase = p.W + (size_t)tile * p.KU * UNIT_BYTES + lane * 16;
  const int m = p.m;

  Stage<MT> st[D];
  auto load = [&](Stage<MT>& s, int u) {
    const uint8_t* q = wbase + (size_t)u * UNIT_BYTES;
    s.w[0] = ldg_w(q);
    s.w[1] = ldg_w(q + 512);
    s.w[2] = ldg_w(q + 1024);
    const int xk = (u < p.KUh ? u : u - p.KUh) * 64 + tq * 8;
#pragma unroll
    for (int j = 0; j < MT; j++) {
      const int tok = j * 8 + g;
      if (tok < m) {
        const __nv_bfloat16* px = p.X + (size_t)tok * p.Kx + xk;
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
    if (d < nu) load(st[d], u0 + d);

  // Exceptions: warp w owns entries [w * EW, w * EW + EW) of this CTA's list; lane l < EW loads
  // entry l now, then the warp gathers x for tokens = lanes after the first stage arrives.
  constexpr int EW = 8;
  int2 ent = make_int2(0, 0);
  if (lane < EW && warp * EW + lane < p.cap && !(p.dbg & 1)) ent = p.exc[(size_t)blockIdx.x * p.cap + warp * EW + lane];

  // Shared memory: part[NW][16][NP] per-warp partials, then exc[F*16][NP] (exception sums, and
  // the CTA's reduced partial for the cluster reduction).
  float* part = red;
  float* excs = red + NW * 16 * NP;
  for (int i = threadIdx.x; i < p.F * 16 * NP; i += NT) excs[i] = 0.f;
  TS(1)

  float acc[MT][4];
#pragma unroll
  for (int j = 0; j < MT; j++)
#pragma unroll
    for (int r = 0; r < 4; r++) acc[j][r] = 0.f;

  float exv[EW];
  int exw[EW];
  auto exc_x = [&]() {
#pragma unroll
    for (int e = 0; e < EW; e++) {
      const int ex = __shfl_sync(0xffffffffu, ent.x, e);
      exw[e] = __shfl_sync(0xffffffffu, ent.y, e);
      exv[e] = (exw[e] != 0 && lane < m) ? __bfloat162float(p.X[(size_t)lane * p.Kx + (ex >> 16)]) : 0.f;
      exw[e] = exw[e] != 0 ? (ex & 0xFFFF) : -1;
    }
  };
  if (nu == 0) exc_x();

  for (int i = 0; i < nu; i += D) {
#pragma unroll
    for (int d = 0; d < D; d++) {
      if (i + d < nu) {
        const int u = u0 + i + d;
        const uint32_t B4 = u < p.KUh ? p.B0 : p.B1;
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int pp = s >> 1, h = s & 1;
          const uint4 lw = st[d].w[pp];
          const uint32_t L0 = h ? lw.z : lw.x, L1 = h ? lw.w : lw.y;
          const uint32_t nw = s == 0 ? st[d].w[2].x : (s == 1 ? st[d].w[2].y : (s == 2 ? st[d].w[2].z : st[d].w[2].w));
          const uint32_t h03 = ((nw << 4) & 0x80808080u) | ((nw & 0x07070707u) + B4);
          const uint32_t h47 = (nw & 0x80808080u) | (((nw >> 4) & 0x07070707u) + B4);
          const uint32_t a0 = __byte_perm(L0, h03, 0x5140);
          const uint32_t a1 = __byte_perm(L0, h03, 0x7362);
          const uint32_t a2 = __byte_perm(L1, h47, 0x5140);
          const uint32_t a3 = __byte_perm(L1, h47, 0x7362);
#pragma unroll
          for (int j = 0; j < MT; j++) {
            const uint4 xv = st[d].x[j][pp];
            mma(acc[j], a0, a1, a2, a3, h ? xv.z : xv.x, h ? xv.w : xv.y);
          }
        }
        if (i + d + D < nu) load(st[d], u + D);
        if (i + d == 0) {
          exc_x();
          TS(2)
        }
      }
    }
  }
  TS(3)
  {
    float* w0 = part + (warp * 16 + g) * NP + tq * 2;
#pragma unroll
    for (int j = 0; j < MT; j++) {
      *(float2*)(w0 + j * 8) = make_float2(acc[j][0], acc[j][1]);
      *(float2*)(w0 + 8 * NP + j * 8) = make_float2(acc[j][2], acc[j][3]);
    }
  }
  __syncthreads();  // partials written, excs zeroed
#pragma unroll
  for (int e = 0; e < EW; e++) {
    const float dl = __int_as_float(__shfl_sync(0xffffffffu, ent.y, e));
    if (exw[e] >= 0 && lane < m) atomicAdd(&excs[exw[e] * NP + lane], dl * exv[e]);
  }
  __syncthreads();
  TS(4)

  // Reduce the KS partials of each (tile, row, token): groups of G lanes, each lane summing KS / G.
  const int rows = nt * 16;
  const int rbase = t0 * 16;
  const int G = p.KS < 32 ? p.KS : 32;
  const int gid = threadIdx.x / G, gl = threadIdx.x - gid * G, ngroups = NT / G;
  const int nout = rows * m;
  for (int o = gid; o < nout; o += ngroups) {
    const int tok = o / rows, r = o - tok * rows;
    const int t = r >> 4, rr = r & 15;
    float v = 0.f;
    for (int k = gl; k < p.KS; k += G) v += part[((t * p.KS + k) * 16 + rr) * NP + tok];
    for (int off = G >> 1; off > 0; off >>= 1) v += __shfl_xor_sync(__activemask(), v, off, G);
    if (gl == 0) {
      v += excs[r * NP + tok];
      if (p.S == 1) {
        if (F32OUT) ((float*)p.OUT)[(size_t)tok * p.N + rbase + r] = v;
        else ((__nv_bfloat16*)p.OUT)[(size_t)tok * p.N + rbase + r] = __float2bfloat16(v);
      } else {
        excs[r * NP + tok] = v;
      }
    }
  }
  if (p.S > 1) {
    cg::cluster_group cl = cg::this_cluster();
    cl.sync();
    const int R = rows / p.S;
    const int r0 = rank * R;
    for (int i = threadIdx.x; i < R * m; i += NT) {
      const int tok = i / R, r = r0 + (i - tok * R);
      float v = 0.f;
      for (int s = 0; s < p.S; s++) v += cl.map_shared_rank(excs, s)[r * NP + tok];
      if (F32OUT) ((float*)p.OUT)[(size_t)tok * p.N + rbase + r] = v;
      else ((__nv_bfloat16*)p.OUT)[(size_t)tok * p.N + rbase + r] = __float2bfloat16(v);
    }
    cl.sync();
  }
  TS(5)
}

template <int MT, int NW, int D, int PJ, bool F32>
static void go(const Params& p, cudaStream_t st) {
  auto kern = skinny<MT, NW, D, PJ, F32>;
  int smem = (NW + p.F) * 16 * MT * 8 * 4;
  static int set = 0;
  if (smem > set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    set = smem;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(((p.T + p.F - 1) / p.F) * p.S);
  cfg.blockDim = dim3(NW * 32);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = p.S;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = p.S > 1 ? 1 : 0;
  cudaLaunchKernelEx(&cfg, kern, p);
}

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EXC, int64_t m, int64_t N,
            int64_t KU, int64_t KUh, int64_t Kx, int64_t B0, int64_t B1, int64_t F, int64_t KS, int64_t S,
            int64_t cap, int64_t NW, int64_t D, int64_t dbg) {
  Params p;
  p.W = W.data_ptr<uint8_t>();
  p.X = (const __nv_bfloat16*)X.data_ptr();
  p.OUT = OUT.data_ptr();
  p.exc = (const int2*)EXC.data_ptr<int>();
  p.m = m; p.N = N; p.T = N / 16; p.KU = KU; p.KUh = KUh; p.Kx = Kx;
  p.B0 = (uint32_t)B0; p.B1 = (uint32_t)B1;
  p.F = F; p.KS = KS; p.S = S; p.cap = cap; p.dbg = dbg;
  auto st = at::cuda::getCurrentCUDAStream();
  const int MT = (m + 7) / 8;
  const bool f32 = OUT.scalar_type() == torch::kFloat32;
  // PJ: exception pairs per thread.
  TORCH_CHECK(cap <= NW * 8, "too many exceptions per CTA");
  const int pj = 1;
  TORCH_CHECK(F * KS == NW, "F * KS must equal the warp count");
#define GO(MT_, NW_, D_, F32_) \
  { if (pj <= 1) go<MT_, NW_, D_, 1, F32_>(p, st); else if (pj <= 2) go<MT_, NW_, D_, 2, F32_>(p, st); \
    else go<MT_, NW_, D_, 4, F32_>(p, st); }
#define GOF(MT_, NW_, D_) { if (f32) GO(MT_, NW_, D_, true) else GO(MT_, NW_, D_, false) }
  if (MT <= 1) {
    if (NW == 32) { if (D <= 2) GOF(1, 32, 2) else GOF(1, 32, 3) }
    else { if (D <= 2) GOF(1, 16, 2) else if (D == 3) GOF(1, 16, 3) else GOF(1, 16, 4) }
  } else if (MT <= 2) {
    if (NW == 32) GOF(2, 32, 2) else { if (D <= 2) GOF(2, 16, 2) else GOF(2, 16, 3) }
  } else {
    if (NW == 32) GOF(4, 32, 1) else { if (D <= 2) GOF(4, 16, 2) else GOF(4, 16, 3) }
  }
}
torch::Tensor get_ts() {
  auto t = torch::empty({1024 * 8}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(t.data_ptr(), g_ts, sizeof(g_ts));
  return t;
}
void clear_ts() {
  static unsigned long long z[1024 * 8];
  cudaMemcpyToSymbol(g_ts, z, sizeof(z));
}
"""

CPP_SRC = """
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EXC, int64_t m, int64_t N,
            int64_t KU, int64_t KUh, int64_t Kx, int64_t B0, int64_t B1, int64_t F, int64_t KS, int64_t S,
            int64_t cap, int64_t NW, int64_t D, int64_t dbg);
torch::Tensor get_ts();
void clear_ts();
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        maj, mnr = torch.cuda.get_device_capability()
        arch = f"{maj}{mnr}" + ("a" if maj == 9 else "")
        _ext = load_inline(name=f"okc_skinny_v4b_{arch}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
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


def _layer(w: torch.Tensor):
    n, k = w.shape
    packed, base, rows, cols, delta = _pack(w)
    return {"w": packed.flatten(), "B0": base * 0x01010101, "B1": base * 0x01010101, "ku": k // 64, "kuh": k // 64,
            "kx": k, "rows": rows, "cols": cols, "kcols": cols, "delta": delta, "nexc": int(rows.numel()),
            "exc": {}}


def _router(r: torch.Tensor):
    n, k = r.shape
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    ph, bh, rh, ch, dh = _pack(hi)
    pl, bl, rl, cl, dl = _pack(lo)
    packed = torch.cat([ph, pl], 1).contiguous()
    return {"w": packed.flatten(), "B0": bh * 0x01010101, "B1": bl * 0x01010101, "ku": 2 * k // 64, "kuh": k // 64,
            "kx": k, "rows": torch.cat([rh, rl]), "cols": torch.cat([ch, cl]), "kcols": torch.cat([ch, cl + k]),
            "delta": torch.cat([dh, dl]), "nexc": int(rh.numel() + rl.numel()), "exc": {}}


def _exc_table(p, n: int, F: int, S: int):
    """Per-CTA exception lists for a launch shape: [grid, cap] of (row in CTA | col << 16, FP32 delta bits).
    An exception goes to the CTA owning its row's tile group and its K slice."""
    key = (n, F, S)
    if key in p["exc"]:
        return p["exc"][key]
    rows, cols, kcols, delta = p["rows"], p["cols"], p["kcols"], p["delta"]
    keep = rows < n
    rows, cols, kcols, delta = rows[keep], cols[keep], kcols[keep], delta[keep]
    T = n // 16
    groups = -(-T // F)
    ku_per = p["ku"] // S
    cta = (rows // 16 // F) * S + (kcols // 64) // ku_per
    grid = groups * S
    counts = torch.bincount(cta, minlength=grid)
    cap = max(1, int(counts.max()) if counts.numel() else 1)
    order = torch.argsort(cta, stable=True)
    cta, rows, cols, delta = cta[order], rows[order], cols[order], delta[order]
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(cta.numel(), device=cta.device) - start[cta]
    tab = torch.zeros((grid, cap, 2), dtype=torch.int32, device=rows.device)
    tab[cta, pos, 0] = ((rows - (rows // 16 // F) * F * 16) | (cols << 16)).to(torch.int32)
    tab[cta, pos, 1] = delta.view(torch.int32)
    p["exc"][key] = (tab.contiguous(), cap)
    return p["exc"][key]


def _bucket(m: int) -> int:
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))


# (F tiles per CTA, KS warps along K per tile, S cluster K split, NW warps, D prefetch depth)
CFG = {
    ("qkv_gate", 1): (2, 16, 1, 32, 2), ("qkv_gate", 8): (2, 16, 1, 32, 2),
    ("qkv_gate", 16): (16, 1, 8, 16, 3), ("qkv_gate", 32): (16, 1, 8, 16, 2),
    ("o", 1): (4, 8, 1, 32, 2), ("o", 8): (4, 8, 1, 32, 2), ("o", 16): (4, 4, 1, 16, 3), ("o", 32): (4, 4, 1, 16, 2),
    ("pooler", 1): (2, 16, 1, 32, 2), ("pooler", 8): (2, 16, 1, 32, 2),
    ("pooler", 16): (16, 1, 4, 16, 3), ("pooler", 32): (16, 1, 4, 16, 2),
    ("unpooler", 1): (1, 32, 1, 32, 2), ("unpooler", 8): (1, 32, 1, 32, 2),
    ("unpooler", 16): (16, 1, 2, 16, 3), ("unpooler", 32): (16, 1, 2, 16, 2),
    ("router", 1): (1, 32, 4, 32, 2), ("router", 8): (1, 32, 4, 32, 2),
    ("router", 16): (1, 16, 4, 16, 3), ("router", 32): (1, 16, 4, 16, 2),
}
MAX_M = 32
LAYERS = ("qkv_gate", "o", "pooler", "unpooler")


def prepare(weights: Weights):
    _module()
    state = {"raw_t": {}}
    for name in LAYERS:
        state["raw_t"][name] = weights[name].t().contiguous()
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["packs"] = {name: _layer(weights[name]) for name in LAYERS}
    state["packs"]["router"] = _router(r)
    for (layer, mb), (F, KS, S, NW, D) in CFG.items():
        n = 1024 if layer == "unpooler" and mb <= 8 else weights[layer].shape[0]
        _exc_table(state["packs"][layer], n, F, S)
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
    F, KS, S, NW, D = cfg or CFG[(layer, _bucket(m))]
    tab, cap = _exc_table(p, n, F, S)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().launch(p["w"], x, out, tab, m, n, p["ku"], p["kuh"], p["kx"], p["B0"], p["B1"], F, KS, S, cap, NW,
                     D, dbg)
    return out
