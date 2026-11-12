# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open on H200: compressed-weight tensor-core GEMV for m <= 32.

Weights are stored losslessly in 12 bits per BF16 value: the raw low byte (exponent bit 0
and the 7 mantissa bits) plus a nibble holding the sign and a 3-bit code for the upper 7
exponent bits relative to a per-tensor base (an 8-value window, 16 binades). The kernel
decodes with a fixed base (an exact power-of-two scaling, undone at the output) and patches
the rare weights outside the window straight into the mma A registers from per-lane slots.
One CTA owns one 16-row tile over a K slice; its warps split the slice and software-pipeline
their weight loads (the next batch is in flight while the current one is decoded); x comes
through L1. K slices across CTAs reduce in the last CTA to finish (semaphore, one launch).
The router (FP32) runs as its BF16 hi + lo pair, both packed the same way.
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
  const uint4* sp = p.slots + ((((size_t)tile * S + sidx) * NW + warp) * 32 + lane) * 2;
  const uint4 sl0 = sp[0], sl1 = sp[1];
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
  for (int i = 0; i < kuw; i += U) {
    // x for the current batch (L1 hits after the first CTA on the SM touches it).
    uint4 xv[U][NB][2];
#pragma unroll
    for (int v = 0; v < U; v++) {
      const int u = u0 + i + v;
      const int xk = (RT && u >= p.KUh ? u - p.KUh : u) * 64;
#pragma unroll
      for (int nb = 0; nb < NB; nb++) {
        xv[v][nb][0] = __ldg((const uint4*)(xr[nb] + xk));
        xv[v][nb][1] = __ldg((const uint4*)(xr[nb] + xk + 32));
      }
    }
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
      const uint32_t ms[4] = {sl0.x, sl0.z, sl1.x, sl1.z};
      bool hit = false;
#pragma unroll
      for (int q = 0; q < 4; q++) hit |= (ms[q] & 0x23ff & ~0x300) == (0x2000u | (uint32_t)u);
      const bool patch = __any_sync(0xffffffffu, hit);
#pragma unroll
      for (int s = 0; s < 4; s++) {
        const int pp = s >> 1, h = s & 1;
        const uint4 lw = a[v][pp];
        const uint32_t nw = s == 0 ? a[v][2].x : (s == 1 ? a[v][2].y : (s == 2 ? a[v][2].z : a[v][2].w));
        uint32_t f[4];
        decode(h ? lw.z : lw.x, h ? lw.w : lw.y, nw, f);
        if (patch) {
          const uint32_t evs[4] = {sl0.y, sl0.w, sl1.y, sl1.w};
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
          const uint4 x4 = xv[v][nb][pp];
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
  __shared__ float4 red[NW][NB * 32];
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

CPP_SRC = """
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor SLOTS, torch::Tensor WS,
            torch::Tensor SEM, int64_t m, int64_t N, int64_t KU, int64_t KUh, int64_t Kx, double scale0,
            double scale1, int64_t S, int64_t NW, int64_t U, int64_t MB);
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        maj, mnr = torch.cuda.get_device_capability()
        arch = f"{maj}{mnr}" + ("a" if maj == 9 else "")
        _ext = load_inline(name=f"okc_gemvz_v10b_{arch}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                           functions=["launch"],
                           extra_cuda_cflags=["-O3", f"-gencode=arch=compute_{arch},code=sm_{arch}", "-lineinfo"],
                           verbose=False)
    return _ext


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


def _layer(w: torch.Tensor):
    n, k = w.shape
    packed, base, rows, cols, vbits = _pack(w)
    sc = 4.0 ** (base - FIXED_BASE)
    return {"w": packed.flatten(), "scale0": sc, "scale1": sc, "ku": k // 64, "kuh": 0, "kx": k,
            "rows": rows, "kcols": cols, "vbits": vbits, "slots": {}}


def _router(r: torch.Tensor):
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


def _bucket(m: int) -> int:
    return 1 if m <= 1 else (8 if m <= 8 else (16 if m <= 16 else 32))


# (S K-slices, NW warps, U units per batch, min CTAs per SM hint) per (layer, m bucket)
CFG = {
    ("qkv_gate", 1): (1, 8, 2, 0), ("o", 1): (1, 4, 2, 0), ("pooler", 1): (1, 8, 2, 0), ("unpooler", 1): (1, 8, 2, 0),
    ("router", 1): (4, 8, 2, 0),
    ("qkv_gate", 8): (1, 8, 2, 0), ("o", 8): (1, 4, 2, 0), ("pooler", 8): (1, 8, 2, 0), ("unpooler", 8): (1, 8, 2, 0),
    ("router", 8): (4, 8, 2, 0),
    ("qkv_gate", 16): (1, 8, 2, 0), ("o", 16): (1, 4, 2, 0), ("pooler", 16): (1, 8, 2, 0), ("unpooler", 16): (1, 8, 2, 0),
    ("router", 16): (4, 8, 2, 0),
    ("qkv_gate", 32): (1, 8, 2, 0), ("o", 32): (1, 4, 2, 0), ("pooler", 32): (1, 8, 2, 0), ("unpooler", 32): (1, 8, 2, 0),
    ("router", 32): (4, 8, 2, 0),
}
LAYERS = ("qkv_gate", "o", "pooler", "unpooler")


def prepare(weights: Weights):
    _module()
    dev = weights["o"].device
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
    for (layer, mb), (S, NW, U, MB) in CFG.items():
        n = 1024 if layer == "unpooler" and mb <= 8 else weights[layer].shape[0]
        _slot_table(state["packs"][layer], n, S, NW)
    state["ws"] = torch.zeros(8192 * 64 * 4, dtype=torch.float32, device=dev)
    state["sem"] = torch.zeros(8192 // 16, dtype=torch.int32, device=dev)
    torch.cuda.synchronize()
    return state


def _fallback(x, state, layer, n):
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    return torch.mm(x, state["raw_t"][layer][:, :n])


def kernel_fn(data: Inputs, state, cfg=None) -> torch.Tensor:
    x = data["x"]
    c = data["config"]
    m, n, layer = c["m"], c["n"], c["layer"]
    if m > 32:
        return _fallback(x, state, layer, n)
    p = state["packs"][layer]
    S, NW, U, MB = (list(cfg) + [0])[:4] if cfg else CFG[(layer, _bucket(m))]
    tab = _slot_table(p, n, S, NW)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().launch(p["w"], x, out, tab, state["ws"], state["sem"], m, n, p["ku"], p["kuh"], p["kx"], p["scale0"],
                     p["scale1"], S, NW, U, MB)
    return out
