# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open, memory-bound skinny GEMMs on H200 (v7, m <= 8 path).

Weights are stored losslessly in 12 bits per BF16 value: the raw low byte (exponent bit 0
and the 7 mantissa bits) plus a nibble holding the sign and a 3-bit code for the upper 7
exponent bits relative to a per-tensor base (an 8-value window, 16 binades). The kernel
decodes with a fixed base instead (an exact power-of-two scaling, undone at the output) and
patches the rare weights outside the window straight into the mma A registers from per-lane
slots. Each warp streams its weight units through a register ring; x arrives in a small
shared-memory ring by TMA bulk copies.
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
#define FIXED_BASE 56u
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
  const uint4* slots;  // [grid][NW][32] per-lane exception patches (2 per lane)
  int m, N, T, KU, KUh, Kx;
  float scale0, scale1;  // output scale of the K halves (4^(base-56))
  int F, KS, WU, R, dbg;
};

__device__ __forceinline__ uint4 ldg_w(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
               "{%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void mbar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"(smem_u32(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(smem_u32(b)), "r"(bytes) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, uint32_t phase) {
  asm volatile("{\n.reg .pred p;\nWAIT_%=: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n@!p bra WAIT_%=;\n}"
               :: "r"(smem_u32(b)), "r"(phase) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}

// Decode one k16 step (8 weights per lane) into the A fragment, fixed base.
__device__ __forceinline__ void decode(uint32_t L0, uint32_t L1, uint32_t nw, uint32_t* a) {
  const uint32_t lowc = 0x07070707u, basec = 0x38383838u, sgn = 0x80808080u;
  const uint32_t h03 = ((nw << 4) & sgn) | (nw & lowc) | basec;
  const uint32_t h47 = (nw & sgn) | ((nw >> 4) & lowc) | basec;
  a[0] = __byte_perm(L0, h03, 0x5140);
  a[1] = __byte_perm(L0, h03, 0x7362);
  a[2] = __byte_perm(L1, h47, 0x5140);
  a[3] = __byte_perm(L1, h47, 0x7362);
}

// m <= 8. One CTA: F tiles of 16 rows over all of K. Warp (tl, ks) takes units ks, ks + KS, ...
template <int NW, int D, bool F32OUT>
__global__ void __launch_bounds__(NW * 32, 1) skinny8(const Params p) {
  extern __shared__ __align__(128) uint8_t smem[];
  constexpr int NT = NW * 32;
  TS(0)
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = lane >> 2, tq = lane & 3;
  const int m = p.m;
  const int t0 = blockIdx.x * p.F;
  const int nt = min(p.F, p.T - t0);
  const int tl = warp / p.KS, ks = warp - tl * p.KS;
  const bool live = tl < nt;
  const int tile = t0 + min(tl, nt - 1);
  const int nu = live ? p.KU / p.KS : 0;
  const uint8_t* wbase = p.W + ((size_t)tile * p.KU + ks) * UNIT_BYTES + lane * 16;
  const size_t wstep = (size_t)p.KS * UNIT_BYTES;

  // Exception slots (4 per lane) first: tiny, so they land before the first weight unit.
  const uint4* sp = p.slots + (((size_t)blockIdx.x * NW + warp) * 32 + lane) * 2;
  const uint4 sl0 = sp[0], sl1 = sp[1];
  // Weight ring next.
  uint4 st[D][3];
#pragma unroll
  for (int d = 0; d < D; d++)
    if (d < nu) {
      const uint8_t* q = wbase + d * wstep;
      st[d][0] = ldg_w(q);
      st[d][1] = ldg_w(q + 512);
      st[d][2] = ldg_w(q + 1024);
    }

  // Shared memory: mbarriers [R], counters [R], x ring [R][m + 1 rows][pitch].
  const int R = p.R, WU = p.WU;
  const int pitch = WU * 128 + 64;
  const int rows = m + 1;
  uint64_t* full = (uint64_t*)smem;
  int* cnt = (int*)(smem + 8 * R);
  uint8_t* xr = smem + 128;
  const int nwin = p.KU / WU;
  if (threadIdx.x == 0) {
    for (int r = 0; r < R; r++) { mbar_init(full + r, 1); cnt[r] = 0; }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  // Zero rows: row m of every slot.
  for (int i = threadIdx.x; i < R * (pitch / 16); i += NT) {
    const int r = i / (pitch / 16), c = i - r * (pitch / 16);
    *(uint4*)(xr + ((size_t)r * rows + m) * pitch + c * 16) = make_uint4(0, 0, 0, 0);
  }
  __syncthreads();
  auto fill = [&](int w) {
    const int slotw = w % R;
    const int u0 = w * WU;
    const int xc = (u0 < p.KUh ? u0 : u0 - p.KUh) * 64;
    mbar_expect(full + slotw, (uint32_t)(m * WU * 128));
    for (int t = 0; t < m; t++)
      bulk_g2s(xr + ((size_t)slotw * rows + t) * pitch, p.X + (size_t)t * p.Kx + xc, WU * 128, full + slotw);
  };
  if (threadIdx.x == 0)
    for (int w = 0; w < min(R, nwin); w++) fill(w);
  TS(1)

  float acc0[4] = {0.f, 0.f, 0.f, 0.f}, acc1[4] = {0.f, 0.f, 0.f, 0.f};
  const int xrow = g < m ? g : m;
  // Iterations (unit index i) of this warp holding any exception, as one warp-uniform mask.
  uint32_t emask;
  {
    uint32_t b = 0;
    const uint32_t ms[4] = {sl0.x, sl0.z, sl1.x, sl1.z};
#pragma unroll
    for (int q = 0; q < 4; q++)
      if (ms[q] & 0x2000) b |= 1u << (((ms[q] & 0xff) - ks) / p.KS);
    emask = __reduce_or_sync(0xffffffffu, b);
  }
  for (int i = 0; i < nu; i += D) {
#pragma unroll
    for (int d = 0; d < D; d++) {
      if (i + d < nu) {
        const int u = ks + (i + d) * p.KS;
        const int w = u / WU;
        const int slotw = w % R;
        mbar_wait(full + slotw, (uint32_t)((w / R) & 1));
        const uint8_t* xp = xr + ((size_t)slotw * rows + xrow) * pitch + (u - w * WU) * 128 + tq * 16;
        const uint4 x0 = *(const uint4*)xp;
        const uint4 x1 = *(const uint4*)(xp + 64);
        const bool hi = p.KUh && u >= p.KUh;
        const bool patch = (emask >> (i + d)) & 1;
#pragma unroll
        for (int s = 0; s < 4; s++) {
          const int pp = s >> 1, h = s & 1;
          const uint4 lw = st[d][pp];
          const uint32_t nw = s == 0 ? st[d][2].x : (s == 1 ? st[d][2].y : (s == 2 ? st[d][2].z : st[d][2].w));
          uint32_t a[4];
          decode(h ? lw.z : lw.x, h ? lw.w : lw.y, nw, a);
          if (patch) {
            const uint32_t exs[4] = {sl0.x, sl0.z, sl1.x, sl1.z}, evs[4] = {sl0.y, sl0.w, sl1.y, sl1.w};
#pragma unroll
            for (int q = 0; q < 4; q++) {
              const uint32_t ex = exs[q], ev = evs[q];
              if ((ex & 0x23ff) == (0x2000u | ((uint32_t)s << 8) | (uint32_t)u)) {
                const int r = (ex >> 10) & 3;
                const uint32_t sh = (ex >> 12) & 1 ? 16 : 0;
#pragma unroll
                for (int rr = 0; rr < 4; rr++)
                  if (rr == r) a[rr] = (a[rr] & ~(0xFFFFu << sh)) | (ev << sh);
              }
            }
          }
          const uint4 xv = pp ? x1 : x0;
          mma((hi || (!p.KUh && h)) ? acc1 : acc0, a, h ? xv.z : xv.x, h ? xv.w : xv.y);
        }
        // Done with this unit's x: the last consumer of a window refills it with window w + R.
        __syncwarp();
        if (lane == 0) {
          const int done = atomicAdd(cnt + slotw, 1) + 1;
          if (done == WU * nt) {
            cnt[slotw] = 0;
            if (w + R < nwin) {
              asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
              fill(w + R);
            }
          }
        }
        if (i + d + D < nu) {
          const uint8_t* q = wbase + (size_t)(i + d + D) * wstep;
          st[d][0] = ldg_w(q);
          st[d][1] = ldg_w(q + 512);
          st[d][2] = ldg_w(q + 1024);
        }
      }
    }
  }
  TS(3)
  // Output scale per K half; partials as part[warp][tok 8][16 rows] FP32.
  float* part = (float*)(xr + (size_t)R * rows * pitch);
  {
    const float s0 = p.scale0, s1 = p.KUh ? p.scale1 : p.scale0;
    float* w0 = part + (warp * 8 + tq * 2) * 16 + g;
    w0[0] = acc0[0] * s0 + acc1[0] * s1;
    w0[16] = acc0[1] * s0 + acc1[1] * s1;
    w0[8] = acc0[2] * s0 + acc1[2] * s1;
    w0[16 + 8] = acc0[3] * s0 + acc1[3] * s1;
  }
  __syncthreads();
  TS(4)
  // One thread per output (rows fastest): sum the KS partials of its tile.
  const int orows = nt * 16;
  for (int o = threadIdx.x; o < orows * m; o += NT) {
    const int tok = o / orows, r = o - tok * orows;
    const int t = r >> 4, rr = r & 15;
    const float* src = part + ((t * p.KS) * 8 + tok) * 16 + rr;
    float v = 0.f;
    for (int k = 0; k < p.KS; k++) v += src[k * 128];
    if (F32OUT) ((float*)p.OUT)[(size_t)tok * p.N + t0 * 16 + r] = v;
    else ((__nv_bfloat16*)p.OUT)[(size_t)tok * p.N + t0 * 16 + r] = __float2bfloat16(v);
  }
  TS(5)
}

template <int NW, int D, bool F32>
static void go(const Params& p, cudaStream_t st) {
  auto kern = skinny8<NW, D, F32>;
  const int pitch = p.WU * 128 + 64;
  int smem = 128 + p.R * (p.m + 1) * pitch + NW * 8 * 16 * 4;
  static int set = 0;
  if (smem > set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    set = smem;
  }
  kern<<<(p.T + p.F - 1) / p.F, NW * 32, smem, st>>>(p);
}

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor SLOTS, int64_t m, int64_t N,
            int64_t KU, int64_t KUh, int64_t Kx, double scale0, double scale1, int64_t F, int64_t KS, int64_t WU,
            int64_t R, int64_t NW, int64_t D, int64_t dbg) {
  Params p;
  p.W = W.data_ptr<uint8_t>();
  p.X = (const __nv_bfloat16*)X.data_ptr();
  p.OUT = OUT.data_ptr();
  p.slots = (const uint4*)SLOTS.data_ptr<int>();
  p.m = m; p.N = N; p.T = N / 16; p.KU = KU; p.KUh = KUh; p.Kx = Kx;
  p.scale0 = (float)scale0; p.scale1 = (float)scale1;
  p.F = F; p.KS = KS; p.WU = WU; p.R = R; p.dbg = dbg;
  TORCH_CHECK(F * KS == NW && KU % KS == 0 && KU % WU == 0 && m <= 8, "bad config");
  auto st = at::cuda::getCurrentCUDAStream();
  const bool f32 = OUT.scalar_type() == torch::kFloat32;
  if (NW == 32) { if (f32) go<32, 2, true>(p, st); else go<32, 2, false>(p, st); }
  else { if (f32) go<16, 3, true>(p, st); else go<16, 3, false>(p, st); }
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
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor SLOTS, int64_t m, int64_t N,
            int64_t KU, int64_t KUh, int64_t Kx, double scale0, double scale1, int64_t F, int64_t KS, int64_t WU,
            int64_t R, int64_t NW, int64_t D, int64_t dbg);
torch::Tensor get_ts();
void clear_ts();
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        maj, mnr = torch.cuda.get_device_capability()
        arch = f"{maj}{mnr}" + ("a" if maj == 9 else "")
        _ext = load_inline(name=f"okc_skinny_v7b_{arch}", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC,
                           functions=["launch", "get_ts", "clear_ts"],
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
    # True value in the decode domain: scaled by 4^(56 - base), exact (power of two).
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


def _slot_table(p, n: int, F: int, KS: int, NW: int):
    """Per-lane exception patches for a launch shape: int32 [grid, NW, 32, 8]:
    (unit | step << 8 | reg << 10 | half << 12 | valid << 13, scaled BF16 bits), four per lane."""
    key = (n, F, KS, NW)
    if key in p["slots"]:
        return p["slots"][key]
    rows, kc, vb = p["rows"], p["kcols"], p["vbits"]
    keep = rows < n
    rows, kc, vb = rows[keep], kc[keep], vb[keep]
    T = n // 16
    grid = -(-T // F)
    tile = rows // 16
    r16 = rows % 16
    rh, g = r16 // 8, r16 % 8
    u = kc // 64
    c = kc % 64
    pp, tq, h, cc2, cc1 = c // 32, (c // 8) % 4, (c // 4) % 2, (c // 2) % 2, c % 2
    lane = g * 4 + tq
    step = pp * 2 + h
    reg = rh + 2 * cc2
    cta = tile // F
    warp = (tile - cta * F) * KS + u % KS
    lin = (cta * NW + warp) * 32 + lane
    order = torch.argsort(lin, stable=True)
    lin, u, step, reg, cc1, vb = lin[order], u[order], step[order], reg[order], cc1[order], vb[order]
    counts = torch.bincount(lin, minlength=grid * NW * 32)
    assert int(counts.max()) <= 4 if counts.numel() else True, "more than 4 exceptions in one lane"
    start = torch.cumsum(counts, 0) - counts
    pos = torch.arange(lin.numel(), device=lin.device) - start[lin]
    tab = torch.zeros((grid * NW * 32, 8), dtype=torch.int32, device=rows.device)
    meta = (u | (step << 8) | (reg << 10) | (cc1 << 12) | (1 << 13)).to(torch.int32)
    tab[lin, pos * 2] = meta
    tab[lin, pos * 2 + 1] = vb.to(torch.int32)
    p["slots"][key] = tab.contiguous()
    return tab


# m <= 8 configs: (F tiles per CTA, KS warps along K, WU x-window units, R ring windows, NW warps)
CFG = {
    "qkv_gate": (2, 16, 8, 4, 32), "o": (4, 8, 8, 2, 32), "pooler": (2, 16, 8, 4, 32),
    "unpooler": (1, 32, 8, 4, 32), "router": (1, 32, 8, 4, 32),
}
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
    for layer, (F, KS, WU, R, NW) in CFG.items():
        n = weights[layer].shape[0]
        _slot_table(state["packs"][layer], n, F, KS, NW)
        if layer == "unpooler":
            _slot_table(state["packs"][layer], 1024, F, KS, NW)
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
    if m > 8:
        return _fallback(x, state, layer, n)
    p = state["packs"][layer]
    F, KS, WU, R, NW = cfg or CFG[layer]
    R = min(R, p["ku"] // WU)
    tab = _slot_table(p, n, F, KS, NW)
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    _module().launch(p["w"], x, out, tab, m, n, p["ku"], p["kuh"], p["kx"], p["scale0"], p["scale1"], F, KS, WU, R,
                     NW, 2 if NW == 32 else 3, dbg)
    return out
