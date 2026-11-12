# okc: problem=muse_bf16_proj
"""wgmma swap-AB GEMM prototype for m = 128/256: compressed W (12-bit) decoded into wgmma A registers,
x via TMA (128B swizzle) as the B operand. v1: one warpgroup per CTA owns 64 rows over all of K."""

from typing import TypedDict

import torch
from torch.utils.cpp_extension import load_inline


class Weights(TypedDict):
    qkv_gate: torch.Tensor
    o: torch.Tensor
    pooler: torch.Tensor
    unpooler: torch.Tensor
    router: torch.Tensor


class Config(TypedDict):
    m: int
    n: int
    k: int
    layer: str


class Inputs(TypedDict):
    x: torch.Tensor
    config: Config


def _wgmma_src(n: int) -> str:
    nreg = n // 2
    dl = ", ".join(f"%{i}" for i in range(nreg))
    a0 = nreg
    cons = ", ".join(f'"+f"(d[{i}])' for i in range(nreg))
    return f'''
__device__ __forceinline__ void wgmma_n{n}(float* d, uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint64_t desc, int scale_d) {{
  asm volatile("{{\\n.reg .pred p;\\nsetp.ne.b32 p, %{a0 + 5}, 0;\\n"
               "wgmma.mma_async.sync.aligned.m64n{n}k16.f32.bf16.bf16 {{{dl}}}, {{%{a0}, %{a0 + 1}, %{a0 + 2}, %{a0 + 3}}}, %{a0 + 4}, p, 1, 1, 0;\\n}}"
               : {cons}
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(desc), "r"(scale_d));
}}
'''


CUDA_SRC = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

#define UNIT_BYTES 1536
__device__ long long g_prof[1024 * 4];
""" + _wgmma_src(128) + _wgmma_src(256) + r"""
__device__ __forceinline__ uint4 ldg_w(const uint8_t* p) {
  uint4 v;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p));
  return v;
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
__device__ __forceinline__ void tma_x(void* dst, const CUtensorMap* map, int k0, int t0, uint64_t* bar) {
  asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
               :: "r"(smem_u32(dst)), "l"(map), "r"(k0), "r"(t0), "r"(smem_u32(bar)) : "memory");
}
__device__ __forceinline__ void decode(uint32_t L0, uint32_t L1, uint32_t nw, uint32_t* a) {
  const uint32_t lowc = 0x07070707u, basec = 0x38383838u, sgn = 0x80808080u;
  const uint32_t h03 = ((nw << 4) & sgn) | (nw & lowc) | basec;
  const uint32_t h47 = (nw & sgn) | ((nw >> 4) & lowc) | basec;
  a[0] = __byte_perm(L0, h03, 0x5140);
  a[1] = __byte_perm(L0, h03, 0x7362);
  a[2] = __byte_perm(L1, h47, 0x5140);
  a[3] = __byte_perm(L1, h47, 0x7362);
}
__device__ __forceinline__ uint64_t make_desc(const void* p) {
  const uint64_t a = smem_u32(p);
  return ((a & 0x3FFFF) >> 4) | ((uint64_t)1 << 16) | ((uint64_t)(1024 >> 4) << 32) | ((uint64_t)1 << 62);
}

struct GP {
  const uint8_t* W;
  void* OUT;
  const int2* exc;      // per 16-row tile, sorted by unit: (unit | step << 8 | reg << 10 | half << 12 | lane << 16, bits)
  const int* eoff;      // [T16 + 1]
  int m, Nout, KU;
  float scale;
};

__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}

// One consumer warpgroup (64 rows = 4 x 16-row tiles, one per warp) x NTOK tokens over all of K, plus a
// producer warp feeding two TMA rings: weights (WS stages of 4 x 1536 B) and x (XS stages of NTOK x 128 B).
template <int NTOK, int WS, int XS>
__global__ void __launch_bounds__(192, 1) wgz(const __grid_constant__ CUtensorMap tmx, const GP p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  constexpr int XB = NTOK * 128;
  constexpr int WB = 4 * UNIT_BYTES;
  uint8_t* xring = smem;
  uint8_t* wring = smem + XS * XB;
  uint64_t* bars = (uint64_t*)(wring + WS * WB);
  uint64_t* wfull = bars;
  uint64_t* wempty = bars + WS;
  uint64_t* xfull = bars + 2 * WS;
  uint64_t* xempty = bars + 2 * WS + XS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int KU = p.KU;
  if (tid == 0) {
    for (int s = 0; s < WS; s++) { mbar_init(wfull + s, 1); mbar_init(wempty + s, 4); }
    for (int s = 0; s < XS; s++) { mbar_init(xfull + s, 1); mbar_init(xempty + s, 4); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 4) {
    // Weight producer.
    if (lane == 0) {
      const uint8_t* wt = p.W + (size_t)(blockIdx.x * 4) * KU * UNIT_BYTES;
      for (int u = 0; u < KU; u++) {
        const int ws = u % WS;
        if (u >= WS) mbar_wait(wempty + ws, ((u / WS) - 1) & 1);
        mbar_expect(wfull + ws, WB);
        for (int t = 0; t < 4; t++)
          bulk_g2s(wring + ws * WB + t * UNIT_BYTES, wt + ((size_t)t * KU + u) * UNIT_BYTES, UNIT_BYTES, wfull + ws);
      }
    }
    return;
  }
  if (warp == 5) {
    // x producer.
    if (lane == 0) {
      for (int xi = 0; xi < KU; xi++) {
        const int xs = xi % XS;
        if (xi >= XS) mbar_wait(xempty + xs, ((xi / XS) - 1) & 1);
        mbar_expect(xfull + xs, XB);
        tma_x(xring + xs * XB, &tmx, xi * 64, 0, xfull + xs);
      }
    }
    return;
  }
  const int t16 = blockIdx.x * 4 + warp;
  const int e0 = p.eoff[t16], e1 = p.eoff[t16 + 1];
  int ebase = e0;
  int2 ent = (e0 + lane < e1) ? p.exc[e0 + lane] : make_int2(0x7fffffff, 0);
  int enext = e0;

  float acc[NTOK / 2];
#pragma unroll
  for (int i = 0; i < NTOK / 2; i++) acc[i] = 0.f;
  uint32_t a[2][4][4];
  long long tw = 0, tx = 0, tg = 0;
  const long long tstart = clock64();
  auto load_a = [&](int u, uint32_t (&aa)[4][4]) {
    const int ws = u % WS;
    long long c0 = clock64();
    mbar_wait(wfull + ws, (u / WS) & 1);
    tw += clock64() - c0;
    const uint4* q = (const uint4*)(wring + ws * WB + warp * UNIT_BYTES) + lane;
    const uint4 l0 = q[0], l1 = q[32], nw4 = q[64];
    __syncwarp();
    if (lane == 0) mbar_arrive(wempty + ws);
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const uint4 lw = (s >> 1) ? l1 : l0;
      const int h = s & 1;
      const uint32_t nw = s == 0 ? nw4.x : (s == 1 ? nw4.y : (s == 2 ? nw4.z : nw4.w));
      decode(h ? lw.z : lw.x, h ? lw.w : lw.y, nw, aa[s]);
    }
    while (enext < e1) {
      if (enext - ebase == 32) {
        ebase += 32;
        ent = (ebase + lane < e1) ? p.exc[ebase + lane] : make_int2(0x7fffffff, 0);
      }
      const int ex = __shfl_sync(0xffffffffu, ent.x, enext - ebase);
      if ((ex & 0xff) != u) break;
      const int ev = __shfl_sync(0xffffffffu, ent.y, enext - ebase);
      if (((ex >> 16) & 31) == lane) {
        const int s_ = (ex >> 8) & 3, rr = (ex >> 10) & 3;
        const uint32_t sh = (ex >> 12) & 1 ? 16 : 0;
        const uint32_t keep = ~(0xFFFFu << sh), put = (uint32_t)ev << sh;
#pragma unroll
        for (int ss = 0; ss < 4; ss++)
#pragma unroll
          for (int r = 0; r < 4; r++)
            if (ss == s_ && r == rr) aa[ss][r] = (aa[ss][r] & keep) | put;
      }
      enext++;
    }
  };
  auto mma_block = [&](int u, uint32_t (&aa)[4][4]) {
    const int xs = u % XS;
    long long c0 = clock64();
    mbar_wait(xfull + xs, (u / XS) & 1);
    tx += clock64() - c0;
    uint8_t* xp = xring + xs * XB;
    asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const uint64_t desc = make_desc(xp + s * 32);
      if (NTOK == 128) wgmma_n128(acc, aa[s][0], aa[s][1], aa[s][2], aa[s][3], desc, 1);
      else wgmma_n256(acc, aa[s][0], aa[s][1], aa[s][2], aa[s][3], desc, 1);
    }
    asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  };
  load_a(0, a[0]);
  for (int u = 0; u < KU; u += 2) {
    mma_block(u, a[0]);
    if (u + 1 < KU) load_a(u + 1, a[1]);
    { long long c0 = clock64();
    asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
    tg += clock64() - c0; }
    if (lane == 0) mbar_arrive(xempty + u % XS);
    if (u + 1 < KU) {
      mma_block(u + 1, a[1]);
      if (u + 2 < KU) load_a(u + 2, a[0]);
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if (lane == 0) mbar_arrive(xempty + (u + 1) % XS);
    }
  }
  if (tid == 0) { g_prof[blockIdx.x * 4] = clock64() - tstart; g_prof[blockIdx.x * 4 + 1] = tw; g_prof[blockIdx.x * 4 + 2] = tx; g_prof[blockIdx.x * 4 + 3] = tg; }
  // Epilogue: acc[4j + q]: row g (+8 for q >= 2), token 8j + 2tq + (q & 1).
  const int g = lane >> 2, tq = lane & 3;
  const int rbase = t16 * 16 + g;
#pragma unroll
  for (int j = 0; j < NTOK / 8; j++) {
#pragma unroll
    for (int q = 0; q < 4; q++) {
      const int tok = 8 * j + 2 * tq + (q & 1);
      const int row = rbase + ((q >> 1) << 3);
      if (tok < p.m) ((__nv_bfloat16*)p.OUT)[(size_t)tok * p.Nout + row] = __float2bfloat16(acc[4 * j + q] * p.scale);
    }
  }
}

typedef CUresult (*EncodeFn)(CUtensorMap*, CUtensorMapDataType, cuuint32_t, void*, const cuuint64_t*, const cuuint64_t*,
                              const cuuint32_t*, const cuuint32_t*, CUtensorMapInterleave, CUtensorMapSwizzle,
                              CUtensorMapL2promotion, CUtensorMapFloatOOBfill);
static EncodeFn get_encode() {
  static EncodeFn fn = nullptr;
  if (!fn) {
    cudaDriverEntryPointQueryResult q;
    cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", (void**)&fn, cudaEnableDefault, &q);
  }
  return fn;
}

template <int NTOK, int WS, int XS>
static void go(torch::Tensor X, const GP& p) {
  CUtensorMap map;
  const int K = X.size(1), m = X.size(0);
  cuuint64_t dims[2] = {(cuuint64_t)K, (cuuint64_t)m};
  cuuint64_t strides[1] = {(cuuint64_t)K * 2};
  cuuint32_t box[2] = {64, NTOK};
  cuuint32_t es[2] = {1, 1};
  CUresult r = get_encode()(&map, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, X.data_ptr(), dims, strides, box, es,
                            CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_128B,
                            CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(r == CUDA_SUCCESS, "tensor map encode failed");
  const int smem = XS * NTOK * 128 + WS * 4 * UNIT_BYTES + 1024 + 16 * (WS + XS) + 64;
  auto kern = wgz<NTOK, WS, XS>;
  static int set = 0;
  if (!set) { cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem); set = 1; }
  kern<<<p.Nout / 64, 192, smem, at::cuda::getCurrentCUDAStream()>>>(map, p);
}

torch::Tensor get_prof() {
  auto t = torch::empty({1024 * 4}, torch::dtype(torch::kInt64));
  cudaMemcpyFromSymbol(t.data_ptr(), g_prof, sizeof(g_prof));
  return t;
}
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EXC, torch::Tensor EOFF, int64_t KU,
            double scale, int64_t ntok, int64_t stages, int64_t D) {
  GP p;
  p.W = W.data_ptr<uint8_t>();
  p.OUT = OUT.data_ptr();
  p.exc = (const int2*)EXC.data_ptr<int>();
  p.eoff = EOFF.data_ptr<int>();
  p.m = X.size(0); p.Nout = OUT.size(1); p.KU = KU; p.scale = (float)scale;
  if (ntok == 128) { if (stages == 4) go<128, 8, 3>(X, p); else go<128, 12, 4>(X, p); }
  else { if (stages == 3) go<256, 8, 2>(X, p); else go<256, 12, 3>(X, p); }
}
"""

CPP_SRC = """
torch::Tensor get_prof();
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EXC, torch::Tensor EOFF, int64_t KU,
            double scale, int64_t ntok, int64_t stages, int64_t D);
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(name="okc_wgz_v2d", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["launch", "get_prof"],
                           extra_cuda_cflags=["-O3", "-gencode=arch=compute_90a,code=sm_90a", "-lineinfo"],
                           extra_ldflags=["-lcuda"], verbose=False)
    return _ext


FIXED_BASE = 56


def _pack_nat(w: torch.Tensor):
    """BF16 [N, K] -> 12-bit packed [T, KU, 1536] in natural K order (wgmma A fragments), plus per-lane patches."""
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
    # k = 64u + 32p + 16h + 8cc2 + 2tq + cc1; row = 16t + 8rh + g
    lv = lo.view(T, 2, 8, KU, 2, 2, 2, 4, 2).permute(0, 3, 4, 2, 7, 5, 6, 1, 8).reshape(T, KU, 1024)
    nv = nib.view(T, 2, 8, KU, 2, 2, 2, 4, 2).permute(0, 3, 2, 7, 4, 5, 6, 1, 8)
    nb = (nv[:, :, :, :, :, :, 0] | (nv[:, :, :, :, :, :, 1] << 4)).reshape(T, KU, 512)
    packed = torch.cat([lv, nb], -1).to(torch.uint8).contiguous()
    idx = exc.nonzero()
    rows, cols = idx[:, 0], idx[:, 1]
    val = (w[rows, cols].double() * 4.0 ** (FIXED_BASE - base)).to(torch.bfloat16)
    vbits = val.view(torch.int16).to(torch.int32) & 0xFFFF
    # per-tile exception lists sorted by unit
    r16 = rows % 16
    rh, g = r16 // 8, r16 % 8
    u = cols // 64
    c = cols % 64
    pp, h, cc2, tq, cc1 = c // 32, (c // 16) % 2, (c // 8) % 2, (c // 2) % 4, c % 2
    lane = g * 4 + tq
    step = pp * 2 + h
    reg = rh + 2 * cc2
    tile = rows // 16
    order = torch.argsort(tile * 4096 + u, stable=True)
    tile, u, step, reg, cc1, lane, vbits = (t[order] for t in (tile, u, step, reg, cc1, lane, vbits))
    counts = torch.bincount(tile, minlength=T)
    eoff = torch.zeros(T + 1, dtype=torch.int32, device=w.device)
    eoff[1:] = torch.cumsum(counts, 0).to(torch.int32)
    ent = torch.stack([(u | (step << 8) | (reg << 10) | (cc1 << 12) | (lane << 16)).to(torch.int32),
                       vbits.to(torch.int32)], 1).contiguous()
    if ent.numel() == 0:
        ent = torch.zeros((1, 2), dtype=torch.int32, device=w.device)
    return {"w": packed.flatten(), "scale": 4.0 ** (base - FIXED_BASE), "ku": KU, "exc": ent, "eoff": eoff}


LAYERS = ("qkv_gate", "o", "pooler", "unpooler")


def prepare(weights: Weights):
    _module()
    state = {"raw_t": {n: weights[n].t().contiguous() for n in LAYERS}}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    state["nat"] = {n: _pack_nat(weights[n]) for n in LAYERS}
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state, cfg=None) -> torch.Tensor:
    x = data["x"]
    c = data["config"]
    m, n, layer = c["m"], c["n"], c["layer"]
    if layer == "router":
        acc = torch.mm(x, state["router_hi_t"], out_dtype=torch.float32)
        acc += torch.mm(x, state["router_lo_t"], out_dtype=torch.float32)
        return acc
    if m < 64 or m > 256 or layer not in ("o", "unpooler", "pooler", "qkv_gate"):
        return torch.mm(x, state["raw_t"][layer][:, :n])
    p = state["nat"][layer]
    ntok = 128 if m <= 128 else 256
    stages = (cfg or {}).get("stages", 4 if ntok == 128 else 3)
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _module().launch(p["w"], x, out, p["exc"], p["eoff"], p["ku"], p["scale"], ntok, stages, 2)
    if (cfg or {}).get("prof"):
        torch.cuda.synchronize()
        pr = _module().get_prof().view(1024, 4)[: n // 64].double() / 1980.0
        print(f"PROF m={m} {layer}: total {pr[:,0].median():.2f}us wfull {pr[:,1].median():.2f} xfull {pr[:,2].median():.2f} wgwait {pr[:,3].median():.2f}", flush=True)
    return out
