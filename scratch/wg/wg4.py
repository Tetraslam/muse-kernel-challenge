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
#include <algorithm>

#define UNIT_BYTES 1536
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
  int m, Nout, KU, S;
  float scale;
};

__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"(smem_u32(b)) : "memory");
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
               :: "r"(smem_u32(dst)), "l"(src), "r"(bytes), "r"(smem_u32(b)) : "memory");
}

__device__ __forceinline__ uint32_t cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(r));
  return r;
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;" ::: "memory");
}
__device__ __forceinline__ float ld_dsmem_f32(const float* local, uint32_t rank) {
  uint32_t a = smem_u32(local), r;
  float v;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(r) : "r"(a), "r"(rank));
  asm volatile("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(r) : "memory");
  return v;
}

// CTA: 128 rows (two consumer warpgroups on different 64-row halves, sharing x stages) x NTOK tokens
// over a K slice; S slices form a cluster and reduce through distributed shared memory.
template <int NTOK, int WS, int XS>
__global__ void __launch_bounds__(320, 1) wgz(const __grid_constant__ CUtensorMap tmx, const GP p) {
  extern __shared__ __align__(1024) uint8_t smem_raw[];
  uint8_t* smem = (uint8_t*)(((uintptr_t)smem_raw + 1023) & ~(uintptr_t)1023);
  constexpr int XB = NTOK * 128;
  constexpr int WB = 8 * UNIT_BYTES;
  uint8_t* xring = smem;
  uint8_t* wring = smem + XS * XB;
  uint64_t* bars = (uint64_t*)(wring + WS * WB);
  uint64_t* wfull = bars;
  uint64_t* wempty = bars + WS;
  uint64_t* xfull = bars + 2 * WS;
  uint64_t* xempty = bars + 2 * WS + XS;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int KU = p.KU, S = p.S;
  const int rank = S > 1 ? (int)cluster_rank() : 0;
  const int r128 = blockIdx.x / S;
  const int kb0 = rank * KU / S, kb1 = (rank + 1) * KU / S, nkb = kb1 - kb0;
  if (tid == 0) {
    for (int s = 0; s < WS; s++) { mbar_init(wfull + s, 1); mbar_init(wempty + s, 8); }
    for (int s = 0; s < XS; s++) { mbar_init(xfull + s, 1); mbar_init(xempty + s, 8); }
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
  }
  __syncthreads();
  if (warp == 8) {
    if (lane == 0) {
      const uint8_t* wt = p.W + (size_t)(r128 * 8) * KU * UNIT_BYTES;
      for (int i = 0; i < nkb; i++) {
        const int u = kb0 + i, ws = i % WS;
        if (i >= WS) mbar_wait(wempty + ws, ((i / WS) - 1) & 1);
        mbar_expect(wfull + ws, WB);
        for (int t = 0; t < 8; t++)
          bulk_g2s(wring + ws * WB + t * UNIT_BYTES, wt + ((size_t)t * KU + u) * UNIT_BYTES, UNIT_BYTES, wfull + ws);
      }
    }
  } else if (warp == 9) {
    if (lane == 0) {
      for (int i = 0; i < nkb; i++) {
        const int xs = i % XS;
        if (i >= XS) mbar_wait(xempty + xs, ((i / XS) - 1) & 1);
        mbar_expect(xfull + xs, XB);
        tma_x(xring + xs * XB, &tmx, (kb0 + i) * 64, 0, xfull + xs);
      }
    }
  }
  float acc[NTOK / 2];
#pragma unroll
  for (int i = 0; i < NTOK / 2; i++) acc[i] = 0.f;
  const int t8 = warp & 7;                // tile within the CTA (consumer warps 0..7)
  const int t16 = r128 * 8 + t8;
  if (warp < 8) {
    const int e1 = p.eoff[t16 + 1];
    int enext = p.eoff[t16];
    int ebase = enext;
    int2 ent = (ebase + lane < e1) ? p.exc[ebase + lane] : make_int2(0x7fffffff, 0);
    uint32_t a[2][4][4];
    auto load_a = [&](int i, uint32_t (&aa)[4][4]) {
      const int u = kb0 + i, ws = i % WS;
      mbar_wait(wfull + ws, (i / WS) & 1);
      const uint4* q = (const uint4*)(wring + ws * WB + t8 * UNIT_BYTES) + lane;
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
        const int eu = ex & 0xff;
        if (eu > u) break;
        if (eu == u) {
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
        }
        enext++;
      }
    };
    auto mma_block = [&](int i, uint32_t (&aa)[4][4]) {
      const int xs = i % XS;
      mbar_wait(xfull + xs, (i / XS) & 1);
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
    if (nkb > 0) load_a(0, a[0]);
    for (int i = 0; i < nkb; i += 2) {
      mma_block(i, a[0]);
      if (i + 1 < nkb) load_a(i + 1, a[1]);
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if (lane == 0) mbar_arrive(xempty + i % XS);
      if (i + 1 >= nkb) break;
      mma_block(i + 1, a[1]);
      if (i + 2 < nkb) load_a(i + 2, a[0]);
      asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
      if (lane == 0) mbar_arrive(xempty + (i + 1) % XS);
    }
  }
  const int g = lane >> 2, tq = lane & 3;
  const int rbase = t16 * 16 + g;
  if (S == 1) {
    if (warp < 8) {
#pragma unroll
      for (int j = 0; j < NTOK / 8; j++)
#pragma unroll
        for (int q = 0; q < 4; q++) {
          const int tok = 8 * j + 2 * tq + (q & 1);
          if (tok < p.m)
            ((__nv_bfloat16*)p.OUT)[(size_t)tok * p.Nout + rbase + ((q >> 1) << 3)] = __float2bfloat16(acc[4 * j + q] * p.scale);
        }
    }
    return;
  }
  // Cluster reduction: partials [256 consumer threads][NTOK / 2] in each CTA's (now idle) rings.
  __syncthreads();
  float* part = (float*)smem;
  if (warp < 8) {
#pragma unroll
    for (int i = 0; i < NTOK / 2; i++) part[i * 256 + tid] = acc[i];
  }
  cluster_sync();
  if (warp < 8) {
    // Rank r finalizes j blocks [r * J / S, (r + 1) * J / S).
    constexpr int J = NTOK / 8;
    const int j0 = rank * J / S, j1 = (rank + 1) * J / S;
    for (int j = j0; j < j1; j++) {
#pragma unroll
      for (int q = 0; q < 4; q++) {
        float v = 0.f;
        for (int s = 0; s < S; s++) v += ld_dsmem_f32(part + (4 * j + q) * 256 + tid, s);
        const int tok = 8 * j + 2 * tq + (q & 1);
        if (tok < p.m)
          ((__nv_bfloat16*)p.OUT)[(size_t)tok * p.Nout + rbase + ((q >> 1) << 3)] = __float2bfloat16(v * p.scale);
      }
    }
  }
  cluster_sync();
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
  const int S = p.S;
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
  const int ring = XS * NTOK * 128 + WS * 8 * UNIT_BYTES;
  const int smem = std::max(ring, 256 * NTOK * 2) + 1024 + 16 * (WS + XS) + 64;
  auto kern = wgz<NTOK, WS, XS>;
  static int set = 0;
  if (!set) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    cudaFuncSetAttribute(kern, cudaFuncAttributeNonPortableClusterSizeAllowed, 1);
    set = 1;
  }
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3((p.Nout / 128) * S);
  cfg.blockDim = dim3(320);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = at::cuda::getCurrentCUDAStream();
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = S;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = S > 1 ? 1 : 0;
  cudaError_t e = cudaLaunchKernelEx(&cfg, kern, map, p);
  TORCH_CHECK(e == cudaSuccess, "launch failed: ", cudaGetErrorString(e));
}

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EXC, torch::Tensor EOFF, int64_t KU,
            double scale, int64_t ntok, int64_t stages, int64_t S) {
  GP p;
  p.W = W.data_ptr<uint8_t>();
  p.OUT = OUT.data_ptr();
  p.exc = (const int2*)EXC.data_ptr<int>();
  p.eoff = EOFF.data_ptr<int>();
  p.m = X.size(0); p.Nout = OUT.size(1); p.KU = KU; p.S = S; p.scale = (float)scale;
  if (ntok == 128) { if (stages == 4) go<128, 8, 4>(X, p); else if (stages == 5) go<128, 6, 5>(X, p); else go<128, 8, 6>(X, p); }
  else { if (stages == 3) go<256, 6, 3>(X, p); else go<256, 4, 4>(X, p); }
}
"""

CPP_SRC = """
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor EXC, torch::Tensor EOFF, int64_t KU,
            double scale, int64_t ntok, int64_t stages, int64_t S);
"""

_ext = None


def _module():
    global _ext
    if _ext is None:
        _ext = load_inline(name="okc_wgz_v4a", cpp_sources=CPP_SRC, cuda_sources=CUDA_SRC, functions=["launch"],
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
    S = (cfg or {}).get("S", {"o": 2, "unpooler": 2, "pooler": 4, "qkv_gate": 7}[layer])
    out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    _module().launch(p["w"], x, out, p["exc"], p["eoff"], p["ku"], p["scale"], ntok, stages, S)
    return out
