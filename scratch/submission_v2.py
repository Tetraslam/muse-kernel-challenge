# okc: problem=muse_bf16_proj
"""Dense projections for Muse Spark Open.

Skinny shapes (m <= 32) are memory-bound, so the BF16 weights are stored losslessly in
12 bits: the low byte of each value keeps sign + 7 mantissa bits, and a 4-bit code holds
the exponent relative to a per-tensor base (a 16-binade window). The ~1e-4 of weights
outside the window are encoded as a fixed placeholder and corrected exactly in FP32 when
each output tile is finalized. The kernel decodes in registers straight into mma.sync
A fragments, splits the work stream-K over all SMs, and reduces partial tiles through a
self-cleaning FP32 workspace. The router (FP32) runs as a BF16 hi + lo pair concatenated
along K, with FP32 accumulation. Larger m goes to cuBLAS.
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
#include <cuda_bf16.h>
#include <stdint.h>

constexpr int ROWS = 128;      // output rows per work unit (8 warps x 16)
constexpr int KUNIT = 64;      // k per work unit
constexpr int THREADS = 256;
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
  uint32_t s = (uint32_t)__cvta_generic_to_shared(smem);
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
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


__device__ __forceinline__ uint4 ldg_stream(const uint8_t* p) {
  uint4 r;
  asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p));
  return r;
}

// One CTA = one 128-row block x one K-slice. Each warp owns 16 rows and streams its
// weight fragments from HBM straight into registers; x's slice sits in shared memory.
template <int NT, bool COMP, bool OUTF32, int PF>
__global__ void __launch_bounds__(THREADS, 1)
skinny_kernel(const uint8_t* __restrict__ W, const __nv_bfloat16* __restrict__ X, void* __restrict__ OUT,
              float* __restrict__ WS, int* __restrict__ CNT, const int* __restrict__ EXC_PTR,
              const int2* __restrict__ EXC, int m, int N, int KU, int Kx, int S, uint32_t base0,
              uint32_t base1, int dbg) {
  constexpr int MP = NT * 8;
  constexpr int WBW = COMP ? 1536 : 2048;
  constexpr int NV = COMP ? 3 : 4;
  constexpr size_t USTRIDE = 8 * WBW;
  extern __shared__ __align__(128) uint8_t smem[];
  __shared__ int s_last;

  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t4 = lane & 3;
  const int rb = blockIdx.x / S, sl = blockIdx.x % S;
  const int ku0 = sl * KU / S, n = (sl + 1) * KU / S - ku0;

  const uint8_t* wp = W + ((size_t)(rb * KU + ku0) * 8 + warp) * WBW + lane * 16;
  uint4 buf[PF][NV];
#pragma unroll
  for (int p = 0; p < PF; p++)
    if (p < n)
#pragma unroll
      for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + p * USTRIDE + v * 512);

  for (int c = tid; c < n * MP * 8; c += THREADS) {
    int ul = c / (MP * 8), r = (c >> 3) % MP, ch = c & 7;
    bool valid = r < m;
    int kx = ((ku0 + ul) * KUNIT) % Kx + ch * 8;
    cp_async16(smem + (ul * MP + r) * 128 + ((ch ^ (r & 7)) << 4), X + (size_t)(valid ? r : 0) * Kx + kx,
               valid ? 16 : 0);
  }
  cp_async_commit();
  cp_async_wait<0>();
  __syncthreads();

  float acc[NT][4];
#pragma unroll
  for (int i = 0; i < NT; i++)
#pragma unroll
    for (int j = 0; j < 4; j++) acc[i][j] = 0.f;

  for (int i = 0; i < n; i += PF) {
#pragma unroll
    for (int p = 0; p < PF; p++) {
      const int u = i + p;
      if (u < n && (dbg & 2)) {
#pragma unroll
        for (int v = 0; v < NV; v++) acc[0][0] += __uint_as_float(buf[p][v].x ^ buf[p][v].w);
        if (u + PF < n)
#pragma unroll
          for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + (u + PF) * USTRIDE + v * 512);
      } else if (u < n) {
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
        if (u + PF < n)
#pragma unroll
          for (int v = 0; v < NV; v++) buf[p][v] = ldg_stream(wp + (u + PF) * USTRIDE + v * 512);
      }
    }
  }

  if (dbg & 1) {
    if (acc[0][0] == 12345.f) WS[0] = acc[NT - 1][3];
    return;
  }
  // Cross-slice reduction through the FP32 workspace; the last slice to arrive finalizes.
  const int n0 = rb * ROWS + warp * 16 + g;
#pragma unroll
  for (int nt = 0; nt < NT; nt++) {
    int t0 = nt * 8 + 2 * t4;
    if (t0 < m) {
      atomicAdd(reinterpret_cast<float2*>(WS + (size_t)n0 * 32 + t0), make_float2(acc[nt][0], acc[nt][1]));
      atomicAdd(reinterpret_cast<float2*>(WS + (size_t)(n0 + 8) * 32 + t0), make_float2(acc[nt][2], acc[nt][3]));
    }
  }
  __syncthreads();
  if (tid == 0) {
    __threadfence();
    s_last = (atomicAdd(CNT + rb, 1) == S - 1);
  }
  __syncthreads();
  if (!s_last) return;
  __threadfence();
  float* fin = reinterpret_cast<float*>(smem);  // [MP][ROWS]
  for (int idx = tid; idx < ROWS * MP; idx += THREADS) {
    int r = idx / MP, t = idx % MP;
    float* p = WS + (size_t)(rb * ROWS + r) * 32 + t;
    fin[t * ROWS + r] = __ldcg(p);
    __stcg(p, 0.f);
  }
  __syncthreads();
  const int e0 = EXC_PTR[rb], e1 = EXC_PTR[rb + 1];
  for (int e = e0 + tid; e < e1; e += THREADS) {
    int2 ent = EXC[e];
    int r = ent.x & 127, k = ent.x >> 7;
    bool hi = k >= Kx;
    int xk = hi ? k - Kx : k;
    float fake = __uint_as_float((hi ? (base1 & 0xFF) : (base0 & 0xFF)) << 23);
    float wv = __int_as_float(ent.y);
    for (int t = 0; t < m; t++) {
      float xv = __bfloat162float(X[(size_t)t * Kx + xk]);
      atomicAdd(&fin[t * ROWS + r], xv * wv - xv * fake);
    }
  }
  __syncthreads();
  for (int idx = tid; idx < m * ROWS; idx += THREADS) {
    int t = idx / ROWS, r = idx % ROWS;
    float v = fin[t * ROWS + r];
    size_t o = (size_t)t * N + rb * ROWS + r;
    if constexpr (OUTF32) reinterpret_cast<float*>(OUT)[o] = v;
    else reinterpret_cast<__nv_bfloat16*>(OUT)[o] = __float2bfloat16_rn(v);
  }
  if (tid == 0) CNT[rb] = 0;
}

template <int NT, bool COMP, bool OUTF32, int PF>
void launch_t(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor CNT,
              torch::Tensor EXC_PTR, torch::Tensor EXC, int m, int N, int KU, int Kx, int64_t base0,
              int64_t base1, int S, int dbg) {
  const int nmax = (KU + S - 1) / S;
  int smem = nmax * NT * 8 * 128;
  if (smem < NT * 8 * ROWS * 4) smem = NT * 8 * ROWS * 4;
  auto kern = skinny_kernel<NT, COMP, OUTF32, PF>;
  static int optin = 0;
  if (!optin) cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  TORCH_CHECK(smem <= optin, "x slice does not fit in shared memory");
  static int attr = 0;
  if (smem > attr) {
    cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attr = smem;
  }
  auto stream = at::cuda::getCurrentCUDAStream();
  kern<<<(N / ROWS) * S, THREADS, smem, stream>>>(
      W.data_ptr<uint8_t>(), reinterpret_cast<const __nv_bfloat16*>(X.data_ptr()), OUT.data_ptr(),
      WS.data_ptr<float>(), CNT.data_ptr<int>(), EXC_PTR.data_ptr<int>(),
      reinterpret_cast<const int2*>(EXC.data_ptr<int>()), m, N, KU, Kx, S, (uint32_t)base0, (uint32_t)base1, dbg);
}

void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor CNT,
            torch::Tensor EXC_PTR, torch::Tensor EXC, int64_t m, int64_t N, int64_t KU, int64_t Kx,
            int64_t base0, int64_t base1, bool comp, int64_t S, int64_t pf, int64_t dbg) {
  bool f32 = OUT.scalar_type() == torch::kFloat32;
  int nt = m <= 8 ? 1 : (m <= 16 ? 2 : 4);
#ifdef DIAG
#define L(NTV, C, F) do { if (pf == 2) launch_t<NTV, C, F, 2>(W, X, OUT, WS, CNT, EXC_PTR, EXC, m, N, KU, Kx, base0, base1, S, dbg); \
                       else if (pf == 8) launch_t<NTV, C, F, 8>(W, X, OUT, WS, CNT, EXC_PTR, EXC, m, N, KU, Kx, base0, base1, S, dbg); \
                       else launch_t<NTV, C, F, 4>(W, X, OUT, WS, CNT, EXC_PTR, EXC, m, N, KU, Kx, base0, base1, S, dbg); } while (0)
#else
#define L(NTV, C, F) launch_t<NTV, C, F, 4>(W, X, OUT, WS, CNT, EXC_PTR, EXC, m, N, KU, Kx, base0, base1, S, dbg)
#endif
#define BY_NT(C, F) { if (nt == 1) L(1, C, F); else if (nt == 2) L(2, C, F); else L(4, C, F); }
  if (comp) { if (f32) BY_NT(true, true) else BY_NT(true, false) }
  else { if (f32) BY_NT(false, true) else BY_NT(false, false) }
#undef BY_NT
#undef L
}
"""

CPP_SRC = """
void launch(torch::Tensor W, torch::Tensor X, torch::Tensor OUT, torch::Tensor WS, torch::Tensor CNT,
            torch::Tensor EXC_PTR, torch::Tensor EXC, int64_t m, int64_t N, int64_t KU, int64_t Kx,
            int64_t base0, int64_t base1, bool comp, int64_t S, int64_t pf, int64_t dbg);
"""

_ext = None


def _slices(nrb: int, ku: int, mp: int, state) -> int:
    """K-slices per row block: fill the SMs, and keep x's slice within shared memory."""
    s = max(1, min(ku, state["sms"] // nrb))
    while s < ku and -(-ku // s) * mp * 128 > state["smem"]:
        s += 1
    return s


def _module(extra=()):
    global _ext
    if _ext is None:
        _ext = load_inline(
            name="okc_bf16_proj_v2" + "".join(x.replace("-D", "_").lower() for x in extra),
            cpp_sources=CPP_SRC,
            cuda_sources=CUDA_SRC,
            functions=["launch"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo", *extra],
            verbose=False,
        )
    return _ext


ROWS, KUNIT = 128, 64
MAX_M = 32


def _frag_view(w: torch.Tensor) -> torch.Tensor:
    """[N, K] -> [rb, w, h, g, ku, j, khalf, t4, p] (row = ((rb*8+w)*2+h)*8+g, k = (((ku*4+j)*2+khalf)*4+t4)*2+p)."""
    n, k = w.shape
    return w.reshape(n // ROWS, 8, 2, 8, k // KUNIT, 4, 2, 4, 2)


def _pack_raw(w: torch.Tensor) -> torch.Tensor:
    """BF16 in mma fragment order: per (rb, ku, warp), per k16 step, 16 bytes per lane."""
    v = _frag_view(w.view(torch.int16))
    return v.permute(0, 4, 1, 5, 3, 7, 6, 2, 8).contiguous().view(torch.uint8).flatten()


def _pack_comp(w: torch.Tensor):
    """12-bit lossless packing; returns (packed bytes, base, exception CSR)."""
    n, k = w.shape
    bits = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (bits >> 7) & 0xFF
    base = max(int(e.max()) - 15, 1)
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

    idx = exc.nonzero()
    rows, cols = idx[:, 0], idx[:, 1]
    rb = rows // ROWS
    order = torch.argsort(rb * k + cols)
    rows, cols, rb = rows[order], cols[order], rb[order]
    vals = w[rows, cols].float().view(torch.int32)
    ent = torch.stack([(rows % ROWS).int() | (cols.int() << 7), vals], dim=1).contiguous()
    ptr = torch.zeros(n // ROWS + 1, dtype=torch.int32, device=w.device)
    ptr[1:] = torch.cumsum(torch.bincount(rb, minlength=n // ROWS), 0)
    return packed, base, ptr, ent


def _layer_state(w: torch.Tensor, comp: bool, kx: int):
    dev = w.device
    n, k = w.shape
    if comp:
        packed, base, ptr, ent = _pack_comp(w)
    else:
        packed, base = _pack_raw(w), 1
        ptr = torch.zeros(n // ROWS + 1, dtype=torch.int32, device=dev)
        ent = torch.zeros((1, 2), dtype=torch.int32, device=dev)
    return {"w": packed, "base": base * 0x01010101, "ptr": ptr, "exc": ent, "ku": k // KUNIT, "kx": kx,
            "comp": comp}


def _router_pack(r: torch.Tensor, comp: bool):
    """FP32 [N, K] -> BF16 hi + lo concatenated along K; both halves get their own base."""
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    k = r.shape[1]
    if not comp:
        return _layer_state(torch.cat([hi, lo], 1), False, k)
    a, b = _layer_state(hi, True, k), _layer_state(lo, True, k)
    n = r.shape[0]
    packed = torch.cat([a["w"].view(n // ROWS, k // KUNIT, -1), b["w"].view(n // ROWS, k // KUNIT, -1)], 1).flatten()
    # Exceptions from lo sit at k + K; merge the two CSR lists per row block.
    ea, eb = a["exc"], b["exc"].clone()
    eb[:, 0] += k << 7
    blocks = []
    for i in range(n // ROWS):
        blocks += [ea[a["ptr"][i]:a["ptr"][i + 1]], eb[b["ptr"][i]:b["ptr"][i + 1]]]
    ent = torch.cat(blocks) if blocks else ea
    ptr = a["ptr"] + b["ptr"]
    return {"w": packed, "base": a["base"], "base1": b["base"], "ptr": ptr, "exc": ent, "ku": 2 * k // KUNIT,
            "kx": k, "comp": True}


COMPRESS = {"qkv_gate": True, "o": True, "pooler": True, "unpooler": True}
ROUTER_COMPRESS = False


def prepare(weights: Weights):
    dev = weights["qkv_gate"].device
    _module()
    state = {"raw": {name: w.contiguous() for name, w in weights.items() if name != "router"}}
    r = weights["router"]
    hi = r.to(torch.bfloat16)
    lo = (r - hi.float()).to(torch.bfloat16)
    state["router_hi_t"] = hi.t().contiguous()
    state["router_lo_t"] = lo.t().contiguous()
    packs = {name: _layer_state(weights[name].contiguous(), COMPRESS[name], weights[name].shape[1])
             for name in COMPRESS}
    packs["router"] = _router_pack(r, ROUTER_COMPRESS)
    state["packs"] = packs
    state["ws"] = torch.zeros(8192 * 32, dtype=torch.float32, device=dev)
    state["cnt"] = torch.zeros(64, dtype=torch.int32, device=dev)
    props = torch.cuda.get_device_properties(dev)
    state["sms"] = props.multi_processor_count
    state["smem"] = props.shared_memory_per_block_optin - 1024
    torch.cuda.synchronize()
    return state


def kernel_fn(data: Inputs, state, s=None, pf=4, dbg=0) -> torch.Tensor:
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
    out = torch.empty((m, n), dtype=torch.float32 if layer == "router" else torch.bfloat16, device=x.device)
    mp = 8 if m <= 8 else (16 if m <= 16 else 32)
    s = s or _slices(n // ROWS, p["ku"], mp, state)
    _module().launch(p["w"], x.contiguous(), out, state["ws"], state["cnt"], p["ptr"], p["exc"], m, n, p["ku"],
                     p["kx"], p["base"], p.get("base1", p["base"]), p["comp"], s, pf, dbg)
    return out
