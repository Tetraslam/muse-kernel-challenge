

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
