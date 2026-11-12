"""H200 streaming-read ceiling under judge-style timing (graph replay, cold L2, empty graph subtracted).

  modal run scratch/v2/bw.py
"""
import modal

image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-bw", image=image)
vol = modal.Volume.from_name("okc-torch-ext-cache", create_if_missing=True)

CODE = r'''
import torch, statistics, os
from torch.utils.cpp_extension import load_inline
os.environ["TORCH_EXTENSIONS_DIR"] = "/cache/torch_ext"
CUDA = r"""
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <stdint.h>

template <int U>
__global__ void ldg_kernel(const uint4* __restrict__ p, size_t n16, uint32_t* out) {
  size_t per = (n16 + gridDim.x - 1) / gridDim.x;
  size_t b = per * blockIdx.x, e = min(n16, b + per);
  uint32_t acc = 0;
  for (size_t i = b + threadIdx.x; i < e; i += (size_t)blockDim.x * U) {
    uint4 v[U];
#pragma unroll
    for (int u = 0; u < U; u++) {
      size_t j = i + (size_t)u * blockDim.x;
      if (j < e) asm volatile("ld.global.nc.L1::no_allocate.L2::256B.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v[u].x), "=r"(v[u].y), "=r"(v[u].z), "=r"(v[u].w) : "l"(p + j));
      else v[u] = make_uint4(0,0,0,0);
    }
#pragma unroll
    for (int u = 0; u < U; u++) acc ^= v[u].x ^ v[u].y ^ v[u].z ^ v[u].w;
  }
  if (acc == 0x12345678) out[0] = acc;
}

__device__ __forceinline__ void mbar_init(uint64_t* b, int c) {
  asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(c));
}
__device__ __forceinline__ void mbar_expect(uint64_t* b, uint32_t bytes) {
  asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"((uint32_t)__cvta_generic_to_shared(b)), "r"(bytes));
}
__device__ __forceinline__ void mbar_wait(uint64_t* b, int phase) {
  uint32_t a = (uint32_t)__cvta_generic_to_shared(b);
  asm volatile("{\n.reg .pred p;\nW: mbarrier.try_wait.parity.shared.b64 p, [%0], %1;\n@!p bra W;\n}" :: "r"(a), "r"(phase));
}
__device__ __forceinline__ void mbar_arrive(uint64_t* b) {
  asm volatile("mbarrier.arrive.shared.b64 _, [%0];" :: "r"((uint32_t)__cvta_generic_to_shared(b)));
}
__device__ __forceinline__ void bulk_g2s(void* dst, const void* src, uint32_t bytes, uint64_t* b) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
    :: "r"((uint32_t)__cvta_generic_to_shared(dst)), "l"(src), "r"(bytes), "r"((uint32_t)__cvta_generic_to_shared(b)) : "memory");
}

// one CTA per SM; warp 0 lane 0 produces; all warps consume.
__global__ void tma_kernel(const uint8_t* __restrict__ p, size_t nbytes, int chunk, int stages, uint32_t* out) {
  extern __shared__ __align__(128) uint8_t smem[];
  uint64_t* full = (uint64_t*)smem;
  uint64_t* empty = full + 32;
  uint8_t* buf = smem + 1024;
  size_t per = (nbytes / gridDim.x + 1023) & ~(size_t)1023;
  size_t b = per * blockIdx.x, e = min(nbytes, b + per);
  int nch = b < e ? (int)((e - b + chunk - 1) / chunk) : 0;
  int nw = blockDim.x / 32;
  if (threadIdx.x == 0) {
    for (int s = 0; s < stages; s++) { mbar_init(full + s, 1); mbar_init(empty + s, nw); }
    asm volatile("fence.mbarrier_init.release.cluster;");
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int c = 0; c < min(stages, nch); c++) {
      uint32_t bytes = (uint32_t)min((size_t)chunk, e - (b + (size_t)c * chunk));
      mbar_expect(full + c, bytes);
      bulk_g2s(buf + (size_t)c * chunk, p + b + (size_t)c * chunk, bytes, full + c);
    }
  }
  uint32_t acc = 0;
  int lane = threadIdx.x & 31;
  for (int c = 0; c < nch; c++) {
    int s = c % stages, ph = (c / stages) & 1;
    mbar_wait(full + s, ph);
    uint32_t bytes = (uint32_t)min((size_t)chunk, e - (b + (size_t)c * chunk));
    const uint4* q = (const uint4*)(buf + (size_t)s * chunk);
    for (int i = threadIdx.x; i < (int)(bytes / 16); i += blockDim.x) { uint4 v = q[i]; acc ^= v.x ^ v.y ^ v.z ^ v.w; }
    __syncwarp();
    if (lane == 0) mbar_arrive(empty + s);
    if (threadIdx.x == 0 && c + stages < nch) {
      mbar_wait(empty + s, ph);
      size_t off = b + (size_t)(c + stages) * chunk;
      uint32_t nb = (uint32_t)min((size_t)chunk, e - off);
      mbar_expect(full + s, nb);
      bulk_g2s(buf + (size_t)s * chunk, p + off, nb, full + s);
    }
  }
  if (acc == 0x12345678) out[0] = acc;
}

void ldg(torch::Tensor p, int64_t grid, int64_t threads, int64_t u, torch::Tensor out) {
  auto st = at::cuda::getCurrentCUDAStream();
  size_t n16 = p.numel() / 16;
  if (u == 4) ldg_kernel<4><<<grid, threads, 0, st>>>((const uint4*)p.data_ptr(), n16, (uint32_t*)out.data_ptr());
  else if (u == 8) ldg_kernel<8><<<grid, threads, 0, st>>>((const uint4*)p.data_ptr(), n16, (uint32_t*)out.data_ptr());
  else ldg_kernel<16><<<grid, threads, 0, st>>>((const uint4*)p.data_ptr(), n16, (uint32_t*)out.data_ptr());
}
void tma(torch::Tensor p, int64_t grid, int64_t threads, int64_t chunk, int64_t stages, torch::Tensor out) {
  auto st = at::cuda::getCurrentCUDAStream();
  int sm = 1024 + chunk * stages;
  cudaFuncSetAttribute(tma_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, sm);
  tma_kernel<<<grid, threads, sm, st>>>((const uint8_t*)p.data_ptr(), p.numel(), chunk, stages, (uint32_t*)out.data_ptr());
}
"""
CPP = "void ldg(torch::Tensor p, int64_t grid, int64_t threads, int64_t u, torch::Tensor out);\nvoid tma(torch::Tensor p, int64_t grid, int64_t threads, int64_t chunk, int64_t stages, torch::Tensor out);"
mod = load_inline("bw_probe_v1", CPP, CUDA, functions=["ldg", "tma"], extra_cuda_cflags=["-O3", "-arch=sm_90a"])
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
out = torch.zeros(4, dtype=torch.int32, device="cuda")
def once(g):
    flush.max(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) * 1e3
def t(fn, reps=30):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return statistics.median(once(g) for _ in range(reps)) - E
ge = torch.cuda.CUDAGraph()
with torch.cuda.graph(ge): pass
E = statistics.median(once(ge) for _ in range(50))
print(f"empty {E:.2f}")
SM = torch.cuda.get_device_properties(0).multi_processor_count
for mb in (6.3, 8.4, 12.6, 16.8, 28.3, 37.7, 50.3, 67.1):
    buf = torch.randint(0, 255, (int(mb * 1e6) // 1024 * 1024,), dtype=torch.uint8, device="cuda")
    res = []
    for grid, th, u in ((SM, 1024, 4), (SM, 1024, 8), (SM*2, 512, 8), (SM*4, 256, 8), (SM*8, 256, 4), (SM, 512, 16)):
        tt = t(lambda: mod.ldg(buf, grid, th, u, out))
        res.append(f"ldg{grid//SM}x{th}u{u} {tt:5.2f}")
    for chunk, stages, th in ((16384, 8, 256), (32768, 6, 256), (8192, 16, 256), (16384, 12, 512), (65536, 3, 256)):
        tt = t(lambda: mod.tma(buf, SM, th, chunk, stages, out))
        res.append(f"tma{chunk//1024}kx{stages} {tt:5.2f}")
    best = min(float(r.split()[-1]) for r in res)
    print(f"{mb:5.1f}MB best {best:5.2f}us {mb/best:4.2f}TB/s | " + " | ".join(res), flush=True)
'''


@app.function(gpu="H200", timeout=1800, volumes={"/cache": vol})
def go():
    import subprocess
    open("/tmp/p.py", "w").write(CODE)
    r = subprocess.run(["python", "/tmp/p.py"], capture_output=True, text=True)
    vol.commit()
    return r.stdout + r.stderr[-3000:]


@app.local_entrypoint()
def main():
    print(go.remote())
