import modal
image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-launch", image=image)
vol = modal.Volume.from_name("okc-torch-ext-cache", create_if_missing=True)
CODE = r'''
import torch, statistics, os
from torch.utils.cpp_extension import load_inline
os.environ["TORCH_EXTENSIONS_DIR"] = "/cache/torch_ext"
CUDA = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
__global__ void __launch_bounds__(1024, 1) ek(int* out, int n) {
  extern __shared__ int sm[];
  if (n > 0) for (int i = threadIdx.x; i < n; i += blockDim.x) sm[i] = i;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = n > 0 ? sm[n - 1] : 1;
}
__global__ void __launch_bounds__(1024, 1) ek2(int* out, int n) {
  extern __shared__ int sm[];
  if (n > 0) for (int i = threadIdx.x; i < n; i += blockDim.x) sm[i] = i;
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = n > 0 ? sm[n - 1] : 1;
}
void run(torch::Tensor out, int64_t grid, int64_t threads, int64_t smem, int64_t maxattr, int64_t cluster, int64_t which) {
  auto st = at::cuda::getCurrentCUDAStream();
  auto k = which ? ek2 : ek;
  if (maxattr) cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, maxattr);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(grid); cfg.blockDim = dim3(threads); cfg.dynamicSmemBytes = smem; cfg.stream = st;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = cluster > 0 ? cluster : 1; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr; cfg.numAttrs = cluster > 0 ? 1 : 0;
  cudaLaunchKernelEx(&cfg, k, (int*)out.data_ptr(), (int)(smem / 4));
}

template <int N>
__global__ void __launch_bounds__(512, 1) codek(int* out, int v) {
  uint32_t a = v + threadIdx.x;
#pragma unroll
  for (int i = 0; i < N; i++) asm volatile("xor.b32 %0, %0, %1;" : "+r"(a) : "r"((uint32_t)(i * 7 + 1)));
  if (a == 0x1234567) out[0] = a;
}
void code(torch::Tensor out, int64_t n) {
  auto st = at::cuda::getCurrentCUDAStream();
  if (n == 0) codek<1><<<132, 512, 0, st>>>((int*)out.data_ptr(), 1);
  else if (n == 1) codek<256><<<132, 512, 0, st>>>((int*)out.data_ptr(), 1);
  else if (n == 2) codek<512><<<132, 512, 0, st>>>((int*)out.data_ptr(), 1);
  else if (n == 3) codek<1024><<<132, 512, 0, st>>>((int*)out.data_ptr(), 1);
  else codek<2048><<<132, 512, 0, st>>>((int*)out.data_ptr(), 1);
}

template <int NB, int BL>
__global__ void __launch_bounds__(512, 1) jumpk(int* out, int v) {
  uint32_t a = v + threadIdx.x;
#pragma unroll
  for (int b = 0; b < NB; b++) {
    if (v & (1 << b)) {
#pragma unroll
      for (int i = 0; i < BL; i++) asm volatile("xor.b32 %0, %0, %1;" : "+r"(a) : "r"((uint32_t)(i * 7 + b * 13 + 1)));
    }
    asm volatile("add.u32 %0, %0, 3;" : "+r"(a));
  }
  if (a == 0x1234567) out[0] = a;
}
void jump(torch::Tensor out, int64_t n, int64_t v) {
  auto st = at::cuda::getCurrentCUDAStream();
  if (n == 1) jumpk<1, 256><<<132, 512, 0, st>>>((int*)out.data_ptr(), v);
  else if (n == 2) jumpk<2, 256><<<132, 512, 0, st>>>((int*)out.data_ptr(), v);
  else if (n == 4) jumpk<4, 256><<<132, 512, 0, st>>>((int*)out.data_ptr(), v);
  else if (n == 8) jumpk<8, 256><<<132, 512, 0, st>>>((int*)out.data_ptr(), v);
  else jumpk<8, 32><<<132, 512, 0, st>>>((int*)out.data_ptr(), v);
}

struct P32 { int a[64]; };
__global__ void __launch_bounds__(512, 1) pk(int* out, P32 p, int which) {
  int v = 0;
  if (which & 1) v += p.a[0];
  if (which & 2) v += p.a[15];
  if (which & 4) v += p.a[31];
  if (which & 8) v += p.a[63];
  if (v == 0x1234567) out[0] = v;
}
void par(torch::Tensor out, int64_t which) {
  auto st = at::cuda::getCurrentCUDAStream();
  P32 p; for (int i = 0; i < 64; i++) p.a[i] = i;
  pk<<<132, 512, 0, st>>>((int*)out.data_ptr(), p, (int)which);
}
"""
CPP = "void par(torch::Tensor out, int64_t which);\nvoid jump(torch::Tensor out, int64_t n, int64_t v);\nvoid code(torch::Tensor out, int64_t n);\nvoid run(torch::Tensor out, int64_t grid, int64_t threads, int64_t smem, int64_t maxattr, int64_t cluster, int64_t which);"
mod = load_inline("launch_probe_v6", CPP, CUDA, functions=["run", "code", "jump", "par"], extra_cuda_cflags=["-O3", "-arch=sm_90a"])
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
out = torch.zeros(4096, dtype=torch.int32, device="cuda")
def once(g):
    flush.max(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) * 1e3
def t(fn, reps=40):
    fn(); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    return statistics.median(once(g) for _ in range(reps)) - E
ge = torch.cuda.CUDAGraph()
with torch.cuda.graph(ge): pass
E = statistics.median(once(ge) for _ in range(50))
print(f"empty {E:.2f}")
for n in ():
    for v in (0, 255):
        print("jump", n, v, f"{t(lambda: mod.jump(out, n, v)):5.2f}", flush=True)
for n in range(1):
    print("code", n, f"{t(lambda: mod.code(out, n)):5.2f}", flush=True)
for (grid, th, smem, maxattr, cl, w) in [(132,1024,kb*1024,227*1024,0,1) for kb in (0,4,8,16,24,32,48,64,80,100,132,164,200)] + [(132,1024,kb*1024,227*1024,0,0) for kb in (0, 8, 32)] + [][:0] + [(132,1024,0,0,0,0),(132,512,0,0,0,0),(132,512,2048,0,0,0),(132,512,2048,200*1024,0,1),
        (132,512,100*1024,200*1024,0,1),(132,512,2048,0,1,0),(132,512,2048,0,2,0),(128,512,2048,0,4,0),(120,512,2048,0,8,0),(132,128,0,0,0,0),(1,32,0,0,0,0)]:
    tt = t(lambda: mod.run(out, grid, th, smem, maxattr, cl, w))
    print(f"grid {grid} th {th} smem {smem} maxattr {maxattr} cluster {cl}: {tt:5.2f}", flush=True)
x = torch.randn(4096, device="cuda")
print("torch fill_", t(lambda: x.fill_(1.0)))
'''
@app.function(gpu="H200", timeout=1200, volumes={"/cache": vol})
def go():
    import subprocess
    open("/tmp/p.py", "w").write(CODE)
    subprocess.run(["python", "-u", "/tmp/p.py"])
    vol.commit()
@app.local_entrypoint()
def main():
    go.remote()
