import modal

image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-cublas-probe", image=image)

CODE = r'''
import torch, statistics
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
def t(fn, reps=30):
    for _ in range(3): fn()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge): pass
    def once(gg):
        flush.max(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); gg.replay(); b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) * 1e3
    e = statistics.median(once(ge) for _ in range(20))
    return statistics.median(once(g) for _ in range(reps)) - e
print(torch.__version__)
import triton; print("triton", triton.__version__)
for name, (n, k) in {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}.items():
    W = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    Wt = W.t().contiguous()
    for m in (16, 32, 128, 256):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        r = {
            "x@W.T": t(lambda: x @ W.T),
            "(W@x.T).T": t(lambda: (W @ x.T).t().contiguous()),
            "W@x.T only": t(lambda: W @ x.T),
            "x@Wt": t(lambda: x @ Wt),
            "mm fp32out": t(lambda: torch.mm(x, W.T, out_dtype=torch.float32)),
        }
        print(f"{name:8s} m={m:3d} " + " | ".join(f"{a} {b:6.2f}" for a, b in r.items()), flush=True)
'''

@app.function(gpu="H200", timeout=1200)
def run():
    import subprocess
    open("/tmp/p.py", "w").write(CODE)
    r = subprocess.run(["python", "/tmp/p.py"], capture_output=True, text=True)
    return r.stdout + r.stderr[-2000:]

@app.local_entrypoint()
def main():
    print(run.remote())
