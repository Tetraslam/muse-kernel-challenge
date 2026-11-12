import modal
image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-lib-time", image=image)
CODE = r'''
import torch, statistics, deep_gemm, flashinfer
from flashinfer.gemm import tinygemm_bf16
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
def t(fn, reps=25):
    for _ in range(3): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge): pass
    def once(gg):
        flush.max(); a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); gg.replay(); b.record(); torch.cuda.synchronize(); return a.elapsed_time(b) * 1e3
    e = statistics.median(once(ge) for _ in range(20))
    return statistics.median(once(g) for _ in range(reps)) - e
def err(a, b): return float((a.float() - b.float()).norm() / b.float().norm())
for name, (n, k) in {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}.items():
    W = (torch.randn(n, k, device="cuda") / k ** .5).to(torch.bfloat16)
    Wt = W.t().contiguous()
    for m in (1, 8, 16, 32, 128, 256):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        ref = x.float() @ W.float().T
        out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        cands = {"cublas": lambda: x @ W.T, "cublasWt": lambda: x @ Wt,
                 "dg_nt": lambda: deep_gemm.bf16_gemm_nt(x, W, out),
                 "tiny": lambda: tinygemm_bf16(x, W, out)}
        for be in ("cutlass", "cudnn", "cute-dsl", "cublaslt"):
            cands["fi_" + be] = (lambda be=be: flashinfer.mm_bf16(x, W.T, backend=be))
        res = []
        for nm, fn in cands.items():
            try:
                r = fn(); torch.cuda.synchronize()
                o_ = out if r is None else r
                e = err(o_, ref)
                res.append(f"{nm} {t(fn):6.2f}{'' if e < 1e-2 else '!ERR'}")
            except Exception as ex:
                res.append(f"{nm} X")
        print(f"{name:8s} m={m:3d} " + " | ".join(res), flush=True)
# router
R = torch.randn(256, 8192, device="cuda") / 8192 ** .5
hi = R.to(torch.bfloat16); lo = (R - hi.float()).to(torch.bfloat16)
cat = torch.cat([hi, lo], 1).contiguous()
for m in (1, 8, 16, 32, 128, 256):
    x = torch.randn(m, 8192, device="cuda", dtype=torch.bfloat16)
    ref = x.double() @ R.double().T
    d = torch.empty(m, 256, device="cuda", dtype=torch.float32)
    def two():
        deep_gemm.bf16_gemm_nt(x, hi, d); deep_gemm.bf16_gemm_nt(x, lo, d, c=d)
    def catk():
        deep_gemm.bf16_gemm_nt(torch.cat([x, x], 1), cat, d)
    res = []
    for nm, fn in {"dg_two": two, "dg_cat": catk}.items():
        try:
            fn(); torch.cuda.synchronize(); e = float((d.double() - ref).norm() / ref.norm())
            res.append(f"{nm} {t(fn):6.2f} err {e:.1e}")
        except Exception as ex:
            res.append(f"{nm} X {str(ex)[:80]}")
    print(f"router m={m:3d} " + " | ".join(res), flush=True)
'''
@app.function(gpu="H200", timeout=3000)
def go():
    import subprocess
    open("/tmp/p.py", "w").write(CODE)
    r = subprocess.run(["python", "/tmp/p.py"], capture_output=True, text=True)
    return r.stdout + r.stderr[-3000:]
@app.local_entrypoint()
def main():
    print(go.remote())
