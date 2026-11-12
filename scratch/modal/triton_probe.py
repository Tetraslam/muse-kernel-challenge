import modal

image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-triton-probe", image=image)

CODE = r'''
import torch, statistics, triton, triton.language as tl
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

@triton.jit
def splitk(x_ptr, w_ptr, o_ptr, M, N, K, KS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid_n = tl.program_id(0); pid_k = tl.program_id(1); pid_m = tl.program_id(2)
    rm = pid_m * BM + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    kchunk = K // KS
    k0 = pid_k * kchunk
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for kk in range(0, kchunk, BK):
        rk = k0 + kk + tl.arange(0, BK)
        a = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=rm[:, None] < M, other=0.)
        b = tl.load(w_ptr + rn[None, :] * K + rk[:, None])   # W [N, K] -> B [BK, BN]
        acc += tl.dot(a, b)
    tl.atomic_add(o_ptr + rm[:, None] * N + rn[None, :], acc, mask=rm[:, None] < M, sem="relaxed")

def run(x, W, o, KS, BM, BN, BK, warps, stages):
    M, K = x.shape; N = W.shape[0]
    grid = (N // BN, KS, triton.cdiv(M, BM))
    splitk[grid](x, W, o, M, N, K, KS=KS, BM=BM, BN=BN, BK=BK, num_warps=warps, num_stages=stages)

for name, (n, k) in {"qkv": (2304, 8192), "o": (8192, 1024), "pooler": (4096, 8192), "unpooler": (8192, 4096)}.items():
    W = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    for m in (16, 32, 128, 256):
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        o = torch.zeros(m, n, device="cuda", dtype=torch.float32)
        ref = (x.float() @ W.float().T)
        cub = t(lambda: x @ W.T)
        best = []
        BMs = [16, 32] if m <= 32 else [64, 128]
        for BM in BMs:
            if BM < m and m <= 32: continue
            for BN in (64, 128):
                for BK in (64, 128):
                    for KS in (1, 2, 4, 8, 16):
                        if k // KS < BK or (k // KS) % BK: continue
                        ctas = (n // BN) * KS * ((m + BM - 1) // BM)
                        if ctas < 60 or ctas > 600: continue
                        for warps, stages in ((4, 4), (4, 6), (8, 4)):
                            try:
                                tt = t(lambda: run(x, W, o, KS, BM, BN, BK, warps, stages))
                                best.append((tt, BM, BN, BK, KS, warps, stages))
                            except Exception as e:
                                pass
        best.sort()
        o.zero_(); b = best[0]; run(x, W, o, *b[4:5], *b[1:4], *b[5:]) if False else None
        print(f"{name:8s} m={m:3d} cublas {cub:6.2f} | triton best " + " ; ".join(f"{b[0]:6.2f} BM{b[1]} BN{b[2]} BK{b[3]} KS{b[4]} w{b[5]} s{b[6]}" for b in best[:3]), flush=True)
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
