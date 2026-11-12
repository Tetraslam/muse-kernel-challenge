import modal
image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-lib-sig", image=image)
CODE = r'''
import inspect, deep_gemm, flashinfer, flashinfer.gemm as fg
for f in (deep_gemm.bf16_gemm_nt, deep_gemm.bf16_gemm_tn, flashinfer.mm_bf16, fg.tinygemm_bf16, fg.routergemm, fg.mm_M1_16_K7168_N256_bf16, flashinfer.trtllm_low_latency_gemm):
    try: sig = str(inspect.signature(f))
    except Exception as e: sig = f"(no sig: {e})"
    doc = (f.__doc__ or "").strip().replace("\n", " ")[:600]
    print(f"== {f.__module__}.{f.__name__}{sig}\n   {doc}\n")
'''
@app.function(gpu="H200", timeout=900)
def go():
    import subprocess
    open("/tmp/p.py", "w").write(CODE)
    r = subprocess.run(["python", "/tmp/p.py"], capture_output=True, text=True)
    return r.stdout + r.stderr[-3000:]
@app.local_entrypoint()
def main():
    print(go.remote())
