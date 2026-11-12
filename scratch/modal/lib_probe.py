import modal

image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-lib-probe", image=image)

CODE = r'''
import importlib, pkgutil
for mod in ("deep_gemm", "flashinfer", "flashinfer.gemm", "sgl_kernel", "cutlass", "cutlass.cute", "tilelang"):
    try:
        m = importlib.import_module(mod)
        names = [n for n in dir(m) if any(s in n.lower() for s in ("gemm", "mm", "bf16", "linear", "router"))]
        print(f"{mod} {getattr(m, '__version__', '')}: {names[:60]}")
    except Exception as e:
        print(f"{mod}: ERR {e}")
import os, glob
for pat in ("/opt/sglang/lib/python3.12/site-packages/cutlass*", "/usr/local/lib/python3.12/dist-packages/cutlass*",
            "/opt/**/examples/**/hopper/*gemm*.py", "/sgl-workspace/**/hopper/*gemm*.py"):
    print(pat, glob.glob(pat, recursive=True)[:10])
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
