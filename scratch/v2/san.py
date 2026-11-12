import modal, pathlib
image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-san", image=image)
vol = modal.Volume.from_name("okc-torch-ext-cache", create_if_missing=True)
PROB = pathlib.Path(__file__).resolve().parents[2] / "problems" / "muse_bf16_proj" if len(pathlib.Path(__file__).resolve().parents) > 2 else None
RUN = r'''
import sys, torch, os
sys.path.insert(0, "/work")
import reference as R
import importlib.util
spec = importlib.util.spec_from_file_location("sub", "/work/submission.py"); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
W = R.generate_weights(R.WEIGHTS_SEED)
st = sub.prepare({k: v.clone() for k, v in W.items()})
for spec_ in os.environ["CASES"].split(","):
    m, l = int(spec_.split(":")[0]), spec_.split(":")[1]
    d = R.generate_input(m, l, 7)
    out = sub.kernel_fn(d, st); torch.cuda.synchronize()
    print(m, l, R.check_implementation(d, out, W)[1][:80], flush=True)
'''
@app.function(gpu="H200", timeout=1800, volumes={"/cache": vol})
def go(files: dict, cases: str):
    import subprocess, os
    os.makedirs("/work", exist_ok=True)
    for k, v in files.items(): open(f"/work/{k}", "w").write(v)
    open("/work/run.py", "w").write(RUN)
    env = dict(os.environ, CASES=cases, TORCH_EXTENSIONS_DIR="/cache/torch_ext")
    r = subprocess.run(["compute-sanitizer", "--tool", "memcheck", "--print-limit", "5", "python", "/work/run.py"], capture_output=True, text=True, env=env, cwd="/work")
    vol.commit()
    return r.stdout[-6000:] + r.stderr[-3000:]
@app.local_entrypoint()
def main(sub: str, cases: str):
    p = pathlib.Path(__file__).resolve().parents[2] / "problems" / "muse_bf16_proj"
    files = {"submission.py": pathlib.Path(sub).read_text(), **{n: (p / n).read_text() for n in ("reference.py", "checks.py")}}
    print(go.remote(files, cases))
