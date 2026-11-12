"""H200 sweep in the judge's image: graph replay, cold L2, paired with SGLang's kernel.

  modal run scratch/v2/h200.py --sub scratch/v2/sub.py --plan plan.json
plan.json: {"m:layer": [kwargs, ...], ...}; an empty kwargs dict is the default config.
"""
import json
import pathlib

import modal

image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-h200-v2", image=image)
vol = modal.Volume.from_name("okc-torch-ext-cache", create_if_missing=True)
_here = pathlib.Path(__file__).resolve()
PROB = (_here.parents[2]) / "problems" / "muse_bf16_proj"

RUNNER = r'''
import importlib.util, sys, os, json, statistics, time
import torch
sys.path.insert(0, "/work")
import reference as R

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

t0 = time.time()
sub = load("sub", "/work/submission.py")
base = load("base", "/work/baseline.py")
W = R.generate_weights(R.WEIGHTS_SEED)
st_sub = sub.prepare({k: v.clone() for k, v in W.items()})
st_base = base.prepare({k: v.clone() for k, v in W.items()})
print(f"prepare+compile {time.time() - t0:.1f}s", flush=True)
flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")

def graph(fn, data, st):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3): fn(data, st)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out = fn(data, st)
    return g, out

def t_once(g):
    flush.max()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record(); g.replay(); b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3

ge = torch.cuda.CUDAGraph()
with torch.cuda.graph(ge): pass
empty = statistics.median(t_once(ge) for _ in range(50))
plan = json.loads(open("/work/plan.json").read())
REPS = int(os.environ.get("REPS", "24"))
if os.environ.get("TSMODE") == "1":
    names = ["entry", "synced", "xready", "loopend", "reduced", "end"]
    for i, (shape, variants) in enumerate(plan.items()):
        m, layer = int(shape.split(":")[0]), shape.split(":")[1]
        data = R.generate_input(m, layer, 3000 + i)
        for kw in variants:
            kw = dict(kw); kw["dbg"] = kw.get("dbg", 0) | 512
            g, out = graph(lambda d, st: sub.kernel_fn(d, st, **kw), data, st_sub)
            rows = []
            for r in range(5):
                sub._module().clear_ts(); torch.cuda.synchronize()
                tt = t_once(g) - empty
                ts = sub._module().get_ts().view(1024, 8)[:, [0, 1, 2, 3, 4, 5]].double()
                live = ts[:, 0] > 0
                ts = ts[live]
                t0 = ts[:, 0].min()
                rel = (ts - t0) / 1e3
                rel[ts == 0] = float("nan")
                rows.append((tt, rel))
            tt, rel = sorted(rows, key=lambda x: x[0])[2]
            desc = " ".join(f"{n} {float(rel[:, j].nanmedian()):5.2f}/{float(rel[:, j].nan_to_num(-1).max()):5.2f}" for j, n in enumerate(names))
            print(f"m={m:4d} {layer:9s} {json.dumps(kw):40s} t {tt:5.2f} ctas {int(live.sum())} | {desc}", flush=True)
    sys.exit(0)
for i, (shape, variants) in enumerate(plan.items()):
    m, layer = int(shape.split(":")[0]), shape.split(":")[1]
    data = R.generate_input(m, layer, 3000 + i)
    gb, _ = graph(base.kernel_fn, data, st_base)
    tb_all = []
    for kw in variants:
        try:
            gs, out = graph(lambda d, st: sub.kernel_fn(d, st, **kw), data, st_sub)
            gs.replay(); torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001
            print(f"m={m:4d} {layer:9s} {json.dumps(kw):44s} ERR {str(e)[:100]}", flush=True); continue
        ok, msg = R.check_implementation(data, out, W)
        ratios, ts, tb = [], [], []
        for r in range(REPS):
            if r % 2 == 0: a = t_once(gs); b = t_once(gb)
            else: b = t_once(gb); a = t_once(gs)
            ts.append(a - empty); tb.append(b - empty); ratios.append((b - empty) / (a - empty))
        print(f"m={m:4d} {layer:9s} {json.dumps(kw):44s} sglang {statistics.median(tb):6.2f} ours {statistics.median(ts):6.2f}"
              f" x{statistics.median(ratios):.3f} {'ok' if ok else 'FAIL ' + msg[:60]}", flush=True)
'''


@app.function(gpu="H200", timeout=1800, volumes={"/cache": vol})
def run(files: dict, reps: int = 24, tsmode: int = 0) -> str:
    import os
    import subprocess
    os.makedirs("/work", exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = "/cache/torch_ext"
    for name, text in files.items():
        open(f"/work/{name}", "w").write(text)
    open("/work/runner.py", "w").write(RUNNER)
    try:
        subprocess.run(["python", "-u", "/work/runner.py"], cwd="/work", env=dict(os.environ, REPS=str(reps), TSMODE=str(tsmode)),
                       timeout=1500)
    finally:
        vol.commit()
    return ""


@app.local_entrypoint()
def main(sub: str, plan: str, reps: int = 24, ts: int = 0):
    p = pathlib.Path(plan)
    text = p.read_text() if p.exists() else plan
    files = {"submission.py": pathlib.Path(sub).read_text(), "plan.json": text,
             **{n: (PROB / n).read_text() for n in ("reference.py", "checks.py", "baseline.py")}}
    print(run.remote(files, reps, ts))
