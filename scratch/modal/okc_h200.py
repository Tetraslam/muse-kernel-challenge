"""H200 dev loop in the judge's image: timing like the judge (graph replay, cold L2, paired with
SGLang's kernel) and ncu profiles.

  modal run okc_h200.py::bench --sub ../../problems/muse_bf16_proj/submission.py [--shapes "1:qkv_gate,8:o"]
  modal run okc_h200.py::profile --sub ... --shape 1:qkv_gate [--kernel skinny] [--section full]
"""

import pathlib

import modal

image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-h200", image=image)
vol = modal.Volume.from_name("okc-torch-ext-cache", create_if_missing=True)
_here = pathlib.Path(__file__).resolve()
PROB = (_here.parents[2] if len(_here.parents) > 2 else _here.parent) / "problems" / "muse_bf16_proj"
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]
SCORED = [f"{m}:{l}" for m in (1, 16, 32, 8, 128, 256) for l in LAYERS]

RUNNER = r'''
import importlib.util, sys, os, json, statistics
import torch
sys.path.insert(0, "/work")
import reference as R

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

sub = load("sub", "/work/submission.py")
base = load("base", "/work/baseline.py")
W = R.generate_weights(R.WEIGHTS_SEED)
st_sub = sub.prepare({k: v.clone() for k, v in W.items()})
st_base = base.prepare({k: v.clone() for k, v in W.items()})
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

mode = sys.argv[1]
shapes = [(int(x.split(":")[0]), x.split(":")[1]) for x in sys.argv[2].split(",")]
if mode == "bench":
    kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    fn_sub = lambda d, st: sub.kernel_fn(d, st, **kw.get(f"{d['config']['m']}:{d['config']['layer']}", {}))
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge): pass
    empty = statistics.median(t_once(ge) for _ in range(50))
    print(f"empty graph {empty:.2f} us (subtracted)")
    loads = {}
    for i, (m, layer) in enumerate(shapes):
        data = R.generate_input(m, layer, 1000 + i)
        gs, out = graph(fn_sub, data, st_sub)
        gb, _ = graph(base.kernel_fn, data, st_base)
        gs.replay(); torch.cuda.synchronize()
        ok, msg = R.check_implementation(data, out, W)
        ratios, ts, tb = [], [], []
        for r in range(30):
            order = (gs, gb) if r % 2 == 0 else (gb, gs)
            a = t_once(order[0]); b = t_once(order[1])
            s_, b_ = (a, b) if r % 2 == 0 else (b, a)
            ts.append(s_ - empty); tb.append(b_ - empty); ratios.append((b_ - empty) / (s_ - empty))
        sp = statistics.median(ratios)
        print(f"m={m:4d} {layer:9s} sglang {statistics.median(tb):7.2f}  ours {statistics.median(ts):7.2f}  x{sp:.3f}  {'ok' if ok else 'FAIL ' + msg}", flush=True)
elif mode == "sweep":
    ge = torch.cuda.CUDAGraph()
    with torch.cuda.graph(ge): pass
    empty = statistics.median(t_once(ge) for _ in range(50))
    variants = json.loads(sys.argv[3])
    for i, (m, layer) in enumerate(shapes):
        data = R.generate_input(m, layer, 2000 + i)
        gb, _ = graph(base.kernel_fn, data, st_base)
        for kw in variants:
            try:
                gs, out = graph(lambda d, st: sub.kernel_fn(d, st, **kw), data, st_sub)
            except Exception as e:  # noqa: BLE001
                print(f"m={m:4d} {layer:9s} {json.dumps(kw):40s} ERR {str(e)[:60]}"); continue
            gs.replay(); torch.cuda.synchronize()
            ok, msg = R.check_implementation(data, out, W)
            ratios, ts = [], []
            for r in range(24):
                if r % 2 == 0: a = t_once(gs); b = t_once(gb)
                else: b = t_once(gb); a = t_once(gs)
                ts.append(a - empty); ratios.append((b - empty) / (a - empty))
            print(f"m={m:4d} {layer:9s} {json.dumps(kw):40s} ours {statistics.median(ts):7.2f} x{statistics.median(ratios):.3f} {'ok' if ok else 'FAIL'}", flush=True)
elif mode == "ts":
    kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    names = ["x", "loop", "part", "exc", "pub", "arrive", "end"]
    for m, layer in shapes:
        data = R.generate_input(m, layer, 7)
        for _ in range(3): sub.kernel_fn(data, st_sub, **{k: v for k, v in kw.items() if k != "dbg"})
        if os.environ.get("WARM") != "1":
            flush.max()
        torch.cuda.synchronize()
        kw2 = dict(kw); extra = kw2.pop("dbg", 0)
        sub.kernel_fn(data, st_sub, dbg=64 | extra, **kw2); torch.cuda.synchronize()
        n = data["config"]["n"]
        plan = kw.get("plan") or sub._plan(st_sub, layer, m, n)
        grid = (n // 128) * plan[0] * (-(-m // 32))
        ts = sub._module().get_ts(min(grid, 1024)).double()
        c0 = ts[:, 1:2]
        rel = (ts[:, 2:9] - c0) / 1.98e3  # us at ~1.98 GHz
        g0 = ts[:, 0].min()
        cols = []
        for i, nm in enumerate(names):
            c = rel[:, i][ts[:, i + 2] > 0]
            c = c[c >= 0]
            if c.numel():
                cols.append(f"{nm} {float(c.median()):5.2f}/{float(c.max()):5.2f}")
        start = (ts[:, 0] - g0) / 1e3
        end = (ts[:, 9] - g0) / 1e3
        print(f"m={m:3d} {layer:9s} plan={plan} grid={grid} start skew {float(start.max()):4.2f} end med/max {float(end.median()):5.2f}/{float(end.max()):5.2f} | " + " | ".join(cols), flush=True)
else:
    m, layer = shapes[0]
    data = R.generate_input(m, layer, 7)
    kw = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    for _ in range(5): out = sub.kernel_fn(data, st_sub, **kw)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    sub.kernel_fn(data, st_sub, **kw)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
'''


def _setup(files: dict):
    import os
    os.makedirs("/work", exist_ok=True)
    os.environ["TORCH_EXTENSIONS_DIR"] = "/cache/torch_ext"
    for name, text in files.items():
        open(f"/work/{name}", "w").write(text)
    open("/work/runner.py", "w").write(RUNNER)


@app.function(gpu="H200", timeout=1800, volumes={"/cache": vol})
def run_bench(files: dict, shapes: str, kw: str = "{}") -> str:
    import subprocess
    _setup(files)
    r = subprocess.run(["python", "/work/runner.py", "bench", shapes, kw], capture_output=True, text=True, cwd="/work")
    vol.commit()
    return r.stdout + r.stderr[-3000:]


@app.function(gpu="H200", timeout=1800, volumes={"/cache": vol})
def run_ts(files: dict, shapes: str, kw: str, warm: bool = False) -> str:
    import os, subprocess
    _setup(files)
    env = dict(os.environ, WARM="1" if warm else "0")
    r = subprocess.run(["python", "/work/runner.py", "ts", shapes, kw], capture_output=True, text=True, cwd="/work",
                       env=env)
    vol.commit()
    return r.stdout + r.stderr[-3000:]


@app.function(gpu="H200", timeout=1800, volumes={"/cache": vol})
def run_profile(files: dict, shape: str, kernel: str, section: str, kw: str) -> str:
    import subprocess
    _setup(files)
    sec = ["--set", "full"] if section == "full" else ["--section", section]
    cmd = ["ncu", "--clock-control", "none", "--profile-from-start", "off", "-k", f"regex:{kernel}", *sec,
           "--import-source", "yes", "-o", "/work/rep", "python", "/work/runner.py", "profile", shape, kw]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd="/work")
    out = r.stdout[-2000:] + r.stderr[-2000:]
    d = subprocess.run(["ncu", "-i", "/work/rep.ncu-rep", "--page", "details"], capture_output=True, text=True)
    src = subprocess.run(["ncu", "-i", "/work/rep.ncu-rep", "--page", "source", "--csv", "--print-source", "sass"],
                         capture_output=True, text=True)
    vol.commit()
    return out + "\n" + d.stdout + "\n==SOURCE==\n" + src.stdout


def _files(sub: str):
    return {"submission.py": pathlib.Path(sub).read_text(),
            **{n: (PROB / n).read_text() for n in ("reference.py", "checks.py", "baseline.py")}}


@app.local_entrypoint()
def bench(sub: str, shapes: str = ",".join(SCORED), kw: str = "{}"):
    """kw: JSON {"m:layer": {kernel_fn kwargs}}; a shape may repeat as "m:layer@tag" with "m:layer@tag" keys."""
    print(run_bench.remote(_files(sub), shapes, kw))


@app.function(gpu="H200", timeout=1800, volumes={"/cache": vol})
def run_sweep(files: dict, shapes: str, variants: str) -> str:
    import subprocess
    _setup(files)
    r = subprocess.run(["python", "/work/runner.py", "sweep", shapes, variants], capture_output=True, text=True,
                       cwd="/work")
    vol.commit()
    return r.stdout + r.stderr[-3000:]


@app.local_entrypoint()
def sweep(sub: str, shapes: str, variants: str):
    print(run_sweep.remote(_files(sub), shapes, variants))


@app.local_entrypoint()
def ts(sub: str, shapes: str = "1:qkv_gate", kw: str = "{}", warm: bool = False):
    print(run_ts.remote(_files(sub), shapes, kw, warm))


@app.local_entrypoint()
def profile(sub: str, shape: str = "1:qkv_gate", kernel: str = "skinny|gemv", section: str = "full", kw: str = "{}",
            out: str = "/tmp/opencode/ncu.txt"):
    text = run_profile.remote(_files(sub), shape, kernel, section, kw)
    pathlib.Path(out).write_text(text)
    print(text.split("==SOURCE==")[0][-12000:])
