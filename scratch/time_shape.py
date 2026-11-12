import importlib.util, sys, torch
sys.path.insert(0, "problems/muse_bf16_proj"); sys.path.insert(0, "scratch")
import reference as R
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
W = R.generate_weights(1234); st = sub.prepare(W)
flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
def gt(fn, d, iters=40, cold=True):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2): fn(d)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn(d)
    e = torch.cuda.CUDAGraph()
    with torch.cuda.graph(e): pass
    ts = []
    for _ in range(iters):
        if cold: flush.zero_()
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); g.replay(); b.record(); torch.cuda.synchronize(); ts.append(a.elapsed_time(b) * 1e3)
    ts.sort(); return ts[len(ts)//2]
for spec_ in sys.argv[2:]:
    m, layer = spec_.split(":"); m = int(m)
    d = R.generate_input(m, layer, 7)
    t = gt(lambda d: sub.kernel_fn(d, st), d)
    tc = gt(lambda d: d["x"] @ W[layer][: d["config"]["n"]].T if layer != "router" else d["x"].float() @ W[layer].T, d)
    print(f"m={m:3d} {layer:9s} ours {t:7.1f}  cublas {tc:7.1f}")
