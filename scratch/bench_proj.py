"""Local check + graph-replay timing for muse_bf16_proj. Usage: bench_proj.py <submission.py> [--time]"""
import importlib.util
import sys

import torch

sys.path.insert(0, "problems/muse_bf16_proj")
import reference as R  # noqa: E402

spec = importlib.util.spec_from_file_location("sub", sys.argv[1])
sub = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sub)

TESTS = [(1, "qkv_gate", "normal"), (7, "pooler", "normal"), (33, "o", "normal"), (130, "unpooler", "normal"),
         (5, "unpooler", "normal"), (8, "router", "normal"), (9, "router", "outliers"),
         (512, "qkv_gate", "outliers"), (1000, "unpooler", "outliers")]
LAYERS = ["qkv_gate", "o", "pooler", "unpooler", "router"]
SCORED = [(m, l, "normal") for m in (1, 16, 32, 8, 128, 256) for l in LAYERS]

W = R.generate_weights(1234)
state = sub.prepare({k: v.clone() for k, v in W.items()})


def baseline(data):
    x, layer, n = data["x"], data["config"]["layer"], data["config"]["n"]
    if layer != "router":
        return x @ W[layer][:n].T
    if x.shape[0] <= 8:
        return torch.nn.functional.linear(x.float(), W["router"])
    return x.float() @ W["router"].T


def graph_time(fn, data, iters=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn(data)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn(data)
    flush = torch.empty(256 << 20, dtype=torch.uint8, device="cuda")
    ts = []
    for _ in range(iters):
        flush.zero_()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        g.replay()
        b.record()
        torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) * 1000)
    ts.sort()
    return out, ts[len(ts) // 2]


ok_all = True
for seed, (m, layer, dist) in enumerate(TESTS + SCORED):
    data = R.generate_input(m, layer, seed + 100, dist)
    out, _ = graph_time(lambda d: sub.kernel_fn(d, state), data, iters=1)
    # graph output was computed on the captured input: check it
    ok, msg = R.check_implementation(data, out, W)
    ok_all &= ok
    line = f"{'PASS' if ok else 'FAIL'} m={m:4d} {layer:9s} {dist:8s} {msg}"
    if "--time" in sys.argv and (m, layer, dist) in SCORED:
        _, t1 = graph_time(lambda d: sub.kernel_fn(d, state), data)
        _, t0 = graph_time(baseline, data)
        line += f" | ours {t1:7.1f}us  cublas {t0:7.1f}us  x{t0 / t1:.2f}"
    print(line, flush=True)
print("ALL PASS" if ok_all else "SOME FAILED")
