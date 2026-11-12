import sys, time, torch
sys.path.insert(0, "/home/tetraslam/Programming/muse-kernel-challenge/problems/muse_bf16_proj")
sys.path.insert(0, "/home/tetraslam/Programming/muse-kernel-challenge/scratch/fable")
import reference as R
import importlib, os; sub = importlib.import_module(os.environ.get("SUB", "sub"))
t = time.time()
W = R.generate_weights(R.WEIGHTS_SEED)
st = sub.prepare({k: v.clone() for k, v in W.items()})
print("prepare", time.time() - t)
cases = [(m, l) for m in (1, 5, 7, 8, 9, 16, 32) for l in ("qkv_gate", "o", "pooler", "unpooler", "router")]
cases += [(9, "router", "outliers"), (33, "o"), (130, "unpooler")]
bad = 0
for i, c in enumerate(cases):
    m, l = c[0], c[1]
    d = R.generate_input(m, l, 100 + i, dist=c[2] if len(c) > 2 else "normal")
    out = sub.kernel_fn(d, st)
    torch.cuda.synchronize()
    ok, msg = R.check_implementation(d, out, W)
    bad += not ok
    print(m, l, ok, msg[:120])
print("BAD", bad)
