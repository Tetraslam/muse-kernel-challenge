import importlib.util, sys, torch
sys.path.insert(0, "problems/muse_bf16_proj")
import reference as R
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
W = R.generate_weights(13); st = sub.prepare(W)
bad = 0
for m in (1, 2, 7, 8, 9, 16, 17, 31, 32, 33, 64, 100, 128):
    d = R.generate_input(m, "router", m, "outliers" if m % 3 == 0 else "normal")
    chunks = -(-m // 32)
    for s in sorted({x for x in (sub._rplan(st, m, 256), 1, 2, 3, 8) if x}):
        if s is None or 16 * s * chunks > st["sms"] or sub._module().router_smem(m, 256, s) > st["smem"]:
            continue
        g = torch.cuda.CUDAGraph()
        sub.kernel_fn(d, st, rs=s)
        with torch.cuda.graph(g):
            out = sub.kernel_fn(d, st, rs=s)
        ref = None
        for rep in range(10):
            g.replay(); torch.cuda.synchronize()
            ok, msg = R.check_implementation(d, out, W)
            if not ok: bad += 1; print("FAIL", m, s, msg); break
            if ref is None: ref = out.clone()
            elif not torch.equal(ref, out): bad += 1; print("NONDET", m, s); break
print("bad", bad)
