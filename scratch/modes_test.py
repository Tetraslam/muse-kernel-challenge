import importlib.util, sys, torch
sys.path.insert(0, "problems/muse_bf16_proj")
import reference as R
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
W = R.generate_weights(77); st = sub.prepare(W)
nsm = st["sms"]
bad = 0
for layer in ("qkv_gate", "o", "pooler", "unpooler", "router"):
    for m in (1, 5, 8, 13, 16, 32) + ((40, 100, 128, 250, 256) if layer == "router" else ()):
        d = R.generate_input(m, layer, m + 3, "outliers" if m % 2 else "normal")
        n, ku = d["config"]["n"], st["packs"][layer]["ku"]
        nrb = n // 128
        plans = {sub._plan(st, layer, m, n)}
        for s in (2, 3, 4, 7, 16, 33):
            if nrb * s * -(-m // 32) <= nsm and s <= ku:
                plans.add((s, 0)); plans.add((s, -1))
                if s <= 8: plans.add((s, s))
        for plan in sorted(p for p in plans if p):
            mp = 8 if m <= 8 else (16 if m <= 16 else 32)
            if -(-ku // plan[0]) * mp * 128 + mp * 512 > st["smem"]:
                continue
            g = torch.cuda.CUDAGraph()
            sub.kernel_fn(d, st, plan=plan)
            with torch.cuda.graph(g):
                out = sub.kernel_fn(d, st, plan=plan)
            ref = None
            for rep in range(20):
                g.replay()
                torch.cuda.synchronize()
                if ref is None:
                    ok, msg = R.check_implementation(d, out, W); ref = out.clone()
                    if not ok:
                        bad += 1; print("FAIL", layer, m, plan, msg)
                elif not torch.equal(out, ref):
                    bad += 1; print("NONDET", layer, m, plan); break
print("bad", bad)
