import importlib.util, sys, torch
sys.path.insert(0, "problems/muse_bf16_proj")
import reference as R
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
W = R.generate_weights(91); st = sub.prepare(W)
bad = 0
for layer in ("qkv_gate", "o", "pooler", "unpooler"):
    for m in (1, 3, 8):
        d = R.generate_input(m, layer, 5 + m, "outliers" if m == 3 else "normal")
        n = d["config"]["n"]; T = n // 16
        p = st["packs"][layer]
        plans = set()
        for G in (st["sms"], 40, 32, T if T <= st["sms"] else 16):
            for F in (0, T // G):
                if G * F <= T:
                    for gd in (3, 4, 6):
                        if sub._module().hgemv_smem(gd, p["comp"], F, m, p["kx"]) <= st["smem"]:
                            plans.add((G, F, G * F, gd)); break
        for hp in sorted(plans):
            g = torch.cuda.CUDAGraph()
            sub.kernel_fn(d, st, hplan=hp)
            with torch.cuda.graph(g):
                out = sub.kernel_fn(d, st, hplan=hp)
            for rep in range(10):
                g.replay(); torch.cuda.synchronize()
                ok, msg = R.check_implementation(d, out, W)
                if not ok:
                    bad += 1; print("FAIL", layer, m, hp, rep, msg); break
print("bad", bad)
