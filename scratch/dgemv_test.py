import importlib.util, sys, torch
sys.path.insert(0, "problems/muse_bf16_proj")
import reference as R
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
W = R.generate_weights(21); st = sub.prepare(W)
bad = 0
for layer in ("qkv_gate", "o", "pooler", "unpooler"):
    for m in (1, 5, 8, 9, 16, 17, 32):
        d = R.generate_input(m, layer, m + 40, "outliers" if m % 2 else "normal")
        n = d["config"]["n"]; T = n // 16
        for F in (1, 2, 4, 8, 16):
            if T % F or T // F > 4 * st["sms"] or sub._module().dgemv_smem(m, F) > st["smem"]:
                continue
            out = sub.kernel_fn(d, st, df=F); torch.cuda.synchronize()
            ok, msg = R.check_implementation(d, out, W)
            if not ok: bad += 1; print("FAIL", layer, m, F, msg)
print("bad", bad)
