import importlib.util, sys, torch
sys.path.insert(0, "problems/muse_bf16_proj")
import reference as R
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
m, layer = int(sys.argv[2]), sys.argv[3]
W = R.generate_weights(1234); st = sub.prepare(W)
d = R.generate_input(m, layer, 7)
for _ in range(3): out = sub.kernel_fn(d, st)
torch.cuda.synchronize()
print(R.check_implementation(d, out, W))
