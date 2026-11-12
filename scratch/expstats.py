import sys, math, torch
sys.path.insert(0, "problems/muse_bf16_proj")
from reference import generate_weights
W = generate_weights(5151)
for name, w in W.items():
    if w.dtype == torch.float32:
        w = w.to(torch.bfloat16)
    b = w.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    e = (b >> 7) & 0xFF
    h = torch.bincount(e.flatten(), minlength=256).double()
    p = h[h > 0] / h.sum()
    ent = float(-(p * p.log2()).sum())
    emax = int(e.max())
    for win in (15, 16):
        exc = int((e < emax - win + 1).sum())
        print(name, f"ent={ent:.2f}b emax={emax} window{win} exceptions={exc} ({exc / e.numel():.2e})")
