import csv, io, sys, collections
text = open(sys.argv[1]).read().split("==SOURCE==")[1]
lines = [l for l in text.splitlines() if l.startswith('"') and not l.startswith('"Kernel Name')]
rows = list(csv.DictReader(io.StringIO("\n".join(lines))))
stall_cols = [c for c in rows[0] if c.startswith("stall_") and "Not Issued" not in c]
tot = collections.Counter()
for r in rows:
    for c in stall_cols:
        try: tot[c] += float(r[c] or 0)
        except ValueError: pass
S = sum(tot.values())
print("total samples", S)
for c, v in tot.most_common(10): print(f"  {c:28s} {100*v/S:5.1f}%")
def samp(r):
    try: return float(r["Warp Stall Sampling (All Samples)"] or 0)
    except ValueError: return 0
top = sorted(rows, key=samp, reverse=True)[:int(sys.argv[2]) if len(sys.argv) > 2 else 25]
for r in top:
    top_st = max(stall_cols, key=lambda c: float(r[c] or 0) if r[c] not in ("", None) else 0)
    print(f"{100*samp(r)/S:5.1f}%  {r['Address']:>6s}  {r['Source'][:70]:70s} {top_st}")
