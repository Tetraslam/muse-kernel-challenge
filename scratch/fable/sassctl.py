import re, sys
txt = open("/tmp/opencode/k.sass").read()
want = sys.argv[1]
funcs = re.split(r"\n\s*Function : ", txt)
f = [x for x in funcs if x.startswith(want)][0]
lines = f.split("\n")
out = []
i = 0
while i < len(lines):
    m = re.match(r"\s+/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\* (0x[0-9a-f]+) \*/", lines[i])
    if m:
        hi = re.search(r"/\* (0x[0-9a-f]+) \*/", lines[i + 1])
        h = int(hi.group(1), 16)
        stall = (h >> 41) & 0xf; yld = (h >> 45) & 1; wb = (h >> 46) & 7; rb = (h >> 49) & 7; wait = (h >> 52) & 0x3f
        out.append((m.group(1), m.group(2), stall, wb, rb, wait))
        i += 2
    else:
        i += 1
lo, hi_ = int(sys.argv[2], 16), int(sys.argv[3], 16)
for addr, ins, stall, wb, rb, wait in out:
    a = int(addr, 16)
    if lo <= a <= hi_ and (("LDG" in ins) or wait or ("HMMA" in ins and wait) or "LDS" in ins or "BAR" in ins or "BRA" in ins or "HMMA" in ins):
        print(f"{addr} wb={wb if wb!=7 else '-'} rb={rb if rb!=7 else '-'} wait={wait:06b} stall={stall:2d} {ins[:70]}")
