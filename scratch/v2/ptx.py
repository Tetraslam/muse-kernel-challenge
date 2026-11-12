import sys, re, subprocess, tempfile, os
sys.path.insert(0, ".")
src = open(sys.argv[1]).read()
ns = {}
exec(src.split("_ext = None")[0], ns)
d = tempfile.mkdtemp()
open(f"{d}/k.cu", "w").write(ns["CUDA_SRC"])
import torch.utils.cpp_extension as ce
inc = " ".join(f"-I{p}" for p in ce.include_paths(device_type="cuda"))
import sysconfig
cmd = f"nvcc -c -O3 -std=c++20 -gencode=arch=compute_90a,code=sm_90a -Xptxas=-v {inc} -I{sysconfig.get_paths()['include']} {d}/k.cu -o {d}/k.o && cuobjdump -sass {d}/k.o > /tmp/opencode/k.sass"
r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
lines = r.stderr.splitlines()
fn = None
for l in lines:
    m = re.search(r"Compiling entry function '(\S+)'", l)
    if m: fn = subprocess.run(["c++filt", m.group(1)], capture_output=True, text=True).stdout.strip()
    if "registers" in l or "spill" in l and fn:
        if "spill" in l and "0 bytes spill" in l: continue
        print(fn[:90], "|", l.strip()[:120])
if r.returncode: print(r.stderr[-3000:])
