import modal

image = modal.Image.from_registry("lmsysorg/sglang:dev-muse-spark-open").entrypoint([])
app = modal.App("okc-ncu-probe", image=image)


@app.function(gpu="H200", timeout=900)
def probe():
    import subprocess
    out = []
    for cmd in ["nvidia-smi -L", "which ncu nsys; ls /usr/local/cuda/bin | grep -E 'ncu|nsys'; ls /opt/nvidia 2>/dev/null",
                "python -c 'import torch,sglang;print(torch.__version__, sglang.__version__)'"]:
        out.append(subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout)
    src = "import torch; a=torch.randn(4096,4096,device='cuda'); b=a@a; torch.cuda.synchronize()"
    open("/tmp/t.py", "w").write(src)
    r = subprocess.run("ncu --clock-control none --section SpeedOfLight python /tmp/t.py 2>&1 | tail -15", shell=True,
                       capture_output=True, text=True)
    out.append(r.stdout + r.stderr)
    return "\n".join(out)


@app.local_entrypoint()
def main():
    print(probe.remote())
