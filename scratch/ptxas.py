import importlib.util, os, sys
os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
spec = importlib.util.spec_from_file_location("sub", sys.argv[1]); sub = importlib.util.module_from_spec(spec); spec.loader.exec_module(sub)
from torch.utils.cpp_extension import load_inline
load_inline(name="ptxas_probe_" + str(abs(hash(sub.CUDA_SRC)) % 10**8), cpp_sources=sub.CPP_SRC, cuda_sources=sub.CUDA_SRC, functions=["launch"],
            extra_cuda_cflags=["-O3", "--use_fast_math", "-Xptxas=-v"] + sys.argv[2:], verbose=True)
