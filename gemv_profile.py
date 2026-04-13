"""
Minimal GEMV for nvprof profiling — fewer iterations to keep replay time short.
"""
import sys
import torch

M, K = 4096, 4096
device = torch.device("cuda")
DTYPE = torch.float32 if (len(sys.argv) > 1 and sys.argv[1] == "fp32") else torch.float16

A = torch.randn(M, K, dtype=DTYPE, device=device)
x = torch.randn(K, dtype=DTYPE, device=device)

# Warmup
for _ in range(10):
    y = torch.mv(A, x)
torch.cuda.synchronize()

# Just 5 timed iterations (nvprof replays each one per metric, so keep this small)
for _ in range(5):
    y = torch.mv(A, x)
torch.cuda.synchronize()
print("Done")
