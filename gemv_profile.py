"""
Minimal GEMV for nvprof profiling — fewer iterations to keep replay time short.
"""
import torch

M, K = 4096, 4096
device = torch.device("cuda")

A = torch.randn(M, K, dtype=torch.float16, device=device)
x = torch.randn(K, dtype=torch.float16, device=device)

# Warmup
for _ in range(10):
    y = torch.mv(A, x)
torch.cuda.synchronize()

# Just 5 timed iterations (nvprof replays each one per metric, so keep this small)
for _ in range(5):
    y = torch.mv(A, x)
torch.cuda.synchronize()
print("Done")
