"""
GEMV Benchmark on GPU
Measures achieved HBM bandwidth for y = A @ x
where A is (4096 x 4096) FP16, x is (4096 x 1) FP16
"""

import sys
import torch

# ─── Config ───
M, K = 4096, 4096
DTYPE = torch.float32 if (len(sys.argv) > 1 and sys.argv[1] == "fp32") else torch.float16
BYTES_PER_ELEM = 4 if DTYPE == torch.float32 else 2
WARMUP = 50
RUNS = 200

# ─── Setup ───
device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(device)}")
print(f"Operation: y = A @ x, A shape ({M}x{K}), dtype {DTYPE}")
print(f"Warmup: {WARMUP}, Timed runs: {RUNS}\n")

A = torch.randn(M, K, dtype=DTYPE, device=device)
x = torch.randn(K, 1, dtype=DTYPE, device=device)

# ─── Total bytes moved ───
bytes_read = M * K * BYTES_PER_ELEM + K * BYTES_PER_ELEM   # matrix A + vector x
bytes_written = M * BYTES_PER_ELEM                          # vector y
total_bytes = bytes_read + bytes_written
total_flops = 2 * M * K              # multiply + accumulate per element

print(f"Matrix A size:  {M * K * BYTES_PER_ELEM / 1e6:.2f} MB")
print(f"Total bytes:    {total_bytes / 1e6:.2f} MB")
print(f"Total FLOPs:    {total_flops / 1e6:.2f} MFLOPs")
print(f"Arithmetic Intensity: {total_flops / total_bytes:.2f} ops/byte\n")

# ─── Warmup (let GPU clock stabilize, JIT compile, cache populate) ───
for _ in range(WARMUP):
    y = torch.mv(A, x.squeeze())

torch.cuda.synchronize()

# ─── Timed runs using CUDA events ───
times_ms = []

for _ in range(RUNS):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    y = torch.mv(A, x.squeeze())
    end.record()

    torch.cuda.synchronize()
    times_ms.append(start.elapsed_time(end))

# ─── Results ───
times_ms.sort()

# Drop top/bottom 10% outliers for stable median
trim = RUNS // 10
trimmed = times_ms[trim:-trim]

avg_ms = sum(trimmed) / len(trimmed)
min_ms = times_ms[0]
median_ms = times_ms[RUNS // 2]

print("─── Timing Results ───")
print(f"Min:    {min_ms:.4f} ms  ({min_ms * 1000:.1f} μs)")
print(f"Median: {median_ms:.4f} ms  ({median_ms * 1000:.1f} μs)")
print(f"Avg (trimmed 10%): {avg_ms:.4f} ms  ({avg_ms * 1000:.1f} μs)\n")

# ─── Bandwidth calculation ───
bw_min = total_bytes / (min_ms / 1000) / 1e9       # GB/s using fastest run
bw_median = total_bytes / (median_ms / 1000) / 1e9  # GB/s using median
bw_avg = total_bytes / (avg_ms / 1000) / 1e9        # GB/s using trimmed avg

print("─── Achieved Bandwidth ───")
print(f"From min time:    {bw_min:.1f} GB/s")
print(f"From median time: {bw_median:.1f} GB/s")
print(f"From avg time:    {bw_avg:.1f} GB/s")

# ─── P2200 peak BW for reference (change this for your GPU) ───
PEAK_BW_GBS = 200.0  # Quadro P2200 GDDR5X
print(f"\nPeak BW (spec):   {PEAK_BW_GBS:.1f} GB/s")
print(f"Efficiency (median): {bw_median / PEAK_BW_GBS * 100:.1f}%")
print(f"Efficiency (min):    {bw_min / PEAK_BW_GBS * 100:.1f}%")
