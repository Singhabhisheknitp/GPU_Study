# GEMV Kernel Analysis on Quadro P2200

**Goal**: Figure out how much memory bandwidth a GEMV kernel actually utilizes on the P2200, step by step.

### Quadro P2200 Specs (relevant subset)
| Parameter | Value |
|---|---|
| Architecture | Pascal GP106 (compute 6.1) |
| SMs | 10 |
| CUDA cores | 1280 |
| Tensor cores | None |
| Memory | 5 GB GDDR5X |
| Peak BW | 200 GB/s (160-bit bus) |
| Peak FP32 | 3.8 TFLOPS |
| Peak FP16 (native) | ~59 GFLOPS (1/64 of FP32 on consumer Pascal) |
| Peak FP16 via FP32 pipe | 3.8 TFLOPS (what cuBLAS actually uses here) |
| **FP32 ridge point** | **19 ops/byte** (3.8 TFLOPS / 200 GB/s) |

---

## Plan

- [x] Step 1: Define the GEMV operation and pick concrete dimensions
- [x] Step 2: Count total bytes read/written from memory
- [x] Step 3: Count total FLOPs and confirm arithmetic intensity
- [x] Step 4: Compute theoretical minimum time (bytes / peak BW)
- [x] Step 5: Identify what prevents hitting peak BW in practice
- [x] Step 6: Estimate achieved bandwidth and why
- [x] Step 7: Experimental verification on P2200
- [x] Step 8: nvprof deep-dive — where did the time go?

---

## Step 1: Define the GEMV Operation

**Operation**: `y = A · x`

| Item | Shape | Dtype | Size |
|---|---|---|---|
| A (matrix) | 4096 × 4096 | FP16 (2 bytes) | 32 MB |
| x (input vector) | 4096 × 1 | FP16 (2 bytes) | 8 KB |
| y (output vector) | 4096 × 1 | FP16 (2 bytes) | 8 KB |

Each output element: `y[i] = Σ_k A[i][k] * x[k]` for k = 0..4095

---

## Step 2: Count Total Bytes Read/Written from Memory

| Item | Direction | Bytes |
|---|---|---|
| Matrix A | Read | 4096 × 4096 × 2 = 33,554,432 = **32 MB** |
| Vector x | Read | 4096 × 2 = 8,192 = **8 KB** |
| Vector y | Write | 4096 × 2 = 8,192 = **8 KB** |
| **Total** | | **~32.016 MB ≈ 32 MB** |

Vectors are rounding error. GEMV cost = cost of reading the matrix.

Note: x (8 KB) fits trivially in L2/L1 cache — loaded once, reused across all rows.
A (32 MB) does not fit in L2 — on any call, every byte comes from DRAM.

---

## Step 3: Count Total FLOPs and Arithmetic Intensity

Each output element y[i] requires:
- 4096 multiplications + 4096 additions = 8192 FLOPs? No.
- Convention: multiply-accumulate = **2 FLOPs** per element (1 mul + 1 add)
- Per row: 4096 × 2 = 8,192 FLOPs
- Total: 4096 rows × 8,192 = **33,554,432 FLOPs ≈ 33.5 MFLOPs**

**Arithmetic Intensity:**
- AI = FLOPs / Bytes = 33,554,432 / 33,570,816 ≈ **1.0 ops/byte**
- P2200 FP32 ridge point = **19 ops/byte**
- **We are 19× below the ridge → memory-bound**

The GPU could do 3.8 TFLOPS, but we only need 33.5 MFLOPs of compute.
The bottleneck is purely: how fast can we stream 32 MB out of GDDR5X?

**Ridge calculation:**
```
Ridge = Peak FLOPS / Peak BW
      = 3.8 × 10¹² FLOPs/s / 200 × 10⁹ B/s
      = 19 FLOPs/byte
```

---

## Step 4: Compute Theoretical Minimum Time

Since GEMV is entirely memory-bound, execution time ≈ time to read the data from DRAM.

```
T_min = Total bytes / Peak BW
T_min = 32 MB / 200 GB/s
T_min = 33,554,432 bytes / 200,000,000,000 bytes/s
T_min ≈ 0.000168 s
T_min ≈ **168 μs**
```

Cross-check with compute time:
```
T_compute = FLOPs / Peak FLOPS
T_compute = 33.5 MFLOPs / 3.8 TFLOPS
T_compute ≈ 8.8 μs
```

Compute time is ~19× smaller than memory time — consistent with AI being 19× below the ridge. Theoretical floor is **~168 μs**, entirely determined by GDDR5X read speed.

---

## Step 5: What Prevents Hitting Peak BW in Practice

Five factors that keep us from hitting the 168 μs theoretical floor:

### 5a. Kernel launch overhead (~3-5 μs)
- Fixed cost per kernel launch
- Small next to our 168 μs theoretical floor (~2-3%)
- Does NOT scale with matrix size — hurts small GEMVs disproportionately

### 5b. Not enough parallel memory requests in flight
- P2200 GDDR5X: 160-bit bus = 5 × 32-bit channels
- GPU needs many outstanding memory requests to keep all channels busy
- Only 10 SMs generating requests → the memory queue is shallow by construction
- This is the binding bottleneck on P2200 (quantified in Step 8)

### 5c. Memory access pattern (coalescing)
- Row-major A: each row is 8 KB contiguous → coalesced 128 B cache line reads → good
- Column-major A: accessing a row strides across memory → uncoalesced → wastes BW on partial cache lines
- A well-written kernel or cuBLAS handles this, but layout matters

### 5d. Partial SM occupancy (tail effect)
- 4096 rows distributed across 10 SMs
- Last wave of threadblocks may leave some SMs idle while others finish
- For our size: plenty of work per SM — tail waste is small (~3-4%)

### 5e. Reduction overhead
- Each row's dot product requires summing partial results across threads
- Uses warp shuffle instructions or shared memory — fast but not free
- Minor cost compared to the memory transfer time

### Net effect
A well-optimized GEMV on a larger GPU (100+ SMs) would hit ~70-85% of peak BW. On a 10-SM Pascal part the deep-queue assumption breaks down — we should expect meaningfully less. How much less is what Step 7 measures.

---

## Step 6: Estimate Achieved Bandwidth

### The calculation

We measure achieved bandwidth by: **BW_achieved = Total bytes moved / Actual kernel time**

Baseline prediction using the 70-85% rule-of-thumb for well-tuned kernels:

| Scenario | BW achieved | Kernel time | Efficiency |
|---|---|---|---|
| Theoretical peak | 200 GB/s | 168 μs | 100% |
| Excellent (cuBLAS, large matrix) | ~170 GB/s | ~197 μs | 85% |
| Good (well-tuned custom kernel) | ~156 GB/s | ~215 μs | 78% |
| Typical (reasonable kernel) | ~140 GB/s | ~240 μs | 70% |
| Naive (uncoalesced, poor occupancy) | ~50-80 GB/s | ~400-670 μs | 25-40% |

This table is for GPUs that can generate enough outstanding memory requests to saturate DRAM. Step 7 will show the P2200 cannot — so actual efficiency falls below this range.

### Key takeaway (pre-measurement)

For a 4096×4096 FP16 GEMV on P2200 we will read **32 MB** from GDDR5X and the time will be dominated by that transfer. Compute is ~99% idle. Whether we land at 70-85% efficiency or lower depends entirely on whether 10 SMs can keep the memory queue deep — measurement follows.

---

## Step 7: Experimental Verification

### Setup
- Benchmark script: `GPU_STUDY/gemv_benchmark.py`
- PyTorch with CUDA 11.3 (matches driver 470)

### How to run
```bash
# Step 1: Install PyTorch (one-time, ~1.8 GB, needs CUDA 11.3 for driver 470)
pip3 install torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# Step 2: Run benchmark
cd /data/frodo/abhishek/GPU_STUDY
python3 gemv_benchmark.py
```

### Benchmark Results (2026-04-12)

```
GPU: Quadro P2200
Operation: y = A @ x, A shape (4096x4096), dtype torch.float16
Warmup: 50, Timed runs: 200

Matrix A size:  33.55 MB
Total bytes:    33.57 MB
Total FLOPs:    33.55 MFLOPs
Arithmetic Intensity: 1.00 ops/byte

─── Timing Results ───
Min:    0.3133 ms  (313.3 μs)
Median: 0.3164 ms  (316.4 μs)
Avg (trimmed 10%): 0.3164 ms  (316.4 μs)

─── Achieved Bandwidth ───
From min time:    107.1 GB/s
From median time: 106.1 GB/s
From avg time:    106.1 GB/s

Peak BW (spec):   200.0 GB/s
Efficiency (median): 53.0%
Efficiency (min):    53.6%
```

### Predicted vs Actual

| Metric | Theoretical floor | Predicted (70-85% rule) | Actual measured |
|---|---|---|---|
| Time | 168 μs | 197-240 μs | **316 μs** |
| Bandwidth | 200 GB/s | 140-170 GB/s | **106 GB/s** |
| Efficiency | 100% | 70-85% | **53%** |

We hit 53% — significantly below the 70-85% prediction. The rule-of-thumb was calibrated for GPUs with 100+ SMs. The P2200 with only 10 SMs has fundamentally different bottlenecks. Step 8 explains.

---

## Step 8: nvprof Deep-Dive — Where Did the Time Go?

Profiled using `nvprof --metrics` with CUDA toolkit 11.4.

**Kernel profiled**: `gemv2T_kernel_val` (cuBLAS FP16 GEMV, 128 threads/block, 8 rows/block, FP32 accumulator)

### 8a. How torch.mv() Becomes a CUDA Kernel

When you write `y = torch.mv(A, x)` in Python, here's the full call chain:

```
torch.mv(A, x)                          # Python
  → PyTorch dispatcher                   # sees: FP16, 4096×4096, contiguous
    → cuBLAS cublasGemvEx()              # PyTorch delegates to NVIDIA's library
      → cuBLAS heuristic selects:        # cuBLAS picks best kernel for this GPU + dims
          kernel:  gemv2T_kernel_val
          grid:    (512, 1, 1)           # 512 thread blocks total
          block:   (128, 1, 1)           # 128 threads per block
      → launches <<<512, 128>>>(...)     # actual CUDA kernel launch
```

**Who decides grid and block size?** cuBLAS does — not the programmer. cuBLAS has a library of pre-written kernels with different tile sizes, unroll factors, etc., and an internal heuristic selects the best one based on matrix dimensions, dtype, and GPU architecture. If you wrote raw CUDA, **you** would specify `<<<grid, block>>>` yourself. But through PyTorch → cuBLAS, those choices are made for you.

### 8b. Kernel Template Signature — What Each Parameter Means

From `nvprof --print-gpu-trace`:

```
Grid: (512, 1, 1)   Block: (128, 1, 1)   Regs: 44   Shared: 1.5KB
Kernel: void gemv2T_kernel_val<int, int, __half, __half, __half, float,
        int=128, int=16, int=2, int=4, bool=0, bool=0,
        cublasGemvParams<...>>
```

Breaking down the template parameters:

| Template Parameter | Value | Meaning |
|---|---|---|
| `int, int` | IndexType | How matrix indices are computed |
| `__half` | T_input_A | Matrix A element type — FP16 |
| `__half` | T_input_x | Vector x element type — FP16 |
| `__half` | T_output | Output vector y element type — FP16 |
| `float` | T_compute | **Accumulator type — FP32** (this causes the 21% exec stall) |
| `int=128` | THREADS_PER_BLOCK | 128 threads = 4 warps per block |
| `int=16` | TILE_SIZE | Tile dimension used internally by the kernel |
| `int=2` | UNROLL_FACTOR | Inner loop unrolled 2× to reduce branch overhead |
| `int=4` | ELEMENTS_PER_THREAD_PER_LOAD | Each thread loads 4 halfs (8 bytes) per memory transaction |
| `bool=0` | USE_BETA_ZERO | beta=0, skip reading old y (just y=A·x, not y=αA·x+βy) |
| `bool=0` | CONJUGATE | Not doing conjugate transpose |
| `cublasGemvParams<...>` | Parameter struct | Wraps tensor descriptors for A (const half), x (const half), y (half) |

**Actual launch configuration** (from `--print-gpu-trace`):
- **Grid (512, 1, 1)**: 512 thread blocks — each handles 4096/512 = **8 rows** of A
- **Block (128, 1, 1)**: 128 threads per block → 128 threads / 8 rows = **16 threads per row**
- **Registers**: 44 per thread
- **Shared memory**: 1.5 KB per block (for partial sum reduction)
- **Total threads**: 512 × 128 = 65,536
- **Blocks per SM**: 512 / 10 SMs ≈ 51 blocks per SM (scheduled in waves)

### 8c. P2200 Per-SM Resource Limits (Pascal, Compute Capability 6.1)

These are the hardware caps on what a single SM can hold simultaneously:

| Resource | Limit per SM |
|---|---|
| **Registers (32-bit)** | 65,536 (64K) |
| **Max registers per thread** | 255 |
| **Shared memory** | 96 KB per SM |
| **Max shared memory per block** | 48 KB |
| **Max threads per SM** | 2,048 |
| **Max warps per SM** | 64 (32 threads × 64 = 2048) |
| **Max thread blocks per SM** | 32 |
| **Max threads per block** | 1,024 |
| **Warp size** | 32 |

### 8d. Occupancy Calculation for This Kernel

Kernel uses: **128 threads/block, 44 registers/thread, 1.5 KB shared memory**

Apply each SM resource limit to find the max co-resident blocks:

| Limiter | Calculation | Max blocks/SM |
|---|---|---|
| Thread limit | 2048 / 128 | 16 blocks |
| Warp limit | 64 warps / 4 warps per block | 16 blocks |
| **Register limit** | **65536 / (128 × 44) = 65536/5632** | **11 blocks ← binding** |
| Shared mem limit | 96 KB / 1.5 KB | 64 blocks |
| Hard block limit | — | 32 blocks |

**Registers are the binding constraint.** 11 blocks × 128 threads = 1408 threads/SM = **44 warps/SM active** out of max 64 → theoretical **68.75% occupancy**.

Measured occupancy was **59.7%** — the small gap vs theoretical 68.75% comes from register allocation granularity (allocated in chunks of 256) and scheduling, typically reducing effective blocks from 11 → ~9-10 per SM.

**Takeaway:** The FP32 accumulator (from `float` template param) drives register pressure up to 44/thread, which caps occupancy below the thread/warp limit. This is the root of the low-occupancy → shallow-memory-queue bottleneck.

### 8e. Raw Metrics from nvprof

| Metric | Min | Max | Avg | Interpretation |
|---|---|---|---|---|
| **dram_read_throughput** | 100.9 GB/s | 102.5 GB/s | **102.1 GB/s** | Ground truth BW from DRAM controller |
| **dram_write_throughput** | 24.6 MB/s | 1.94 GB/s | **291 MB/s** | Tiny — just writing output vector y |
| **dram_read_transactions** | 1,049,952 | 1,050,736 | **1,050,081** | 32B per txn → 33.6 MB read |
| **dram_write_transactions** | 247 | 20,110 | **2,920** | Negligible |
| **achieved_occupancy** | 0.590 | 0.605 | **0.597 (59.7%)** | Only 60% of max warps active |
| **sm_efficiency** | 93.9% | 97.0% | **95.75%** | All 10 SMs busy — no idle SMs |
| **warp_execution_efficiency** | 98.64% | 98.64% | **98.64%** | No branch divergence |
| **stall_memory_dependency** | 44.0% | 45.0% | **44.4%** | #1 stall: waiting for DRAM |
| **stall_exec_dependency** | 21.2% | 21.7% | **21.4%** | #2 stall: FP16→FP32 pipeline |
| **stall_not_selected** | 2.73% | 2.81% | **2.8%** | Scheduler had another warp to pick |

### 8f. Coalescing Efficiency — PERFECT

```
Actual DRAM read transactions:    1,050,081
Expected (32 MB / 32B per txn):   1,048,576
Overhead:                          1,505 extra txns = 0.14%
```

cuBLAS reads matrix A row-by-row in contiguous 128 B cache lines. Essentially zero wasted transactions. **Coalescing is NOT our problem.**

### 8g. Stall Analysis — THE SMOKING GUN

Where warps spend their time when they can't issue the next instruction:

```
stall_memory_dependency:    44.4%   ← waiting for data from DRAM
stall_exec_dependency:      21.4%   ← waiting for previous instruction to finish
stall_not_selected:          2.8%   ← warp ready but scheduler picked another
(remaining ~31%:             other / not stalled — actually executing)
```

**44% memory stall**: Nearly half the time, warps sit idle waiting for DRAM reads to return. Root cause:
- Only 10 SMs generating memory requests
- Only 60% occupancy → ~38 active warps per SM (max 64 on Pascal)
- Fewer warps in flight = shallower memory request queue = GDDR5X controller not fully saturated
- GDDR5X latency (~100-150 ns per request) needs deep queues to hide; 10 SMs can't provide that

**21% exec dependency**: The FP16-on-Pascal penalty. Pascal has no native FP16 datapath. The kernel:
1. Loads FP16 data from memory
2. Converts to FP32 (kernel template shows `float` as accumulator type)
3. Does FP32 FMA (fused multiply-add)
4. Each FMA depends on the previous accumulation result → pipeline bubble

This 21% is pure compute pipeline stall, not memory. Would vanish on any architecture with a native FP16 datapath.

**2.8% not selected**: Negligible. With low occupancy, the scheduler rarely has multiple ready warps competing.

### 8h. Bottleneck Summary

| Factor | Contribution | Fixable? |
|---|---|---|
| Low occupancy (59.7%) → shallow memory queue | **#1 bottleneck** | No — hardware limit (10 SMs, register pressure) |
| Memory latency stalls (44%) | Direct consequence of #1 | Needs more SMs / lower-latency memory |
| FP16→FP32 exec dependency (21%) | **#2 bottleneck** | Needs a GPU with native FP16 datapath |
| Coalescing waste | ~0% | Already perfect |
| SM idle / tail effect | ~4% | Already fine |

### Key takeaway

53% efficiency on a 10-SM Pascal GPU is **reasonable and explained by the hardware**. The cuBLAS kernel is well-written (perfect coalescing, 96% SM utilization, no divergence). The bottleneck is that 10 SMs simply cannot generate enough outstanding memory requests to keep the GDDR5X controller fully saturated, compounded by a 21% FP16→FP32 conversion tax inherent to consumer Pascal.

---

## Step 9: FP32 Comparison — Isolating the FP16→FP32 Conversion Tax

To verify Step 8's claim that the 21% exec-dependency stall is caused by the FP16→FP32 conversion on Pascal, we re-ran the same GEMV in FP32 (matrix A, x, y all `float`). FP32 doubles the bytes moved (64 MB vs 32 MB) and halves the arithmetic intensity (0.5 vs 1.0 ops/byte) but uses Pascal's native FP32 datapath end-to-end.

### Benchmark Results (2026-04-13)

```
GPU: Quadro P2200
Operation: y = A @ x, A shape (4096x4096), dtype torch.float32
Matrix A size:  67.11 MB      Total bytes: 67.14 MB
Total FLOPs:    33.55 MFLOPs   Arithmetic Intensity: 0.50 ops/byte

Min:    480.7 μs    Median: 486.0 μs
Achieved BW (median): 138.2 GB/s   Efficiency: 69.1%
```

### Side-by-Side Metrics (nvprof)

| Metric | FP16 | FP32 | Change | Interpretation |
|---|---|---|---|---|
| Kernel template dtypes | `__half,__half,__half,float` | `float,float,float,float` | native FP32 path | no conversion step |
| Unroll factor | int=4 | int=2 | less unrolling | cuBLAS picks different tile |
| **stall_exec_dependency** | **21.42%** | **9.67%** | **−11.75 pp** ✓ | FP16→FP32 conversion stall ~halved |
| **achieved_occupancy** | 59.7% | **72.0%** | +12.3 pp | FP32 kernel uses fewer regs/thread |
| **dram_read_throughput** | 102.1 GB/s | **131.4 GB/s** | +29% | deeper memory queue |
| **stall_memory_dependency** | 44.30% | **69.33%** | +25 pp | now purely DRAM-bound |
| DRAM read transactions | 1,050,008 (32 MB) | 2,099,754 (64 MB) | 2× | as expected |
| sm_efficiency | 95.62% | 96.27% | ~same | all SMs busy |
| warp_execution_efficiency | 98.64% | 98.84% | ~same | no divergence |
| stall_not_selected | 2.79% | 1.70% | small ↓ | low scheduler pressure |
| Kernel time | 316 μs | 486 μs | +54% (not +100%) | bytes doubled but throughput improved |
| **Achieved BW** | 106 GB/s | **138 GB/s** | +30% | |
| **Efficiency (% peak)** | **53.0%** | **69.1%** | **+16.1 pp** | |

### What this proves

1. **The 21% exec-dependency stall was half conversion, half FMA chain.**
   Dropping FP16 eliminated the FP16→FP32 conversion path but left a ~10% stall from the FMA accumulation dependency (each multiply-add depends on the previous partial sum). That ~10% floor exists in *any* dot-product kernel on any architecture — it's the latency of a single FMA pipeline step.

2. **Occupancy went up (59.7% → 72.0%).**
   The FP32 kernel uses fewer registers per thread — no FP16 load → FP32 convert → FP32 FMA temporary chain — and a lower unroll factor (2 vs 4). Lower register pressure relaxes the Step 8d binding constraint, so more warps fit per SM.

3. **Memory stall went *up* (44% → 69%) — which is good.**
   Higher memory-stall fraction means the kernel spends more of its time waiting for DRAM vs spinning on the compute pipeline. This is exactly the profile of a well-behaved memory-bound kernel. With conversion cost removed, DRAM is now the only thing holding us back.

4. **Time did not double despite 2× bytes.**
   Moving from 32 MB to 64 MB only increased kernel time from 316 μs → 486 μs (1.54×, not 2×). The extra occupancy and removed conversion stall let the kernel issue memory requests more densely — a clean demonstration that wall time = bytes / *effective* BW, not bytes / peak BW.

5. **Bandwidth efficiency jumped 53% → 69%.**
   The remaining 31% gap to peak is still the fundamental 10-SM-count bottleneck (insufficient outstanding memory requests to saturate GDDR5X). That ceiling cannot be moved without more SMs or lower-latency memory.

### Updated bottleneck attribution for FP16

| Bottleneck | Cost | Confirmed by |
|---|---|---|
| 10-SM memory queue depth | ~31 pp below peak | remaining gap in FP32 run |
| FP16→FP32 conversion | ~12 pp (of the 47 pp gap in FP16) | 21.42% → 9.67% exec-stall drop |
| FMA accumulation chain | ~10 pp (residual, unavoidable) | persists in FP32 |

### Key takeaway

The FP16 53% → FP32 69% jump is **not** because FP32 is faster per byte — it's because FP32 removes a compute-pipeline stall that was starving the memory subsystem. On Pascal, FP16 is actively harmful for memory-bound kernels: you pay a stall tax to use a format the hardware doesn't natively support. On an architecture with a native FP16 datapath (Volta+, Turing, Ampere, Hopper), this penalty vanishes and FP16 would win by 2× on bandwidth-bound work.
