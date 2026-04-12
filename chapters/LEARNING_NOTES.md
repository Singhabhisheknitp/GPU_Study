# Programming Massively Parallel Processors — Learning Notes

**Learner:** PhD Year 1 (strong CPU architecture background)
**Book:** Kirk & Hwu, *Programming Massively Parallel Processors: A Hands-on Approach* (1st Ed.)
**Goal:** Master GPU parallel programming concepts for bandwidth/performance research

---

## Chapter 1: Introduction (Pages 22–41)

### The Big Picture — Why GPUs Exist for Computing

**The CPU wall (post-2003):** Clock frequency scaling hit a wall due to power/thermal limits. The industry split into two trajectories:

| Trajectory | Philosophy | Example | Design Priority |
|-----------|-----------|---------|-----------------|
| **Multicore (CPU)** | Keep sequential speed high, add a few fat cores | Intel Core i7 (4 cores) | Latency optimization |
| **Many-core (GPU)** | Maximize throughput with many thin cores | NVIDIA GTX 280 (240 cores) | Throughput optimization |

### The Fundamental CPU vs GPU Design Tradeoff

This is the single most important diagram in the chapter (Fig 1.2):

```
CPU:                              GPU:
┌──────────────────────┐          ┌──────────────────────┐
│  ┌──────┐ ┌───────┐  │          │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐ │
│  │Control│ │ Cache  │  │          │ │A││A││A││A││A││A│ │
│  │ Logic │ │(large) │  │          │ │L││L││L││L││L││L│ │
│  └──────┘ └───────┘  │          │ │U││U││U││U││U││U│ │
│  ┌────┐ ┌────┐       │          │ └─┘└─┘└─┘└─┘└─┘└─┘ │
│  │ALU │ │ALU │       │          │ ┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐ │
│  └────┘ └────┘       │          │ │A││A││A││A││A││A│ │
│         DRAM          │          │ │L││L││L││L││L││L│ │
└──────────────────────┘          │ │U││U││U││U││U││U│ │
                                   │ └─┘└─┘└─┘└─┘└─┘└─┘ │
  Transistors spent on:            │    Small Cache       │
  - Branch prediction              │         DRAM         │
  - Out-of-order execution         └──────────────────────┘
  - Large caches                   
  → Optimizes LATENCY              Transistors spent on:
                                   - Floating point ALUs
                                   → Optimizes THROUGHPUT
```

**Key insight for a CPU architect:** On CPUs, you spend transistors hiding latency (OoO, speculation, large caches). On GPUs, you spend transistors on compute and *tolerate* latency by switching to other threads. The GPU doesn't try to make one thread fast — it makes 10,000 threads collectively fast.

### Memory Bandwidth — The Critical Differentiator

| Metric | CPU (circa 2009) | GPU (G80 / GT200) |
|--------|------------------|--------------------|
| Peak FLOPS | ~100 GFLOPS | ~1 TFLOPS (10x) |
| Memory bandwidth | ~50 GB/s | 86–150 GB/s (2–3x) |
| PCIe host↔device | — | ~8 GB/s (4 up + 4 down) |

**Why GPUs have higher bandwidth:** GPUs use GDDR (graphics DRAM) with simpler memory models and fewer legacy constraints. CPUs must support coherence protocols, virtual memory for OS/IO, backward compatibility — all of which limit bandwidth scaling.

**Critical bottleneck:** The PCIe link (8 GB/s) between CPU and GPU is *much* slower than GPU memory bandwidth (150 GB/s). This means **data transfer between host and device is often the bottleneck** — a theme that will recur throughout the book.

### GPU Architecture at a Glance (Fig 1.3)

```
Host CPU ←──PCIe──→ GPU
                     ├── Thread Execution Manager
                     ├── SM 0: [SP SP SP SP SP SP SP SP] + Shared Memory
                     ├── SM 1: [SP SP SP SP SP SP SP SP] + Shared Memory
                     ├── ...
                     ├── SM 15: [SP SP SP SP SP SP SP SP] + Shared Memory
                     └── Global Memory (GDDR DRAM)
```

- **SM** = Streaming Multiprocessor (the "core" of a GPU)
- **SP** = Streaming Processor (individual ALU within an SM)
- G80: 16 SMs × 8 SPs = 128 SPs, 768 threads/SM → **~12,000 simultaneous threads**
- GT200: 30 SMs × 8 SPs = 240 SPs, 1024 threads/SM → **~30,000 simultaneous threads**

**CPU analogy:** An SM is like a CPU core (but simpler, in-order, with massive multithreading). An SP is like a SIMD lane.

### Amdahl's Law — The Sobering Reality

The chapter emphasizes a key point with concrete numbers:

| Parallel portion | Speedup of parallel part | Overall app speedup |
|-----------------|-------------------------|---------------------|
| 30% | 100x | **1.4x** (almost nothing!) |
| 99% | 100x | **50x** |
| 99.9% | 100x | **~100x** |

**Takeaway:** You need >99% of execution time to be parallelizable to see meaningful GPU speedups. The sequential portion dominates — this is why CUDA is a *heterogeneous* model (CPU handles sequential, GPU handles parallel).

### The CUDA Programming Model — Why It Won

Before CUDA (2007), GPU programming required abusing graphics APIs (OpenGL/Direct3D) — this was called GPGPU. CUDA changed everything:

- Direct C/C++ programming (no graphics API required)
- NVIDIA added **dedicated hardware** on-chip for general-purpose computing
- Heterogeneous model: CPU + GPU cooperate, each doing what they're good at

**Compared to other parallel models:**

| Model | Shared Memory? | Scales to? | Ease of use |
|-------|---------------|-----------|-------------|
| MPI | No (message passing) | 100,000+ nodes | Hard (explicit data movement) |
| OpenMP | Yes | ~hundreds of cores | Easy (compiler directives) |
| CUDA | Yes (within GPU) | Thousands of cores | Moderate (explicit but cleaner than MPI) |

### Key Takeaways for Your Research

1. **Bandwidth is the real bottleneck**, not compute — even in 2009, GPUs had more FLOPS than most apps could use. The challenge is feeding the ALUs fast enough.
2. **Memory optimization is everything** — "straightforward parallelization often saturates DRAM bandwidth, resulting in only ~10x speedup. The trick is to get around memory bandwidth limitations using specialized on-chip memories."
3. **The heterogeneous model matters** — know when to use CPU vs GPU; not everything benefits from GPU execution.

---

*Chapter 1 complete.*

---

## Chapter 2: History of GPU Computing (Pages 42–59)

### Why This Chapter Matters (in 1 sentence)

The GPU's architectural quirks — massive parallelism, tiny caches, bandwidth-centric memory — are not arbitrary; they are direct consequences of how graphics pipelines evolved. Understanding this history explains *why* GPUs are the way they are.

### The Four Eras of GPU Evolution

```
Era 1 (1980s–2001)          Era 2 (2001–2006)         Era 3 (2006)              Era 4 (2007+)
Fixed-Function Pipeline  →  Programmable Shaders  →  Unified Processors    →  CUDA / GPU Computing
                                                                               
- Hardwired stages          - Vertex shader prog.     - GeForce 8800 (G80)     - C/C++ programming
- Not programmable           - Pixel shader prog.      - Single processor array  - No graphics API needed
- Configurable only          - Still through gfx API   - Dynamic load balancing  - General memory access
- DirectX 1–7                - DirectX 9               - DirectX 10              - Scatter/gather writes
```

### Era 1: Fixed-Function Pipeline (Key Insight)

The graphics pipeline was a **literal hardware pipeline** with fixed stages:

```
CPU → Host Interface → Vertex Control → VS/T&L → Triangle Setup → Raster → Shader → ROP → Frame Buffer
```

Each stage did one thing in hardware. The critical design choice that persists today:

**Frame buffer memory was designed for bandwidth, not latency.** High-resolution displays needed to push millions of pixels per frame at 60 fps. GPU memory interfaces were engineered from day one to maximize GB/s, using:
- Special DRAM (not standard DDR)
- Multiple memory channels/banks
- Relaxed memory ordering (no coherence needed)

This is the origin of why GPU DRAM bandwidth >> CPU DRAM bandwidth.

### Era 2: Programmable Shaders (The Key Transition)

Why did graphics hardware become programmable?

**Data independence.** In a single frame: ~1 million triangles, ~6 million pixels — each pixel's color can be computed **independently**. This is textbook data parallelism. Game developers wanted custom shading algorithms, so hardware designers made the vertex and pixel shader stages programmable.

The programming model was restricted:
- Input: read from textures only
- Output: write only as pixel colors to predetermined framebuffer locations
- **No scatter writes** (can't write to arbitrary memory addresses)
- **No general memory access** — everything had to be cast as "graphics operations"

This was the GPGPU era — researchers forced general computations through graphics APIs. It worked, but was painful.

### Era 3: Unified Processors (GeForce 8800 / G80 — 2006)

The breakthrough: instead of separate vertex processors and pixel processors, use **one unified processor array** that handles all shader types.

```
Before (separate):                    After (unified, G80):
┌──────────────┐                      ┌──────────────────────┐
│ Vertex Procs │ ← underutilized      │                      │
├──────────────┤   when few vertices   │  Unified Processor   │
│ Pixel Procs  │ ← underutilized      │  Array (128 SPs)     │
└──────────────┘   when few pixels     │                      │
                                       │  Dynamically assigned│
                                       │  to vertex/pixel/    │
                                       │  geometry work       │
                                       └──────────────────────┘
```

**Why this matters architecturally:** Different scenes have wildly different vertex-to-pixel ratios. A fixed partition wastes silicon. Unification enables **dynamic load balancing** — the same hardware resource pool serves whatever work is needed most. This is the same principle behind modern GPU compute: the SM array dynamically distributes thread blocks.

### Era 4: CUDA (2007) — The Final Step

NVIDIA added to the unified G80 hardware:
- **Load/store instructions with byte addressing** (general memory access, not just textures)
- **Scatter writes** (write to arbitrary memory locations)
- Large instruction memory + instruction cache per SM
- Barrier synchronization and atomic operations
- C/C++ compiler (no more graphics API required)

The cost of adding instruction cache/sequencing was amortized by **sharing it across 8 SPs in each SM** — this works because all threads in a warp execute the same instruction.

### The Three Design Consequences (That Matter for Your Research)

| GPU trait | Historical origin | Implication |
|-----------|------------------|-------------|
| **Massive multithreading** | Graphics has millions of independent pixels per frame | Latency hiding by warp switching instead of OoO hardware |
| **Small caches** | Frame buffer memory was the bottleneck, not caches; graphics has high spatial locality | Must optimize memory access patterns (coalescing) — can't rely on cache to save you |
| **Bandwidth-centric memory** | Frame buffer needed GB/s for real-time display at 60 fps | GPU DRAM bandwidth >> CPU DRAM bandwidth, but latency is worse |

### Scalability — The GPU Programming Contract

The book makes an important point about **transparent scalability**: A CUDA program is written once and runs on any GPU regardless of core count. You expose *more parallelism than the hardware has resources for*, and the hardware maps as much as it can. When a bigger GPU arrives, the same program automatically uses the extra cores.

This is fundamentally different from CPU parallelism (OpenMP/pthreads), where you often tune for a specific core count and must restructure code when core count doubles.

### Key Takeaway

> GPU memory interfaces emphasize **bandwidth over latency** (as latency can be readily hidden by massively parallel execution); indeed, bandwidth is typically many times higher than that for a CPU, exceeding 100 GB/s.

This single design philosophy — **trade latency for bandwidth, compensate with threads** — is the thread that runs through the entire book.

---

*Chapter 2 complete.*

---

## Chapter 3: Introduction to CUDA (Pages 60–79)

### The Mental Model — Host + Device

CUDA views the world as two separate machines connected by a bus:

```
┌───────────────┐     PCIe Bus      ┌───────────────────────┐
│     HOST      │ ←──(~8 GB/s)───→  │       DEVICE          │
│    (CPU)      │                    │       (GPU)           │
│               │                    │                       │
│  Host Memory  │  cudaMemcpy()     │  Device Global Memory │
│  (System RAM) │ ───────────────→  │  (GDDR DRAM)          │
│               │ ←───────────────  │                       │
└───────────────┘                    └───────────────────────┘
```

**Separate address spaces.** A pointer on the host CANNOT be dereferenced on the device and vice versa. All data movement is explicit through API calls. This is the biggest conceptual shift from CPU programming — there's no shared virtual address space (in this era of CUDA).

### The Three-Step Pattern (Almost Every CUDA Program)

```
Step 1: Allocate device memory + copy data to device
        cudaMalloc()  →  cudaMemcpy(Host → Device)

Step 2: Launch kernel (parallel computation on device)
        kernel<<<grid, block>>>(args)

Step 3: Copy results back + free device memory
        cudaMemcpy(Device → Host)  →  cudaFree()
```

This is the "outsourcing agent" pattern: the host code ships data to the GPU, tells it to compute, and collects the result. The main program doesn't need to know the computation happened on a GPU.

### The Key APIs

| API Function | Purpose | CPU Equivalent |
|-------------|---------|---------------|
| `cudaMalloc(void** ptr, size)` | Allocate device global memory | `malloc()` |
| `cudaFree(ptr)` | Free device memory | `free()` |
| `cudaMemcpy(dst, src, size, direction)` | Transfer data between host ↔ device | `memcpy()` (but across address spaces) |

**Note:** `cudaMalloc` takes a *pointer to a pointer* (void**) — different from C's `malloc` which returns the pointer directly. This is so CUDA can use the return value for error reporting.

### From Sequential Loops to Parallel Threads

This is the core transformation that makes CUDA work. The CPU matrix multiply has three nested loops:

```
// CPU: Sequential — 3 nested loops
for (i = 0; i < Width; i++)          // loop over rows
  for (j = 0; j < Width; j++)        // loop over columns
    for (k = 0; k < Width; k++)      // dot product
      P[i*Width+j] += M[i*Width+k] * N[k*Width+j];
```

In CUDA, the **outer two loops (i, j) become the thread grid**. Each thread computes one output element:

```
// GPU: Kernel — only the innermost loop remains
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width) {
    int tx = threadIdx.x;  // replaces j
    int ty = threadIdx.y;  // replaces i
    float Pvalue = 0;
    for (int k = 0; k < Width; k++)   // only this loop survives
        Pvalue += Md[ty*Width+k] * Nd[k*Width+tx];
    Pd[ty*Width+tx] = Pvalue;
}
```

**The insight:** The hardware *generates* all (tx, ty) combinations simultaneously — you don't loop over them. For a 1000×1000 matrix, CUDA spawns 1,000,000 threads, each doing one dot product. The outer loops are replaced by parallelism.

### Thread Hierarchy — Grid → Blocks → Threads

```
Grid (launched by one kernel call)
├── Block (0,0)          Block (1,0)          Block (2,0)
│   ├── Thread (0,0)     ├── Thread (0,0)     ...
│   ├── Thread (1,0)     ├── Thread (1,0)
│   ├── Thread (0,1)     ├── Thread (0,1)
│   └── ...              └── ...
├── Block (0,1)          Block (1,1)
│   └── ...              └── ...
```

- **Grid:** 1D or 2D array of blocks (identified by `blockIdx.x`, `blockIdx.y`)
- **Block:** 1D, 2D, or 3D array of threads (identified by `threadIdx.x`, `.y`, `.z`)
- **Max 512 threads per block** (in this hardware generation)

**Why two levels?** Threads within a block can:
- **Synchronize** with each other (barrier)
- **Share data** through fast shared memory

Threads in *different blocks* **cannot cooperate** — they are independent. This is what enables transparent scalability: the hardware can execute blocks in any order, on any SM. 
**Q** each blocks maps to SMs but these blocks compuation can be totally asynchroonous? then how would they eventually be collected as GEMV if distributed across multiple SMs so finally vector output when projected needs to be collected across all the SMs. I am alittle confused? 


### Function Qualifiers

| Qualifier | Executes on | Called from |
|-----------|------------|-------------|
| `__global__` | Device (GPU) | Host (CPU) — this launches a grid |
| `__device__` | Device | Device only (from kernel or other device functions) |
| `__host__` | Host | Host only (default if no qualifier) |
| `__host__ __device__` | Both | Compiler generates two versions |

### The Limitation of This Simple Example

The kernel in this chapter uses only `threadIdx` (no `blockIdx`), which means it's limited to **a single block of 512 threads**. That means max matrix size is ~22×22. Clearly useless for real work. Chapter 4 fixes this by using multiple blocks and `blockIdx` to scale to millions of threads.

### Key Takeaways for Your Research

1. **Data transfer cost is explicit and visible.** Unlike CPU code where memory is "just there," CUDA forces you to think about every byte that crosses the PCIe bus. This is why minimizing host↔device transfers is a first-order optimization.

2. **Row-major layout matters.** The book emphasizes C's row-major storage: element `M[i][j]` lives at address `i*Width+j`. This will become critical in Chapter 5-6 when we discuss memory coalescing — adjacent threads must access adjacent memory addresses for efficient bandwidth use.

3. **SPMD ≠ SIMD.** CUDA is Single-Program, Multiple-Data — all threads run the same kernel code but on different data. This is NOT the same as SIMD (Single-Instruction, Multiple-Data) where all lanes execute the same instruction at the same instant. In CUDA, threads *can* diverge at branches (though at a performance cost, as we'll see in Ch 6).

---

*Chapter 3 complete.*

---

## Chapter 4: CUDA Threads (Pages 80–97)

### The Global Thread ID Formula

In Ch 3, we used only `threadIdx` (single block). Now we combine `blockIdx` and `threadIdx` to address millions of elements.

**1D case:**
```
globalThreadID = blockIdx.x * blockDim.x + threadIdx.x
```

Example: Grid of 128 blocks, each with 32 threads → 4096 threads total.
Thread 3 of Block 5 → ID = 5×32 + 3 = **163**

**2D case (matrix tiling):**
```
Row = blockIdx.y * TILE_WIDTH + threadIdx.y
Col = blockIdx.x * TILE_WIDTH + threadIdx.x
```

This is how you scale matrix multiply from 22×22 (single block) to arbitrarily large matrices.

### The Tiling Concept (Foundation for Ch 5)

The output matrix P is divided into **tiles**, one per block:

```
         Col →
         Block(0,0)  Block(1,0)  Block(2,0)
Row ↓    ┌────────┬────────┬────────┐
         │ 16×16  │ 16×16  │ 16×16  │  Block(0,0): threads compute P[0..15][0..15]
Block    │threads │threads │threads │  Block(1,0): threads compute P[0..15][16..31]
(0,y)    └────────┴────────┴────────┘  ...etc
         ┌────────┬────────┬────────┐
Block    │ 16×16  │ 16×16  │ 16×16  │
(1,y)    │threads │threads │threads │
         └────────┴────────┴────────┘
```

Each thread computes exactly ONE output element. The block/thread indices give it coordinates to find its row of M and column of N.

### Synchronization and Transparent Scalability — The Key Tradeoff

**`__syncthreads()`** = barrier within a block. All threads in the block must reach the barrier before any can proceed. This is critical for shared memory usage (Ch 5).

**The design constraint:** Threads in **different blocks CANNOT synchronize** with each other.

This seems limiting, but it's actually the **key enabler for scalability:**

```
Cheap GPU (2 SMs):              Expensive GPU (8 SMs):
                                
Block 0  Block 1                Block 0  Block 1  Block 2  Block 3
Block 2  Block 3  (sequential)  Block 4  Block 5  Block 6  Block 7
Block 4  Block 5                (all at once!)
Block 6  Block 7
↓ slower, same result           ↓ faster, same result
```

Because blocks are independent, the hardware can execute them in **any order, any number at a time**. Same code runs on a $200 GPU and a $5000 GPU — just faster on the expensive one. This is **transparent scalability**.

**CPU analogy:** This is like having no dependencies between loop iterations — the ultimate parallelizable loop. If blocks could sync with each other, the hardware would need to guarantee co-scheduling, destroying this flexibility.

### Thread Assignment to SMs — The Hardware Reality

```
Grid of blocks
    ↓ assigned by runtime
┌─────────────────────────────────┐
│  SM 0          SM 1          ...│
│  Block A       Block D          │
│  Block B       Block E          │
│  Block C       Block F          │
│                                 │
│  Max per SM:                    │
│  - 8 blocks (hard limit)       │
│  - 1024 threads (GT200)        │
│  - 768 threads (G80)           │
│  whichever is hit first        │
└─────────────────────────────────┘
```

Blocks are assigned to SMs as a unit. All threads in a block run on the **same SM** (this is required for `__syncthreads()` and shared memory to work). When an SM finishes a block, the runtime assigns a new one from the queue.

### Warps — The Actual Unit of Execution

Once a block lands on an SM, it's split into **warps of 32 threads:**

```
Block of 256 threads:
├── Warp 0:  threads 0–31
├── Warp 1:  threads 32–63
├── Warp 2:  threads 64–95
├── Warp 3:  threads 96–127
├── Warp 4:  threads 128–159
├── Warp 5:  threads 160–191
├── Warp 6:  threads 192–223
└── Warp 7:  threads 224–255
```

**All 32 threads in a warp execute the same instruction at the same time** (SIMT — Single Instruction, Multiple Thread). The warp is the unit of scheduling.

**CPU analogy:** A warp is like a 32-wide SIMD lane (think AVX-512 with 16 lanes, but doubled). The difference: if threads within a warp diverge at a branch, the GPU **serializes the paths** (both paths execute, with threads masked out). There's no branch predictor — divergence is handled by masking.
**Q** How exactly is execution handled by CPU SIMD then? it does not divergere unlike SIMT??


### Latency Hiding — Why You Need Many Warps

This is the GPU's fundamental trick (and the payoff from your Ch 1-2 understanding):

```
Time →
Warp 0: [COMPUTE] [LOAD ──── waiting 400 cycles ────→ data arrives] [COMPUTE]
Warp 1:            [COMPUTE] [LOAD ──── waiting ────]
Warp 2:                       [COMPUTE] [LOAD ──── waiting ────]
Warp 3:                                  [COMPUTE] ...
...
Warp 0:                                              [COMPUTE] ← data arrived, resume!

SM never idles if enough warps are resident!
```

**Zero-overhead scheduling:** Switching between warps costs ZERO cycles (unlike CPU context switches). Each thread's register state is permanently allocated — no save/restore needed. The scheduler just picks a different warp.

**This is why GPUs can tolerate 400-800 cycle memory latencies** without caches — as long as there are enough warps to keep the SPs busy during the wait.

### The Block Size Exercise — Practical Reasoning

For matrix multiply on GT200 (max 1024 threads/SM, max 8 blocks/SM):

| Block Size | Threads/Block | Blocks to fill SM | Actual threads/SM | Warps/SM | Verdict |
|-----------|--------------|-------------------|-------------------|----------|---------|
| 8×8 = 64 | 64 | need 16, but max is 8 | 8×64 = **512** | 16 | Bad — only 50% occupancy |
| 16×16 = 256 | 256 | 1024/256 = 4 (≤8 ✓) | 4×256 = **1024** | 32 | **Good — full occupancy** |
| 32×32 = 1024 | 1024 | Exceeds 512 thread/block limit! | — | — | **Invalid** |

**16×16 is optimal** — fills the SM to capacity with maximum warps for latency hiding.

### Key Takeaways

1. **`globalID = blockIdx * blockDim + threadIdx`** — the universal formula for mapping threads to data. You'll use this in every CUDA kernel.

2. **No inter-block synchronization = transparent scalability.** This is the fundamental contract: blocks are independent, hardware maps them freely.

3. **Warps of 32 threads** are the real unit of execution. Understanding warps is essential for performance (coalescing in Ch 5, divergence in Ch 6).

4. **Latency hiding via warp switching** is the GPU's alternative to CPU caches/OoO. It's zero-overhead but requires **enough resident warps** — which means choosing block sizes that maximize occupancy.

5. **Block size selection** must balance: thread count (maximize occupancy) vs. resource limits (registers, shared memory — coming in Ch 5).

---

*Chapter 4 complete.*

---

## Chapter 5: CUDA Memories (Pages 98–115) ★ Critical Chapter

### The Problem — Why Global Memory Alone Kills Performance

The matrix multiply kernel from Ch 4 does **2 global memory accesses** (one from M, one from N) per **2 floating-point ops** (one multiply, one add). This gives a **CGMA ratio of 1.0** (Compute to Global Memory Access ratio).

On the G80:
```
Global memory bandwidth = 86.4 GB/s
Each float = 4 bytes
→ Max data delivery = 86.4 / 4 = 21.6 billion floats/sec
→ With CGMA = 1.0 → only 21.6 GFLOPS achievable
→ Peak hardware = 367 GFLOPS
→ You're using only 5.9% of the GPU's compute power!
```

**The bottleneck is not compute — it's memory bandwidth.** This is the central message of the book and the most important concept for your research.

### The GPU Memory Hierarchy

```
Speed       Memory Type         Scope           Size (G80)        Declared with
─────       ───────────         ─────           ──────────        ─────────────
Fastest →   Registers           Per-thread      8K per SM         automatic scalars
            Shared Memory       Per-block       16 KB per SM      __shared__
            Constant Memory     All grids       64 KB (cached)    __constant__
Slowest →   Global Memory       All grids       Up to 4 GB        __device__ / cudaMalloc
            (Local Memory)      Per-thread      (in global mem)   automatic arrays
```

**CPU analogy mapping:**
| GPU Memory | CPU Equivalent | Key Difference |
|-----------|---------------|----------------|
| Registers | CPU registers | Same concept, but GPU has thousands per SM (shared among all threads) |
| Shared Memory | **L1 cache / scratchpad** | Explicitly managed by programmer (not automatic like CPU cache) |
| Constant Memory | Broadcast read from cache | Efficient only when all threads read the same address |  **Q** Where is this used for the constant memory examples??
| Global Memory | Main memory (DRAM) | High bandwidth but high latency (~400-800 cycles) |

**The critical difference from CPUs:** On a CPU, the cache hierarchy is **hardware-managed** (transparent). On a GPU, shared memory is **programmer-managed** (explicit). You decide what goes in, when it's loaded, and when it's evicted. More work, but more control.

### The Tiling Algorithm — The Key Technique

**Problem:** In matrix multiply, threads in the same block access overlapping rows of M and columns of N. Without tiling, each thread independently loads from global memory → massive redundant traffic.

**Solution:** Collaboratively load tiles into shared memory, then compute from shared memory.

```
Without tiling (CGMA = 1.0):          With 16×16 tiling (CGMA = 16.0):
                                       
Thread 0: load M[0][k] from GLOBAL     All threads: collaboratively load
Thread 1: load M[0][k] from GLOBAL     16×16 tile of M into SHARED MEMORY
  ↑ same data loaded twice!            (each thread loads 1 element)
                                       Then all threads read from SHARED
                                       → each global load serves 16 threads
```

**The math:**
- Without tiling: each element loaded from global memory is used by **1 thread**
- With N×N tiling: each element loaded is used by **N threads**
- Reduction in global memory traffic: **N× fewer accesses**
- With 16×16 tiles: 16× reduction → CGMA goes from 1.0 to **16.0**

```
New achievable performance:
(86.4 GB/s / 4 bytes) × 16 = 345.6 GFLOPS
→ Now close to peak 367 GFLOPS! 
```

### How Tiled Matrix Multiply Works (Step by Step)

```
Phase 1:                          Phase 2:
┌─────┬─────┐                     ┌─────┬─────┐
│Tile │     │ ← load this         │     │Tile │ ← load this
│of M │     │   into Mds          │     │of M │   into Mds
└─────┴─────┘                     └─────┴─────┘
┌─────┐                                 ┌─────┐
│Tile │ ← load this                     │Tile │ ← load this
│of N │   into Nds                      │of N │   into Nds
├─────┤                                 ├─────┤
│     │                                 │     │
└─────┘                                 └─────┘

1. All threads load one M element + one N element into shared memory
2. __syncthreads()  ← barrier: wait until ALL threads finished loading
3. Each thread computes partial dot product using shared memory
4. __syncthreads()  ← barrier: wait before overwriting shared memory
5. Move to next phase (next tile)
```

**Why two `__syncthreads()` calls?**
1. **After loading:** Ensure all elements are in shared memory before any thread tries to read them
2. **After computing:** Ensure all threads are done reading shared memory before the next phase overwrites it

Missing either barrier causes **race conditions** — threads would read uninitialized or stale data.

### Memory as a Limiter of Parallelism (Occupancy)

Even with tiling, you must balance resource usage. The SM has **finite** registers and shared memory, and using too much **reduces the number of blocks (and thus warps) per SM**.

**G80 resource constraints:**

| Resource | Per SM | Impact |
|----------|--------|--------|
| Registers | 8,192 | At 768 threads/SM → only 10 registers per thread. Use 11 → drop to 512 threads (1/3 reduction!) |
| Shared Memory | 16 KB | At 8 blocks/SM → max 2 KB per block. Use 5 KB → only 3 blocks fit |
| Threads | 768 max | Hard limit on total threads |
| Blocks | 8 max | Hard limit on blocks |

**For 16×16 tiled matrix multiply:**
- Shared memory per block: 2 × (16×16×4 bytes) = **2 KB** (Mds + Nds)
- 16 KB / 2 KB = 8 blocks → within limit ✓
- But thread limit (768) restricts to only 3 blocks of 256 → uses **6 KB of 16 KB**
- Fewer warps → less latency hiding

**This is the occupancy tradeoff:** using more shared memory per block can improve CGMA but reduce the number of resident warps, hurting latency hiding. Finding the sweet spot is a core GPU optimization skill.

### Key Takeaways for Your Research

1. **CGMA ratio is the single most important performance metric.** If your kernel has CGMA ≈ 1, you're bandwidth-bound regardless of how many FLOPS the GPU can do. Tiling is how you increase CGMA.

2. **Shared memory is an explicitly managed cache.** Unlike CPU L1/L2 which automatically caches recently accessed data, you must write code to load tiles into shared memory. This gives you precise control over data reuse.

3. **Global memory traffic reduction scales with tile size.** N×N tiles → N× reduction. But larger tiles require more shared memory per block → fewer blocks per SM → lower occupancy. There's always a tradeoff.

4. **`__syncthreads()` is the glue.** Tiling requires two barriers per phase — one after loading (ensure data is ready) and one after computing (ensure data isn't overwritten prematurely).

5. **Tiling is universal.** The same strategy applies to CPU caches (cache blocking/tiling in BLAS libraries), GPU shared memory, and any memory hierarchy. The principle: restructure computation to maximize reuse within fast local storage.

---

*Chapter 5 complete. Next: Chapter 6 (Performance Considerations)*
