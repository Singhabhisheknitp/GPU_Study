# Programming Massively Parallel Processors — Chapter Index

**Authors:** David B. Kirk (NVIDIA) & Wen-mei W. Hwu (UIUC)
**Source:** `Programming massive parallel processors.pdf` (279 pages)

## Chapters

| File | Chapter | Pages | Key Topics |
|------|---------|-------|------------|
| `Ch00_Front_Matter_and_Preface.pdf` | Front Matter & Preface | 1–21 | Copyright, TOC, Preface, Acknowledgments |
| `Ch01_Introduction.pdf` | Ch 1: Introduction | 22–41 | GPUs as parallel computers, modern GPU architecture, parallelism motivation, programming models, book organization |
| `Ch02_History_of_GPU_Computing.pdf` | Ch 2: History of GPU Computing | 42–59 | Graphics pipeline evolution, fixed-function → programmable, unified processors, GPGPU, scalable GPUs |
| `Ch03_Introduction_to_CUDA.pdf` | Ch 3: Introduction to CUDA | 60–79 | Data parallelism, CUDA program structure, matrix multiplication example, device memory & data transfer, kernel functions, threading |
| `Ch04_CUDA_Threads.pdf` | Ch 4: CUDA Threads | 80–97 | Thread organization, blockIdx/threadIdx, synchronization, transparent scalability, thread assignment, scheduling, latency tolerance |
| `Ch05_CUDA_Memories.pdf` | Ch 5: CUDA Memories | 98–115 | Memory access efficiency, device memory types (global/shared/constant/registers), tiling strategy, memory as parallelism limiter |
| `Ch06_Performance_Considerations.pdf` | Ch 6: Performance Considerations | 116–145 | Thread execution, **global memory bandwidth**, warps, coalescing, SM resource partitioning, data prefetching, instruction mix, thread granularity |
| `Ch07_Floating_Point_Considerations.pdf` | Ch 7: Floating Point | 146–161 | IEEE 754 format, normalized representation, excess encoding, representable numbers, precision, rounding, algorithm accuracy |
| `Ch08_Advanced_MRI_Reconstruction.pdf` | Ch 8: Case Study — MRI Reconstruction | 162–193 | FHd computation, kernel parallelism, memory bandwidth optimization, hardware trig functions, performance tuning |
| `Ch09_Molecular_Visualization_and_Analysis.pdf` | Ch 9: Case Study — Molecular Viz | 194–211 | Direct Coulomb summation, instruction efficiency, memory coalescing, multi-GPU usage |
| `Ch10_Parallel_Programming_and_Computational_Thinking.pdf` | Ch 10: Parallel Programming & Thinking | 212–225 | Problem decomposition, algorithm selection, computational thinking, parallel programming goals |
| `Ch11_Introduction_to_OpenCL.pdf` | Ch 11: Introduction to OpenCL | 226–241 | OpenCL data parallelism, device architecture, kernel functions, device management, comparison with CUDA |
| `Ch12_Conclusion_and_Future_Outlook.pdf` | Ch 12: Conclusion & Future | 242–253 | Memory architecture evolution, unified memory, configurable caching, atomic ops, kernel execution control, double-precision |
| `AppA_Matrix_Multiplication_Source_Code.pdf` | Appendix A: Source Code | 254–265 | matrixmul.cu, matrixmul_gold.cpp, matrixmul.h, assist.h |
| `AppB_GPU_Compute_Capabilities.pdf` | Appendix B: GPU Capabilities | 266–271 | Compute capability tables, memory coalescing variations |
| `Index.pdf` | Index | 272–279 | Alphabetical index |

## Topic Quick-Lookup

| If you're asking about... | Read this chapter |
|---------------------------|-------------------|
| What is CUDA / getting started | Ch 1, Ch 3 |
| GPU history and evolution | Ch 2 |
| Thread hierarchy (grids, blocks, warps) | Ch 4 |
| Memory types (shared, global, constant, registers) | Ch 5 |
| **Bandwidth, coalescing, performance optimization** | **Ch 5, Ch 6** |
| Floating point precision / accuracy | Ch 7 |
| Real-world CUDA optimization examples | Ch 8, Ch 9 |
| Parallel algorithm design patterns | Ch 10 |
| OpenCL vs CUDA | Ch 11 |
| Future GPU architecture directions | Ch 12 |
