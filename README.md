# Optimizing a Custom CUDA GEMM Kernel Toward cuBLAS-Level Performance

An incremental, ablation-style high-performance computing project dedicated to implementing, analyzing, and progressively optimizing a custom CUDA General Matrix Multiplication (GEMM) kernel on NVIDIA GPUs, aiming to bridge the performance gap between a naive implementation and the industry-standard NVIDIA cuBLAS library.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Implemented Milestones & Functionality](#implemented-milestones--functionality)
  - [Milestone 1: Problem Definition, Literature Review & CPU Baseline](#milestone-1-problem-definition-literature-review--cpu-baseline)
  - [Milestone 2: Naive CUDA Parallelization & Scaling Baseline](#milestone-2-naive-cuda-parallelization--scaling-baseline)
- [File & Module Details](#file--module-details)
- [Empirical Results](#empirical-results)
- [Build and Execution Guide](#build-and-execution-guide)
  - [Compilation](#compilation)
  - [Execution Modes](#execution-modes)
  - [Running on Google Colab](#running-on-google-colab)
- [Optimization Roadmap](#optimization-roadmap)

---

## Project Overview

General Matrix Multiplication (GEMM) is the fundamental computational kernel underlying modern deep learning models, scientific computing, and high-performance linear algebra. The single-precision BLAS Level-3 GEMM operation is defined as:

$$C \leftarrow \alpha AB + \beta C$$

where:
- $A \in \mathbb{R}^{M \times K}$
- $B \in \mathbb{R}^{K \times N}$
- $C \in \mathbb{R}^{M \times N}$
- $\alpha, \beta \in \mathbb{R}$ are scalar coefficients.

For square matrices of dimension $N$ ($M = N = K$), GEMM requires $2N^3$ floating-point operations while operating on $O(N^2)$ matrix elements. While GEMM has a theoretically high operational intensity, naive implementations fail to achieve hardware peak throughput because performance is governed by memory access patterns and data-supply bottlenecks across the memory hierarchy (registers, shared memory, L1/L2 caches, and global DRAM).

### Central Research Question
> *How can a custom CUDA GEMM kernel be systematically optimized through memory hierarchy utilization, tiling, register blocking, and GPU-specific execution techniques to approach the performance of the highly optimized NVIDIA cuBLAS GEMM implementation?*

Rather than treating GPU optimization as a black box, this repository implements a controlled, stepwise experimental pipeline where each optimization stage is isolated, measured, and verified for numerical correctness against trusted CPU references.

---

## Repository Structure

```text
Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/
│
├── README.md                           # Comprehensive project documentation
│
├── Milestone 1/                        # Milestone 1: Foundations & Sequential Baseline
│   ├── Problem Definition.pdf          # Problem statement, research question & objectives
│   ├── Lit Review.pdf                  # Literature review on GPU GEMM bottlenecks & roadmap
│   └── Sequential Implementation.pdf   # CPU baseline benchmarks, cache locality & correctness
│
└── Milestone 2/                        # Milestone 2: Initial CUDA Parallelization
    ├── gemm_cuda_naive.cu              # Complete CUDA naive kernel, CPU references, validation & benchmark harness
    ├── Milestone_2.ipynb               # Google Colab notebook for GPU compilation and execution
    ├── results_milestone2.csv          # Benchmark output data (CPU vs. CUDA execution time and GFLOP/s)
    ├── Milestone_2.pdf                 # Milestone 2 technical report (draft)
    ├── Milestone_2 FINAL PC.pdf        # Milestone 2 final IEEE-style conference report
    └── PPT_PC.pdf                      # Comprehensive 35-slide presentation deck covering Milestones 1 & 2
```

---

## Implemented Milestones & Functionality

### Milestone 1: Problem Definition, Literature Review & CPU Baseline
- **Conceptual & Architectural Foundations**: Analysis of GPU execution hierarchy (Grids, Thread Blocks, Warps, Threads, Streaming Multiprocessors), the memory hierarchy (Global DRAM, L2 cache, shared memory, registers), and the Roofline Model ($P_{\text{achievable}} = \min(P_{\text{peak}}, \text{AI} \times \text{BW})$).
- **Literature Review**: Synthesis of three landmark studies:
  1. *Ootomo & Yokota (2023)*: Demonstrating that shared-memory bandwidth constrains Tensor Core throughput and analyzing register-blocking tradeoffs.
  2. *Choi & Stegailov (2022)*: Analyzing tile-size sensitivity and communication bottlenecks in multi-GPU GEMM.
  3. *Anastasiadis et al. (2024, Uncut-GEMMs)*: System-level static scheduling, communication-computation overlap, and hierarchical decomposition.
- **Sequential CPU Reference Implementation**:
  - `ijk_gemm`: Naive nested triple loop with stride-$N$ memory access on matrix $B$.
  - `ikj_gemm`: Loop-reordered variant providing unit-stride access for both $B$ and $C$, yielding a $3\times$ to $22\times$ performance increase on single-core CPU purely via cache locality.
  - `reference_gemm_highprec`: Independent ground-truth implementation using `long double` precision accumulation.
- **Validation Metric**: Established Frobenius-norm relative error:
  $$\varepsilon_F = \frac{\|C - C_{\text{ref}}\|_F}{\|C_{\text{ref}}\|_F}$$
  Thresholds: $\varepsilon_F < 5 \times 10^{-4}$ and max absolute element-wise error $< 5 \times 10^{-3}$.

### Milestone 2: Naive CUDA Parallelization & Scaling Baseline
- **First Parallel Baseline**: One thread per output element $C_{i,j}$ launched over a 2D grid of $16 \times 16$ thread blocks.
- **Hardware Target**: Tested on NVIDIA Tesla T4 GPU (Turing architecture, compute capability 7.5, 16 GB VRAM) with CUDA 12.2.
- **Direct Global Memory Access**: Threads perform length-$K$ inner products directly from global DRAM without shared memory staging or register reuse, establishing the parallel floor ($O(1)$ FLOP/byte arithmetic intensity).
- **Kernel-Only Timing**: Benchmarking uses `cudaEventRecord` immediately bracketing the kernel launch to measure compute throughput without PCIe host-device transfer distortions.
- **Cross-Validation**: Confirms numerical correctness (8/8 test cases pass) against both the CPU sequential `ikj_gemm` and the high-precision `long double` oracle across problem sizes $N \in \{8, 16, 32, 63, 64, 129, 200, 256\}$.
- **Scalability Sweep**: Benchmarking across square matrix dimensions $N \in [64, 4096]$, sustaining up to **423.05 GFLOP/s** and achieving up to **47.02× speedup** over single-core CPU execution.

---

## File & Module Details

### 1. [gemm_cuda_naive.cu](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu)
A self-contained CUDA C++ source file providing the kernel, host wrapper, CPU reference kernels, validation suite, and benchmark runner.

- **[gemm_naive_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L78-L91)**:
  - CUDA global kernel launched with `dim3 blockDim(16, 16)`.
  - Determines matrix indices `row = blockIdx.y * blockDim.y + threadIdx.y` and `col = blockIdx.x * blockDim.x + threadIdx.x`.
  - Performs boundary checking (`row < M && col < N`).
  - Accumulates the inner product over dimension $K$ using global memory fetches:
    `acc += A[row * K + k] * B[k * N + col]`.
  - Updates output element: `C[row * N + col] = alpha * acc + beta * C[row * N + col]`.

- **[gemm_naive_launch](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L96-L144)**:
  - Manages device memory allocation (`cudaMalloc`), host-to-device transfers (`cudaMemcpy`), grid dimension calculation, kernel execution, and device-to-host writeback.
  - Optional kernel-only timing via CUDA event profiling (`cudaEventRecord`, `cudaEventElapsedTime`).

- **[ikj_gemm](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L152-L166)**:
  - CPU reference kernel with loop order $i \rightarrow k \rightarrow j$.
  - Preserves row-major spatial locality for $B$ and $C$ in the innermost loop.
  - Used as the reference speedup denominator and secondary correctness oracle.

- **[reference_gemm_highprec](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L171-L186)**:
  - Independent CPU oracle accumulating floating-point operations into `long double` scalars to provide ground-truth precision.

- **[compare](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L209-L222)**:
  - Computes the maximum absolute difference and the Frobenius-norm relative residual against references.

- **[run_validation](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L229-L282)**:
  - Automated test harness checking problem dimensions $8, 16, 32, 63, 64, 129, 200, 256$ with random data seeded for determinism ($\alpha = 1.3, \beta = 0.7$).
  - Validates `C_cuda` against both `C_seq` and `C_ref`.

- **[run_benchmark](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L291-L355)**:
  - Sweeps square problem sizes $N \in \{64, 128, 256, 384, 512, 768, 1024, 2048, 4096\}$.
  - Computes theoretical FLOP count: $\text{FLOPs} = 2N^3$.
  - Measures elapsed kernel time and achieved GFLOP/s:
    $$\text{GFLOP/s} = \frac{2N^3}{t_{\text{kernel}} \times 10^9}$$
  - Logs results to stdout and writes CSV records to `results_milestone2.csv`.

- **[main](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L357-L379)**:
  - Detects active CUDA device and properties; parses execution mode arguments (`validate`, `bench`, or `all`).

---

### 2. [Milestone_2.ipynb](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/Milestone_2.ipynb)
Jupyter notebook preconfigured for Google Colab environments:
- Sets compilation flags for NVIDIA Tesla T4 (`-arch=sm_75 -O3 -std=c++14`).
- Executes `./gemm_cuda_naive all` and embeds raw test outputs showing pass status and performance metrics.

---

### 3. [results_milestone2.csv](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/results_milestone2.csv)
Tabulated empirical run data comparing CPU sequential execution against the naive CUDA kernel on an NVIDIA Tesla T4 GPU.

---

### 4. Technical Reports & Presentation
- **[Problem Definition.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%201/Problem%20Definition.pdf)**: Formalizes the project scope, background, and research questions.
- **[Lit Review.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%201/Lit%20Review.pdf)**: State-of-the-art literature review covering GPU memory constraints, multi-GPU load balancing, and Tensor Core data starvation.
- **[Sequential Implementation.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%201/Sequential%20Implementation.pdf)**: Report for Milestone 1 documenting CPU experiments, loop permutations, and Frobenius residual validation.
- **[Milestone_2 FINAL PC.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/Milestone_2%20FINAL%20PC.pdf)**: Complete academic report for Milestone 2 analyzing naive kernel scalability, Roofline bounds, and architectural performance bottlenecks.
- **[PPT_PC.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/PPT_PC.pdf)**: 35-slide presentation deck providing visual illustrations of the GPU execution model, memory hierarchy, experimental results, and upcoming optimization phases.

---

## Empirical Results

The following table summarizes the measured performance recorded in [results_milestone2.csv](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/results_milestone2.csv) (benchmarked on an **NVIDIA Tesla T4 GPU** vs. single-core virtualized **Intel Xeon CPU @ 2.10 GHz**):

| Matrix Size ($N$) | CPU `ikj` Time (s) | CPU `ikj` (GFLOP/s) | CUDA Naive Time (s) | CUDA Naive (GFLOP/s) | Speedup vs. CPU |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **64** | 0.000105 | 5.000 | 0.000014 | 36.571 | **7.31×** |
| **128** | 0.000375 | 11.197 | 0.000021 | 201.960 | **18.04×** |
| **256** | 0.003330 | 10.076 | 0.000111 | 302.707 | **30.04×** |
| **384** | 0.011031 | 10.266 | 0.000313 | 361.264 | **35.19×** |
| **512** | 0.028843 | 9.307 | 0.000709 | 378.718 | **40.69×** |
| **768** | 0.107621 | 8.418 | 0.002289 | 395.850 | **47.02×** |
| **1024** | 0.207370 | 10.356 | 0.005379 | 399.248 | **38.55×** |
| **2048** | *skipped* | *skipped* | 0.043356 | 396.250 | — |
| **4096** | *skipped* | *skipped* | 0.324879 | 423.047 | — |

### Key Findings & Bottleneck Analysis
1. **Low Computational Intensity**: Each thread repeatedly fetches an entire row of $A$ and column of $B$ from global memory for every multiply-accumulate operation, resulting in an arithmetic intensity of $O(1)$ FLOP/byte.
2. **Bandwidth Saturation**: Throughput scales rapidly from $N=64$ to $N=768$ as thread launch overheads are amortized and GPU occupancy increases. For $N \ge 1024$, performance plateaus between $\sim 396$ and $423$ GFLOP/s because execution is strictly limited by off-chip DRAM memory bandwidth.
3. **Motivation for Shared-Memory Tiling**: To break past the bandwidth ceiling and progress along the Roofline curve toward compute-bound execution, input matrices must be staged in fast on-chip shared memory for reuse across thread blocks.

---

## Build and Execution Guide

### Compilation
The CUDA program requires the NVIDIA CUDA Toolkit (`nvcc` compiler) supporting C++14:

```bash
cd "Milestone 2"

# Compile with target architecture (e.g., sm_75 for Tesla T4 / RTX 20-series)
nvcc -O3 -arch=sm_75 -std=c++14 -o gemm_cuda_naive gemm_cuda_naive.cu
```

> **Target Architecture Reference (`-arch=sm_XX`)**:
> - `sm_70`: NVIDIA Volta (V100)
> - `sm_75`: NVIDIA Turing (Tesla T4, RTX 2080)
> - `sm_80` / `sm_86`: NVIDIA Ampere (A100, RTX 3080)
> - `sm_89`: NVIDIA Ada Lovelace (RTX 4090, L4)
> - `sm_90`: NVIDIA Hopper (H100)

### Execution Modes
The compiled binary supports three execution modes:

```bash
# 1. Run validation test suite only (checks 8 problem sizes against sequential and high-prec oracles)
./gemm_cuda_naive validate

# 2. Run performance/scaling benchmark only (sweeps N=64 to N=4096 and writes results_milestone2.csv)
./gemm_cuda_naive bench

# 3. Run both validation and benchmarking (default)
./gemm_cuda_naive all
# or simply:
./gemm_cuda_naive
```

### Running on Google Colab
1. Upload [gemm_cuda_naive.cu](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu) to the Colab working environment.
2. Open [Milestone_2.ipynb](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/Milestone_2.ipynb).
3. Ensure runtime hardware accelerator is set to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Execute the compilation and runner cells.

---

## Optimization Roadmap

The project follows a 7-step incremental optimization methodology. At every step, correctness is verified and performance gains are isolated:

```text
[Step 0] Sequential CPU Baseline (ikj_gemm ~21 GFLOP/s)  <-- Implemented (Milestone 1)
   │
[Step 1] Naive CUDA Kernel (Global Memory Only, ~423 GFLOP/s) <-- Implemented (Milestone 2)
   │
[Step 2] Memory Coalescing & Access Pattern Optimization      <-- Planned (Milestone 3)
   │
[Step 3] Shared-Memory Tiling (Block-level data reuse)        <-- Planned (Milestone 3)
   │
[Step 4] Register Blocking (Thread-level tile reuse)          <-- Planned (Milestone 4)
   │
[Step 5] Tile-Size Auto-Tuning & Resource Balancing           <-- Planned (Milestone 4)
   │
[Step 6] Tensor Core Acceleration (CUDA WMMA API)             <-- Planned (Milestone 5)
   │
[Step 7] Final Performance Comparison against cuBLAS          <-- Planned (Milestone 5)
```