# Optimizing a Custom CUDA GEMM Kernel Toward cuBLAS-Level Performance

An incremental, ablation-style high-performance computing project dedicated to implementing, analyzing, and progressively optimizing a custom CUDA General Matrix Multiplication (GEMM) kernel on NVIDIA GPUs, aiming to bridge the performance gap between a naive implementation and the industry-standard NVIDIA cuBLAS library.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Repository Structure](#repository-structure)
- [Implemented Milestones & Technical Progress](#implemented-milestones--technical-progress)
  - [Milestone 1: Problem Definition, Literature Review & CPU Baseline](#milestone-1-problem-definition-literature-review--cpu-baseline)
  - [Milestone 2: Naive CUDA Parallelization & Scaling Baseline](#milestone-2-naive-cuda-parallelization--scaling-baseline)
  - [Milestone 3: Advanced Optimizations & Distributed Scaling](#milestone-3-advanced-optimizations--distributed-scaling)
- [File & Module Details](#file--module-details)
  - [Milestone 1 Modules](#milestone-1-modules)
  - [Milestone 2 Modules](#milestone-2-modules)
  - [Milestone 3 Modules](#milestone-3-modules)
- [Empirical Performance Results](#empirical-performance-results)
  - [Comprehensive Optimization Ladder vs. cuBLAS](#comprehensive-optimization-ladder-vs-cublas)
  - [Device Microbenchmark Ceilings](#device-microbenchmark-ceilings)
  - [Hybrid MPI Distributed Scaling Results](#hybrid-mpi-distributed-scaling-results)
  - [16-Configuration Tile Parameter Sweep](#16-configuration-tile-parameter-sweep)
  - [FP16 Tensor-Core Error Budget Decomposition](#fp16-tensor-core-error-budget-decomposition)
- [Build and Execution Guide](#build-and-execution-guide)
  - [Milestone 2 (Naive CUDA Kernel)](#milestone-2-naive-cuda-kernel)
  - [Milestone 3 (Advanced Optimizations & Benchmarking)](#milestone-3-advanced-optimizations--benchmarking)
  - [Milestone 3 (MPI Distributed Scaling)](#milestone-3-mpi-distributed-scaling)
  - [Running on Google Colab](#running-on-google-colab)
- [Optimization Roadmap](#optimization-roadmap)

---

## Project Overview

General Matrix Multiplication (GEMM) is the fundamental computational kernel underlying modern deep learning architectures, scientific computing, and high-performance linear algebra. The single-precision BLAS Level-3 GEMM operation is defined as:

$$C \leftarrow \alpha AB + \beta C$$

where:
- $A \in \mathbb{R}^{M \times K}$
- $B \in \mathbb{R}^{K \times N}$
- $C \in \mathbb{R}^{M \times N}$
- $\alpha, \beta \in \mathbb{R}$ are scalar coefficients.

For square matrices of dimension $N$ ($M = N = K$), GEMM performs $2N^3$ floating-point operations while operating on $O(N^2)$ matrix elements. While GEMM has a theoretically high operational intensity, naive implementations fail to achieve hardware peak throughput because performance is governed by memory access patterns and data-supply bottlenecks across the memory hierarchy (registers, shared memory, L1/L2 caches, and global DRAM).

### Central Research Question
> *How can a custom CUDA GEMM kernel be systematically optimized through memory hierarchy utilization, tiling, register blocking, and GPU-specific execution techniques to approach the performance of the highly optimized NVIDIA cuBLAS GEMM implementation?*

Rather than treating GPU optimization as a black box, this repository implements a controlled, stepwise experimental pipeline where each optimization stage is isolated, measured, and verified for numerical correctness against trusted CPU references.

---

## Repository Structure

```text
Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/
│
├── README.md                           # Master project documentation
│
├── Milestone 1/                        # Milestone 1: Foundations & Sequential Baseline
│   ├── Problem Definition.pdf          # Problem statement, research question & objectives
│   ├── Lit Review.pdf                  # Literature review on GPU GEMM bottlenecks & roadmap
│   └── Sequential Implementation.pdf   # CPU baseline benchmarks, cache locality & correctness
│
├── Milestone 2/                        # Milestone 2: Initial CUDA Parallelization
│   ├── gemm_cuda_naive.cu              # Complete CUDA naive kernel, CPU references, validation & benchmark harness
│   ├── Milestone_2.ipynb               # Google Colab notebook for GPU compilation and execution
│   ├── results_milestone2.csv          # Benchmark output data (CPU vs. CUDA execution time and GFLOP/s)
│   ├── Milestone_2.pdf                 # Milestone 2 technical report (draft)
│   ├── Milestone_2 FINAL PC.pdf        # Milestone 2 final IEEE-style conference report
│   └── PPT_PC.pdf                      # Comprehensive 35-slide presentation deck covering Milestones 1 & 2
│
└── Milestone3/                         # Milestone 3: Advanced Optimizations & Distributed Scaling
    ├── gemm_cuda_opt.cu                # Advanced CUDA kernels (tiling, double buffering, 2D reg-blocking, WMMA TC)
    ├── gemm_mpi_cuda.cu                # Hybrid MPI + CUDA distributed GEMM with non-blocking overlap
    ├── MILESTONE3_REPORT.md            # Comprehensive technical report (models, benchmarks, error analysis)
    ├── READMEM3.md                     # Directory guide for Milestone 3 files and CLI usage
    ├── Milestone_3.ipynb               # Google Colab execution notebook for Milestone 3
    └── Milestone_3.tex                 # LaTeX report in IEEE conference format
```

---

## Implemented Milestones & Technical Progress

### Milestone 1: Problem Definition, Literature Review & CPU Baseline
- **Conceptual & Architectural Foundations**: Analysis of the GPU execution model (Grids, Thread Blocks, Warps, Threads, Streaming Multiprocessors), the memory hierarchy (Global DRAM, L2 cache, shared memory, registers), and the Roofline Model ($P_{\text{achievable}} = \min(P_{\text{peak}}, \text{AI} \times \text{BW})$).
- **Literature Review**: Synthesis of three landmark studies:
  1. *Ootomo & Yokota (2023)*: Demonstrating that shared-memory bandwidth constrains Tensor Core throughput and analyzing register-blocking tradeoffs.
  2. *Choi & Stegailov (2022)*: Analyzing tile-size sensitivity and communication bottlenecks in multi-GPU GEMM.
  3. *Anastasiadis et al. (2024, Uncut-GEMMs)*: System-level static scheduling, communication-computation overlap, and hierarchical decomposition.
- **Sequential CPU Reference Implementation**:
  - `ijk_gemm`: Naive nested triple loop with stride-$N$ memory access on matrix $B$.
  - `ikj_gemm`: Loop-reordered variant providing unit-stride access for both $B$ and $C$, yielding a $3\times$ to $22\times$ performance increase on single-core CPU purely via cache locality (reaching ~21 GFLOP/s).
  - `reference_gemm_highprec`: Independent ground-truth implementation using `long double` precision accumulation.
- **Validation Metric**: Established Frobenius-norm relative error:
  $$\varepsilon_F = \frac{\|C - C_{\text{ref}}\|_F}{\|C_{\text{ref}}\|_F}$$
  Thresholds: $\varepsilon_F < 5 \times 10^{-4}$ and max absolute element-wise error $< 5 \times 10^{-3}$.

---

### Milestone 2: Naive CUDA Parallelization & Scaling Baseline
- **First Parallel Baseline**: One thread per output element $C_{i,j}$ launched over a 2D grid of $16 \times 16$ thread blocks.
- **Hardware Target**: Tested on NVIDIA Tesla T4 GPU (Turing architecture, compute capability 7.5, 16 GB VRAM) with CUDA 12.2.
- **Direct Global Memory Access**: Threads perform length-$K$ inner products directly from global DRAM without shared memory staging or register reuse, establishing the parallel floor ($O(1)$ FLOP/byte arithmetic intensity).
- **Kernel-Only Timing**: Benchmarking uses `cudaEventRecord` immediately bracketing the kernel launch to measure compute throughput without PCIe host-device transfer distortions.
- **Cross-Validation**: Confirms numerical correctness (8/8 test cases pass) against both the CPU sequential `ikj_gemm` and the high-precision `long double` oracle across problem sizes $N \in \{8, 16, 32, 63, 64, 129, 200, 256\}$.
- **Scalability Sweep**: Benchmarking across square matrix dimensions $N \in [64, 4096]$, sustaining up to **423.05 GFLOP/s** and achieving up to **47.02× speedup** over single-core CPU execution.

---

### Milestone 3: Advanced Optimizations & Distributed Scaling
- **Re-evaluation of the Milestone 2 Bottleneck**: Measured machine ceilings (DRAM copy BW = 242 GB/s, FP32 FMA peak = 6,854 GFLOP/s). Proved the naive kernel requested **1.69 TB/s of operand traffic** ($7\times$ DRAM bandwidth), confirming that the naive bottleneck was **L1/L2 cache and load-store unit (LSU) throughput**, not raw DRAM streaming.
- **Shared-Memory Tiling**: Explored tile sizes $T \in \{8, 16, 32\}$, staging inputs into on-chip `__shared__` memory and boosting throughput to **1,515 GFLOP/s**.
- **Communication Hiding via Double Buffering**: Implemented a 2-stage software pipeline using double buffers (`As[2]`, `Bs[2]`) in shared memory. Prefetches global memory for tile $k+1$ into registers while computing on tile $k$, pushing throughput to **1,842 GFLOP/s** ($4.35\times$ over naive).
- **Vectorization (float2 and float4)**: Utilized 128-bit `float4` loads to reduce memory instruction overhead by 75% and maximize memory bus transaction efficiency.
- **2-D Register Blocking**: Transposed tile $A$ in shared memory to eliminate bank conflicts and held $T_M \times T_N = 8 \times 8$ micro-tiles in thread registers, achieving **5,114.2 GFLOP/s** (**85.7% of cuBLAS SGEMM**).
- **Tensor Core Acceleration (WMMA API)**: Integrated NVIDIA Turing Tensor Cores ($16 \times 16 \times 16$ half-precision fragments with FP32 accumulation), reaching **18,740.5 GFLOP/s** (**18.7 TFLOP/s**, 78.2% of cuBLAS Tensor-Op).
- **Epilogue Fusion**: Fused bias addition and ReLU activation ($Y = \max(\alpha AB + \beta C + \mathbf{b}, 0)$) directly into the store phase, eliminating an entire DRAM roundtrip and outperforming cuBLAS + separate elementwise pass by up to **1.15×**.
- **Host-Device Panel Pipelining**: Decomposed matrix into row-panels across multiple non-blocking CUDA streams, overlapping PCIe transfers ($H2D$ and $D2H$) with GPU execution to hide 62% of transfer overhead.
- **Hybrid MPI + CUDA Distributed Scaling**: Sliced matrices across distributed nodes using 1D block-row domain decomposition and non-blocking MPI communication (`MPI_Ibcast`, `MPI_Isend`, `MPI_Irecv`), scaling efficiently across multiple ranks.

---

## File & Module Details

### Milestone 1 Modules
- **[Milestone 1/Problem Definition.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%201/Problem%20Definition.pdf)**: Formalizes the mathematical problem statement and research scope.
- **[Milestone 1/Lit Review.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%201/Lit%20Review.pdf)**: State-of-the-art literature review on GPU memory constraints and Tensor Core bottlenecks.
- **[Milestone 1/Sequential Implementation.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%201/Sequential%20Implementation.pdf)**: Documents CPU experiments, loop permutations, and Frobenius residual validation.

### Milestone 2 Modules
- **[Milestone 2/gemm_cuda_naive.cu](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu)**:
  - [gemm_naive_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L78-L91): Naive 1-thread-per-element kernel.
  - [ikj_gemm](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L152-L166): CPU sequential reference.
  - [reference_gemm_highprec](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L171-L186): `long double` ground truth.
  - [run_validation](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L229-L282): Validates 8 problem sizes ($N \in \{8, \dots, 256\}$).
  - [run_benchmark](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/gemm_cuda_naive.cu#L291-L355): Sweeps $N \in [64, 4096]$ and writes `results_milestone2.csv`.
- **[Milestone 2/results_milestone2.csv](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/results_milestone2.csv)**: Empirical run data for the naive kernel.
- **[Milestone 2/Milestone_2 FINAL PC.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/Milestone_2%20FINAL%20PC.pdf)** & **[Milestone 2/PPT_PC.pdf](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/PPT_PC.pdf)**: IEEE report and 35-slide presentation.

### Milestone 3 Modules
- **[Milestone3/gemm_cuda_opt.cu](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu)**:
  - [gemm_naive_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L100-L112) & [gemm_naive_uncoal_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L118-L130): Baseline and uncoalesced control.
  - [gemm_smem_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L133-L154): Shared-memory tiling ($T=8, 16, 32$).
  - [gemm_smem_double_buffer_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L162-L208): 2-stage software-pipelined double buffering for communication hiding.
  - [gemm_reg2d_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L218-L304): 2D register blocking ($B_M \times B_N \times B_K$, $T_M \times T_N$) with `float4` vectorization and transposed shared memory.
  - [gemm_wmma_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L320-L388): NVIDIA Tensor Core acceleration using WMMA API.
  - [bias_relu_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L398-L409) & fused epilogue: Fused GEMM + bias + ReLU.
  - [copy_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L412-L416) & [fma_peak_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L421-L439): Microbenchmarks measuring DRAM bandwidth and FP32 FMA peak.
  - [run_micro](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L664-L733): Machine ceiling calibration harness.
  - [run_validation](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L738-L829): 11-shape correctness validation suite (including non-tile-aligned rectangular shapes).
  - [run_benchmark](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L832-L894): 7-rung ablation ladder vs cuBLAS SGEMM and Tensor-Op.
  - [run_tuning](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L896-L936): 16-configuration tile sweep measuring registers, spills, and occupancy.
  - [run_precision](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L939-L998): FP16 error budget decomposition (input quantization vs accumulation).
  - [run_pipeline](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L1032-L1127): Multi-stream panel pipelining overlapping PCIe transfers with kernel compute.
  - [run_fusion](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_cuda_opt.cu#L1130-L1195): Fused epilogue evaluation against cuBLAS + separate pass.
- **[Milestone3/gemm_mpi_cuda.cu](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_mpi_cuda.cu)**:
  - [gemm_smem_kernel](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_mpi_cuda.cu#L57-L88): Local GPU kernel.
  - [local_gpu_gemm](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_mpi_cuda.cu#L123-L157): Rank-specific GPU memory and execution wrapper.
  - [run_mpi_distributed](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_mpi_cuda.cu#L164-L245): Real MPI distributed execution with Scatterv, Bcast, and Gatherv.
  - [run_standalone_simulation](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/gemm_mpi_cuda.cu#L251-L328): Multi-rank simulation model with non-blocking overlap.
- **[Milestone3/MILESTONE3_REPORT.md](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/MILESTONE3_REPORT.md)**:
  - 22 KB comprehensive technical engineering report with analytical traffic models, empirical comparison tables, error analysis, and scaling charts.
- **[Milestone3/READMEM3.md](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/READMEM3.md)**:
  - Directory documentation covering Milestone 3 files, compilation commands, and CLI options.
- **[Milestone3/Milestone_3.ipynb](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/Milestone_3.ipynb)** & **[Milestone3/Milestone_3.tex](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/Milestone_3.tex)**:
  - Interactive Colab execution notebook and IEEE conference report source.

---

## Empirical Performance Results

All GPU benchmarks were measured on an **NVIDIA Tesla T4 GPU** (Turing architecture, 40 SMs, 16 GB GDDR6) under CUDA 12.2, with timing recorded via `cudaEvent` over adaptive repetitions (median reported).

### Comprehensive Optimization Ladder vs. cuBLAS
Measured at $N = 4096$ (square single-precision GEMM, $\alpha=1.0, \beta=0.0$):

| Stage | Implementation / Kernel Variant | Median Time (ms) | Achieved Throughput | % of cuBLAS SGEMM | Speedup vs. Naive M2 |
|:---:|---|:---:|:---:|:---:|:---:|
| **—** | CPU Sequential Reference (`ikj_gemm`, single-core) | 184.42 ms$^*$ | 11.6 GFLOP/s | 0.2% | — |
| **0a** | Naive Uncoalesced Control (`threadIdx.x -> row`) | 927.42 ms | 148.2 GFLOP/s | 2.5% | 0.35× |
| **0b** | **Naive CUDA Baseline (Milestone 2)** | 324.88 ms | **423.0 GFLOP/s** | 7.1% | **1.00×** |
| **1a** | Shared-Memory Tiling ($T=8$) | 169.18 ms | 812.4 GFLOP/s | 13.6% | 1.92× |
| **1b** | Shared-Memory Tiling ($T=16$) | 96.76 ms | 1,420.5 GFLOP/s | 23.8% | 3.36× |
| **1c** | Shared-Memory Tiling ($T=32$) | 90.73 ms | 1,514.8 GFLOP/s | 25.4% | 3.58× |
| **2** | **Double-Buffered Shared Memory ($T=16$)** | 74.61 ms | **1,842.1 GFLOP/s** | 30.9% | **4.35×** |
| **3** | 2D Register-Blocked ($128 \times 128 \times 8$, scalar loads) | 32.64 ms | 4,210.6 GFLOP/s | 70.6% | 9.95× |
| **4** | **2D Register-Blocked ($128 \times 128 \times 8$, `float4` vec)** | **26.87 ms** | **5,114.2 GFLOP/s** | **85.7%** | **12.09×** |
| **5** | **WMMA Tensor Cores (FP16 in, FP32 accumulate)** | **7.33 ms** | **18,740.5 GFLOP/s** | **78.2%**$^\dagger$ | **44.30×** |
| **Ref 1** | **NVIDIA cuBLAS SGEMM** (FP32 library target) | **23.04 ms** | **5,965.8 GFLOP/s** | **100.0%** | **14.10×** |
| **Ref 2** | **NVIDIA cuBLAS Tensor-Op** (FP16 library target) | **5.74 ms** | **23,960.0 GFLOP/s** | **100.0%**$^\dagger$ | **56.64×** |

*$^*$CPU timing evaluated at $N=1024$ and scaled to operation count for comparison.*  
*$^\dagger$Evaluated relative to cuBLAS FP16 Tensor-Op.*

---

### Device Microbenchmark Ceilings
| Metric / Quantity | Datasheet Nominal | Measured Empirical Ceiling |
|---|:---:|:---:|
| **FP32 Compute Peak** | 8,141 GFLOP/s | **6,854 GFLOP/s** (8-chain FMA microbenchmark) |
| **DRAM Memory Bandwidth** | 320 GB/s | **242 GB/s** (float4 copy microbenchmark) |
| **Observed SM Boost Clock** | 1.59 GHz (max) | **1.34 GHz** (clock64 / globaltimer) |
| **Shared-Memory Bandwidth** | — | **6,861 GB/s** (6.86 TB/s @ 128 B/cycle/SM) |

---

### Hybrid MPI Distributed Scaling Results
Evaluated across simulated ranks $P \in \{1, 2, 4, 8\}$ for $M=N=K=2048$ with non-blocking communication overlap:

| Ranks ($P$) | Compute Time ($T_{\text{comp}}$) | Comm Time ($T_{\text{comm}}$) | Serial Total Time | Overlapped Total Time | Speedup vs. 1 GPU | Parallel Efficiency |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | 6.84 ms | 0.00 ms | 6.84 ms | **6.84 ms** | **1.00×** | **100.0%** |
| **2** | 3.42 ms | 1.82 ms | 5.24 ms | **3.85 ms** | **1.78×** | **88.8%** |
| **4** | 1.71 ms | 1.34 ms | 3.05 ms | **2.12 ms** | **3.23×** | **80.7%** |
| **8** | 0.86 ms | 1.10 ms | 1.96 ms | **1.38 ms** | **4.96×** | **61.9%** |

---

### 16-Configuration Tile Parameter Sweep
Across 16 configurations ($B_M \times B_N \times B_K$ block tiles and $T_M \times T_N$ thread tiles at $N=4096$):

| Configuration Name | Tile Dimensions ($B_M \times B_N \times B_K$) | Thread Tile ($T_M \times T_N$) | Vectorized? | Regs/Thread | Blocks/SM | GFLOP/s | Model Ceiling |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `r128x128x8_t8x8_v` | $128 \times 128 \times 8$ | $8 \times 8$ | Yes (`float4`) | 64 | 2 | **5,114** | 6,854 |
| `r128x64x8_t8x4_v` | $128 \times 64 \times 8$ | $8 \times 4$ | Yes (`float4`) | 48 | 3 | **4,860** | 6,854 |
| `r64x128x8_t4x8_v` | $64 \times 128 \times 8$ | $4 \times 8$ | Yes (`float4`) | 48 | 3 | **4,790** | 6,854 |
| `r128x128x8_t8x8` | $128 \times 128 \times 8$ | $8 \times 8$ | No (scalar) | 64 | 2 | **4,210** | 6,854 |
| `r64x64x16_t4x4_v` | $64 \times 64 \times 16$ | $4 \times 4$ | Yes (`float4`) | 32 | 4 | **3,150** | 4,288 |
| `r32x32x8_t2x2` | $32 \times 32 \times 8$ | $2 \times 2$ | No (scalar) | 24 | 6 | **1,840** | 2,144 |

---

### FP16 Tensor-Core Error Budget Decomposition
Evaluating numerical error vs. $K$ dimension:

| $K$ Dimension | Signed Input Total Error ($\varepsilon_{\text{total}}$) | Signed Accumulation Error ($\varepsilon_{\text{accum}}$) | Non-Negative Total Error ($\varepsilon_{\text{total}}$) | Non-Negative Accumulation Error ($\varepsilon_{\text{accum}}$) |
|:---:|:---:|:---:|:---:|:---:|
| **16** | $4.82 \times 10^{-4}$ | $2.14 \times 10^{-7}$ | $4.76 \times 10^{-4}$ | $1.98 \times 10^{-7}$ |
| **64** | $4.91 \times 10^{-4}$ | $3.25 \times 10^{-7}$ | $2.41 \times 10^{-4}$ | $3.01 \times 10^{-7}$ |
| **256** | $4.87 \times 10^{-4}$ | $5.12 \times 10^{-7}$ | $1.22 \times 10^{-4}$ | $4.89 \times 10^{-7}$ |
| **1024** | $4.95 \times 10^{-4}$ | $8.40 \times 10^{-7}$ | $6.15 \times 10^{-5}$ | $7.62 \times 10^{-7}$ |
| **4096** | $4.90 \times 10^{-4}$ | $1.41 \times 10^{-6}$ | $3.10 \times 10^{-5}$ | $1.28 \times 10^{-6}$ |

---

## Build and Execution Guide

### Milestone 2 (Naive CUDA Kernel)
```bash
cd "Milestone 2"

# Compile with target architecture (e.g. sm_75 for Turing / Tesla T4)
nvcc -O3 -arch=sm_75 -std=c++14 -o gemm_cuda_naive gemm_cuda_naive.cu

# Execution options:
./gemm_cuda_naive validate    # Run validation test suite only
./gemm_cuda_naive bench       # Run performance sweep (writes results_milestone2.csv)
./gemm_cuda_naive all         # Run both validation and benchmark (default)
```

### Milestone 3 (Advanced Optimizations & Benchmarking)
```bash
cd "Milestone3"

# Compile with target architecture and cuBLAS:
nvcc -O3 -arch=sm_75 -std=c++14 -lineinfo -o gemm_cuda_opt gemm_cuda_opt.cu -lcublas

# Execution modes:
./gemm_cuda_opt micro       # Measure device ceilings (DRAM copy BW, FP32 FMA peak, clock)
./gemm_cuda_opt validate    # Run 11-shape correctness validation suite
./gemm_cuda_opt bench       # Run full 7-rung ablation ladder vs cuBLAS
./gemm_cuda_opt tune        # Run 16-configuration tile sensitivity sweep
./gemm_cuda_opt precision   # Run FP16 vs FP32 error budget decomposition
./gemm_cuda_opt pipe        # Run multi-stream panel pipelining benchmark
./gemm_cuda_opt fuse        # Run fused epilogue (ReLU(AB + bias)) study
./gemm_cuda_opt all         # Run all benchmark suites sequentially (default)
```

### Milestone 3 (MPI Distributed Scaling)
```bash
cd "Milestone3"

# 1. Standalone Simulation Mode (No MPI required):
nvcc -O3 -std=c++14 -o gemm_mpi_cuda gemm_mpi_cuda.cu -lcublas
./gemm_mpi_cuda 2048 2048 2048

# 2. Native MPI Multi-Process Mode (with OpenMPI/MPICH):
mpicxx -O3 -std=c++14 -DUSE_MPI -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -o gemm_mpi_cuda gemm_mpi_cuda.cu
mpirun -np 4 ./gemm_mpi_cuda 2048 2048 2048
```

### Running on Google Colab
1. Upload the respective directory files (`gemm_cuda_naive.cu` for Milestone 2 or `gemm_cuda_opt.cu` / `gemm_mpi_cuda.cu` for Milestone 3) to the Colab working environment.
2. Open [Milestone 2/Milestone_2.ipynb](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone%202/Milestone_2.ipynb) or [Milestone3/Milestone_3.ipynb](file:///d:/BITS/acads/4-1/Parallel%20computing/Optimizing-a-Custom-CUDA-GEMM-Kernel-Toward-cuBLAS-Level-Performance/Milestone3/Milestone_3.ipynb).
3. Ensure the runtime accelerator is set to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
4. Execute the compilation and runner cells.

---

## Optimization Roadmap

The project follows a 7-step incremental optimization methodology. At every step, correctness is verified and performance gains are isolated:

```text
[Step 0] Sequential CPU Baseline (ikj_gemm ~21 GFLOP/s)                <-- Completed (Milestone 1)
   │
[Step 1] Naive CUDA Kernel (Global Memory Only, ~423 GFLOP/s)          <-- Completed (Milestone 2)
   │
[Step 2] Memory Coalescing & Access Pattern Analysis                   <-- Completed (Milestone 3)
   │
[Step 3] Shared-Memory Tiling (T=8, 16, 32, ~1,515 GFLOP/s)           <-- Completed (Milestone 3)
   │
[Step 4] Communication Hiding & Double Buffering (~1,842 GFLOP/s)      <-- Completed (Milestone 3)
   │
[Step 5] 2-D Register Blocking + float4 Vectorization (~5,114 GFLOP/s) <-- Completed (Milestone 3)
   │
[Step 6] Tensor Core Acceleration (NVIDIA WMMA API, ~18,740 GFLOP/s)   <-- Completed (Milestone 3)
   │
[Step 7] Epilogue Fusion, Stream Pipelining & Distributed MPI Scaling  <-- Completed (Milestone 3)
   │
[Future] Hardware Performance Counters Profiling (Nsight Compute)      <-- Planned (Milestone 4)
```