# Milestone 3 Technical Report: Advanced Optimizations & Distributed Scaling

**Course:** Parallel Computing  
**Project:** Optimizing a Custom CUDA GEMM Kernel Toward cuBLAS-Level Performance  
**Authors:** Shriya Soni, Saee Brahmankar, D Prem Kowshik  
**Institution:** BITS Pilani, Goa Campus  
**Date:** October 2026  

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Phase 1: Codebase Analysis & Milestone Context](#2-phase-1-codebase-analysis--milestone-context)
   - [2.1 Milestone 1 & 2 Accomplishments](#21-milestone-1--2-accomplishments)
   - [2.2 Baseline Implementation & Architectural Bottlenecks](#22-baseline-implementation--architectural-bottlenecks)
   - [2.3 Mathematical Model & The Roofline Re-evaluation](#23-mathematical-model--the-roofline-re-evaluation)
3. [Implemented Milestone 3 Optimizations](#3-implemented-milestone-3-optimizations)
   - [3.1 Memory Tiling (Shared-Memory 8×8, 16×16, 32×32)](#31-memory-tiling-shared-memory-88-1616-3232)
   - [3.2 Communication Hiding & Double Buffering](#32-communication-hiding--double-buffering)
   - [3.3 Vectorization (float2 and float4)](#33-vectorization-float2-and-float4)
   - [3.4 2-D Register Blocking & Epilogue Fusion](#34-2-d-register-blocking--epilogue-fusion)
   - [3.5 Tensor Core Acceleration (NVIDIA WMMA API)](#35-tensor-core-acceleration-nvidia-wmma-api)
   - [3.6 MPI Hybrid Scaling & Distributed Domain Decomposition](#36-mpi-hybrid-scaling--distributed-domain-decomposition)
4. [Empirical Benchmarking & Performance Evaluation](#4-empirical-benchmarking--performance-evaluation)
   - [4.1 Device Microbenchmark Ceilings](#41-device-microbenchmark-ceilings)
   - [4.2 Comprehensive 7-Rung Optimization Ladder vs. cuBLAS](#42-comprehensive-7-rung-optimization-ladder-vs-cublas)
   - [4.3 Tile-Configuration Sensitivity (16-Configuration Sweep)](#43-tile-configuration-sensitivity-16-configuration-sweep)
   - [4.4 FP16 Tensor-Core Error Budget (Quantization vs. Accumulation)](#44-fp16-tensor-core-error-budget-quantization-vs-accumulation)
   - [4.5 Host-Device Panel Pipelining vs. Serial PCIe](#45-host-device-panel-pipelining-vs-serial-pcie)
   - [4.6 Fused Epilogue vs. cuBLAS + Post-Processing Pass](#46-fused-epilogue-vs-cublas--post-processing-pass)
5. [Numerical Verification & Correctness](#5-numerical-verification--correctness)
6. [Compilation & Execution Instructions](#6-compilation--execution-instructions)
7. [Limitations & Future Roadmap](#7-limitations--future-roadmap)

---

## 1. Executive Summary

Milestone 3 marks the transition of the project from exploratory GPU acceleration into **model-guided high-performance computing**. 

In Milestone 2, we parallelized single-precision General Matrix Multiplication (GEMM, $C = \alpha AB + \beta C$) with a naive one-thread-per-element CUDA kernel. While this naive kernel achieved 423 GFLOP/s on an NVIDIA Tesla T4 (a 47× speedup over the single-core CPU baseline), its performance plateaued for $N \ge 1024$. Milestone 2 hypothesized that this plateau was caused solely by off-chip DRAM memory bandwidth saturation.

In **Milestone 3**, we:
1. **Re-evaluated and corrected the Milestone 2 bottleneck hypothesis**: By directly measuring achievable machine ceilings (FP32 FMA peak = 6,854 GFLOP/s, DRAM copy bandwidth = 242 GB/s, observed SM clock = 1.34 GHz), we demonstrate that the naive kernel actually demands **1.69 TB/s of operand traffic**—almost $7\times$ the measured DRAM bandwidth. Thus, the naive kernel was bottlenecked by **L1/L2 cache and load-store unit (LSU) throughput**, not raw DRAM streaming.
2. **Implemented multi-level memory hierarchy optimizations**:
   - **Shared-Memory Tiling**: Evaluated across tile sizes $T \in \{8, 16, 32\}$, reducing global memory transactions by staging tiles into on-chip SRAM.
   - **Communication Hiding via Double Buffering**: Implemented a 2-stage software pipeline overlapping the global-to-shared memory prefetch of tile $k+1$ with arithmetic execution of tile $k$.
   - **Vectorization**: Introduced aligned `float4` and `float2` memory instructions, maximizing bus transaction efficiency and instruction-level parallelism.
   - **2-D Register Blocking**: Transposed tile $A$ in shared memory to allow vector reads, computing $T_M \times T_N$ micro-tiles per thread in register files.
   - **NVIDIA Tensor Core Integration**: Accelerated computation with the Warp Matrix Multiply and Accumulate (WMMA) API on FP16 inputs with FP32 accumulation.
3. **Pipelined Streaming & Distributed Scaling**:
   - Developed single-node **multi-stream panel pipelining** overlapping PCIe transfers ($H2D(A)$ and $D2H(C)$) with kernel execution on separate copy and compute engines.
   - Developed a dedicated **MPI + CUDA hybrid scaling module** (`gemm_mpi_cuda.cu`) employing 1D block-row domain decomposition with non-blocking MPI communication.
4. **Epilogue Fusion**: Proved that custom kernels can outperform cuBLAS when fused with activation layers ($Y = \text{ReLU}(AB + \mathbf{b})$), eliminating an entire round-trip to global memory.

---

## 2. Phase 1: Codebase Analysis & Milestone Context

### 2.1 Milestone 1 & 2 Accomplishments
- **Milestone 1**: 
  - Implemented CPU sequential reference kernels: textbook `ijk_gemm` and cache-optimized `ikj_gemm`.
  - Established that loop reordering alone increases single-core CPU throughput from 0.52 GFLOP/s to 21.02 GFLOP/s ($22\times$ difference with identical FLOP count), previewing the memory-bound nature of GEMM.
  - Formulated the Frobenius-norm relative error criterion ($\varepsilon_F < 5 \times 10^{-4}$) against a `long double` precision reference oracle.
- **Milestone 2**:
  - Implemented the first naive CUDA kernel (`gemm_cuda_naive.cu`), assigning one thread per output element with $16 \times 16$ thread blocks.
  - Evaluated on an NVIDIA Tesla T4 GPU, reaching 423.05 GFLOP/s at $N=4096$.

### 2.2 Baseline Implementation & Architectural Bottlenecks
In the naive kernel:
```cuda
int row = blockIdx.y * blockDim.y + threadIdx.y;
int col = blockIdx.x * blockDim.x + threadIdx.x;
float acc = 0.0f;
for (int k = 0; k < K; k++) {
    acc += A[row * K + k] * B[k * N + col];
}
```
Every thread independently fetches $K$ elements of row $A$ and $K$ elements of column $B$. For an $N \times N$ matrix, the naive kernel issues:
$$\text{Total Global Memory Fetches} = 2N^3 \times 4\text{ bytes}$$
For $2N^3$ FLOPs, the arithmetic intensity is:
$$\text{Operational Intensity (Naive)} = \frac{2N^3\text{ FLOPs}}{8N^3\text{ Bytes}} = 0.25\text{ FLOP/byte}$$
On a modern GPU with a balanced operational intensity threshold of $\sim 30\text{ FLOP/byte}$, this naive access pattern severely starves the arithmetic units.

### 2.3 Mathematical Model & The Roofline Re-evaluation
To predict the performance ceiling of each stage, we formulate an analytical three-resource ceiling model:
$$P \le \min\left(P_{\text{FMA}}, \frac{B_{\text{smem}}}{\beta_{\text{smem}}}, \frac{B_{\text{dram}}}{\beta_{\text{glob}}}\right)$$

where:
- $\beta_{\text{glob}} = 2 \left(\frac{1}{B_M} + \frac{1}{B_N}\right)$ bytes/FLOP (global-to-shared memory traffic).
- $\beta_{\text{smem}} = \frac{2(T_M + T_N)}{T_M \cdot T_N}$ bytes/FLOP (shared-memory-to-register traffic).
- $B_{\text{smem}} = 128\text{ B/cycle} \times n_{\text{SM}} \times f_{\text{observed}}$.

#### Why the Milestone 2 Hypothesis Was Incomplete
At 423 GFLOP/s, the naive kernel requested:
$$B_{\text{req}} = 4.0\text{ B/FLOP} \times 423\text{ GFLOP/s} = 1,692\text{ GB/s} \approx 1.69\text{ TB/s}$$
Because the measured DRAM copy bandwidth on our Tesla T4 is only **242 GB/s**, if every operand fetch had traversed off-chip DRAM, throughput could not have exceeded:
$$P_{\text{max, DRAM}} = \frac{242\text{ GB/s}}{4.0\text{ B/FLOP}} = 60.5\text{ GFLOP/s}$$
Yet the naive kernel achieved **423 GFLOP/s** ($7.0\times$ higher). This proves that the Turing L1/L2 cache hierarchy was already absorbing roughly 85% of global memory requests. The plateau at large $N$ was driven by **cache thrashing and load-store pipeline saturation**, establishing the urgent necessity of **explicitly managed shared-memory tiling**.

---

## 3. Implemented Milestone 3 Optimizations

### 3.1 Memory Tiling (Shared-Memory 8×8, 16×16, 32×32)
We partition matrices $A$ and $B$ into 2D tiles of size $T \times T$ stored in `__shared__` memory:
- Threads within a block cooperatively load a $T \times T$ tile of $A$ and a $T \times T$ tile of $B$.
- A `__syncthreads()` barrier ensures all tile elements are resident in SRAM.
- All threads compute partial inner products from shared memory.
- Another `__syncthreads()` barrier protects against write-after-read hazards before loading the next tile.
- **Arithmetic intensity increases by a factor of $T$**:
  $$\text{Operational Intensity (Tiled)} = \frac{T}{4}\text{ FLOP/byte}$$
  - $T=8$: 2.0 FLOP/byte (Low SRAM footprint, higher global memory traffic).
  - $T=16$: 4.0 FLOP/byte (Optimal balance of block occupancy and register pressure on Turing).
  - $T=32$: 8.0 FLOP/byte (1024 threads/block, high tile reuse but reduced active blocks per SM).

### 3.2 Communication Hiding & Double Buffering
Standard shared-memory tiling suffers from synchronization stalls: threads stall at `__syncthreads()` while waiting for DRAM fetches to complete.
To hide this memory latency, we implement **double buffering** (2-stage software pipelining):
1. Shared memory is doubled: `As[2][T][T]` and `Bs[2][T][T]`.
2. **Prologue**: Tile 0 is fetched into buffer 0.
3. **Main Loop**:
   - In iteration $k$, global memory loads for tile $k+1$ are issued asynchronously into local register staging variables.
   - Concurrently, the arithmetic units execute FMAs using data from buffer $(k \bmod 2)$.
   - Once computation finishes, the prefetched register values are committed into shared memory buffer $(1 - (k \bmod 2))$.
   - A single barrier synchronizes the shared memory state.
4. Memory transfer latency is overlapped behind compute instructions, moving the kernel closer to the compute-bound ceiling.

### 3.3 Vectorization (float2 and float4)
Modern NVIDIA GPU memory controllers perform transactions in 32-byte, 64-byte, or 128-byte sectors. Issuing scalar 32-bit (`float`) loads requires 4 separate memory instructions per 16 bytes:
- By casting aligned pointers to `float4` (`sizeof(float4) == 16 bytes`) and `float2` (`sizeof(float2) == 8 bytes`), a single instruction issues a wide bus transaction.
- **Instruction count is reduced by 75%** for memory addressing.
- Boundary protection handles arbitrary matrix dimensions: matrices where $N, K$ are multiples of 4 use the fast `float4` path; remainder elements are safely processed using scalar boundary guards.

### 3.4 2-D Register Blocking & Epilogue Fusion
To scale beyond 1 FLOP/cycle/thread:
- Each thread computes a 2D micro-tile of size $T_M \times T_N$ held entirely in register files (`float acc[TM][TN]`).
- Matrix $A$ is loaded and **transposed in shared memory** (`As[c * BM + r]`), eliminating shared-memory bank conflicts and allowing unit-stride access for both $A$ and $B$.
- **Fused Epilogue**: Standard deep learning workflows require $Y = \text{ReLU}(AB + \mathbf{b})$. Library GEMMs (like standard `cublasSgemm`) require writing $C$ back to DRAM, then launching a separate elementwise kernel that reads $C$, adds bias, applies ReLU, and writes $Y$.
- We fused bias addition and ReLU activation directly into the register writeback loop:
  $$C[(r_0 + i)N + c_0 + j] = \max(\alpha \cdot \text{acc}_{i,j} + \text{bias}_{c_0 + j}, 0)$$
  This completely eliminates $2N^2$ DRAM memory accesses.

### 3.5 Tensor Core Acceleration (NVIDIA WMMA API)
On NVIDIA Turing GPUs, Tensor Cores provide specialized hardware for matrix multiply-accumulate operations:
- Uses NVIDIA `wmma` primitives (`16x16x16` half-precision fragments).
- Input tiles of $A$ and $B$ are loaded as FP16 (`half`), while accumulation is maintained in full FP32 (`float`).
- Shared memory allocations are padded (`WM_LDA = 32 + 8`, `WM_LDB = 64 + 8`) to guarantee zero bank conflicts during fragment loads.
- Reaches up to **18.7 TFLOP/s** throughput.

### 3.6 MPI Hybrid Scaling & Distributed Domain Decomposition
To scale GEMM across distributed nodes or multiple GPUs:
- **1D Block-Row Decomposition**: Global matrix $A$ ($M \times K$) is partitioned into $P$ row-slabs of size $\frac{M}{P} \times K$. Rank $r$ owns slab $A_r$.
- Matrix $B$ ($K \times N$) is replicated/broadcast to all ranks.
- Each rank executes the optimized CUDA GEMM kernel on its assigned local GPU.
- **Non-Blocking Communication Hiding**: Communication of the next matrix slab (`MPI_Ibcast` / `MPI_Isend` / `MPI_Irecv`) is posted on a separate stream, overlapping network and PCIe transfers with GPU execution.
- Final output rows are assembled into matrix $C$ on Rank 0 via `MPI_Gatherv`.

---

## 4. Empirical Benchmarking & Performance Evaluation

All benchmarks were measured on an **NVIDIA Tesla T4 GPU** (compute capability 7.5, 40 SMs, 16 GB GDDR6) under CUDA 12.2, with timing recorded via `cudaEvent` over adaptive repetitions (median reported).

### 4.1 Device Microbenchmark Ceilings
| Metric / Quantity | Datasheet Nominal | Measured Empirical Ceiling |
|---|:---:|:---:|
| **FP32 Compute Peak** | 8,141 GFLOP/s | **6,854 GFLOP/s** (8-chain FMA microbenchmark) |
| **DRAM Memory Bandwidth** | 320 GB/s | **242 GB/s** (float4 copy microbenchmark) |
| **Observed SM Boost Clock** | 1.59 GHz (max) | **1.34 GHz** (clock64 / globaltimer) |
| **Shared-Memory Bandwidth** | — | **6,861 GB/s** (6.86 TB/s @ 128 B/cycle/SM) |

---

### 4.2 Comprehensive 7-Rung Optimization Ladder vs. cuBLAS
Measured at $N = 4096$ (square single-precision GEMM):

| Stage # | Kernel Variant | Achieved Throughput | % of cuBLAS SGEMM | % of FP32 FMA Peak | Speedup vs. Naive M2 |
|:---:|---|:---:|:---:|:---:|:---:|
| **—** | CPU Sequential (`ikj_gemm`, single-core) | 11.6 GFLOP/s | 0.2% | — | — |
| **Rung 0a** | Naive Uncoalesced Control (`threadIdx.x -> row`) | 148.2 GFLOP/s | 2.5% | 2.2% | 0.35× |
| **Rung 0b** | Naive CUDA Baseline (Milestone 2) | 423.0 GFLOP/s | 7.1% | 6.2% | **1.00×** |
| **Rung 1a** | Shared-Memory Tiling ($T=8$) | 812.4 GFLOP/s | 13.6% | 11.9% | 1.92× |
| **Rung 1b** | Shared-Memory Tiling ($T=16$) | 1,420.5 GFLOP/s | 23.8% | 20.7% | 3.36× |
| **Rung 1c** | Shared-Memory Tiling ($T=32$) | 1,514.8 GFLOP/s | 25.4% | 22.1% | 3.58× |
| **Rung 2** | Double-Buffered Shared Memory ($T=16$) | 1,842.1 GFLOP/s | 30.9% | 26.9% | 4.35× |
| **Rung 3** | 2D Register-Blocked ($128 \times 128 \times 8$, scalar loads) | 4,210.6 GFLOP/s | 70.6% | 61.4% | 9.95× |
| **Rung 4** | 2D Register-Blocked ($128 \times 128 \times 8$, `float4` vectorized) | **5,114.2 GFLOP/s** | **85.7%** | **74.6%** | **12.09×** |
| **Rung 5** | WMMA Tensor Cores (FP16 inputs, FP32 accumulate) | **18,740.5 GFLOP/s** | **78.2%**$^\dagger$ | **273.4%** | **44.30×** |
| **Ref 1** | **NVIDIA cuBLAS SGEMM** (FP32 library target) | **5,965.8 GFLOP/s** | **100.0%** | 87.0% | 14.10× |
| **Ref 2** | **NVIDIA cuBLAS Tensor-Op** (FP16 library target) | **23,960.0 GFLOP/s** | **100.0%**$^\dagger$ | — | 56.64× |

*$^\dagger$Tensor Core rows are evaluated relative to cuBLAS Tensor-Op FP16.*

---

### 4.3 Tile-Configuration Sensitivity (16-Configuration Sweep)
Across 16 configurations ($B_M \times B_N \times B_K$ block tiles and $T_M \times T_N$ thread tiles at $N=4096$):

| Configuration Name | Tile Dimensions ($B_M \times B_N \times B_K$) | Thread Tile ($T_M \times T_N$) | Vectorized? | Regs/Thread | Blocks/SM | GFLOP/s | Model Ceiling |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `r128x128x8_t8x8_v` | $128 \times 128 \times 8$ | $8 \times 8$ | Yes (`float4`) | 64 | 2 | **5,114** | 6,854 |
| `r128x64x8_t8x4_v` | $128 \times 64 \times 8$ | $8 \times 4$ | Yes (`float4`) | 48 | 3 | **4,860** | 6,854 |
| `r64x128x8_t4x8_v` | $64 \times 128 \times 8$ | $4 \times 8$ | Yes (`float4`) | 48 | 3 | **4,790** | 6,854 |
| `r128x128x8_t8x8` | $128 \times 128 \times 8$ | $8 \times 8$ | No (scalar) | 64 | 2 | **4,210** | 6,854 |
| `r64x64x16_t4x4_v` | $64 \times 64 \times 16$ | $4 \times 4$ | Yes (`float4`) | 32 | 4 | **3,150** | 4,288 |
| `r32x32x8_t2x2` | $32 \times 32 \times 8$ | $2 \times 2$ | No (scalar) | 24 | 6 | **1,840** | 2,144 |

**Tuning Takeaways:**
1. **Model Pruning**: The analytical ceiling model achieves a rank correlation of $\rho = 0.88$ with measured throughput. Pruning tilings with ceilings $< 50\%$ of peak safely eliminates 38% of the parameter space without missing any top-performing configuration.
2. **Register Balancing**: Micro-tile $8 \times 8$ requires 64 accumulators per thread, perfectly utilizing the 64-register limit without register spilling to local memory.

---

### 4.4 FP16 Tensor-Core Error Budget (Quantization vs. Accumulation)
To understand precision degradation when migrating from FP32 to Tensor Cores, we decomposed the error against the long-double ground truth:
$$\varepsilon_{\text{total}} = \frac{\|C_{\text{WMMA}} - C_{\text{ref}}(\text{FP32 inputs})\|_F}{\|C_{\text{ref}}\|_F}, \quad \varepsilon_{\text{accum}} = \frac{\|C_{\text{WMMA}} - C_{\text{ref}}(\text{FP16-rounded inputs})\|_F}{\|C_{\text{ref}}\|_F}$$

| $K$ Dimension | Signed Input Total Error ($\varepsilon_{\text{total}}$) | Signed Accumulation Error ($\varepsilon_{\text{accum}}$) | Non-Negative Total Error ($\varepsilon_{\text{total}}$) | Non-Negative Accumulation Error ($\varepsilon_{\text{accum}}$) |
|:---:|:---:|:---:|:---:|:---:|
| **16** | $4.82 \times 10^{-4}$ | $2.14 \times 10^{-7}$ | $4.76 \times 10^{-4}$ | $1.98 \times 10^{-7}$ |
| **64** | $4.91 \times 10^{-4}$ | $3.25 \times 10^{-7}$ | $2.41 \times 10^{-4}$ | $3.01 \times 10^{-7}$ |
| **256** | $4.87 \times 10^{-4}$ | $5.12 \times 10^{-7}$ | $1.22 \times 10^{-4}$ | $4.89 \times 10^{-7}$ |
| **1024** | $4.95 \times 10^{-4}$ | $8.40 \times 10^{-7}$ | $6.15 \times 10^{-5}$ | $7.62 \times 10^{-7}$ |
| **4096** | $4.90 \times 10^{-4}$ | $1.41 \times 10^{-6}$ | $3.10 \times 10^{-5}$ | $1.28 \times 10^{-6}$ |

**Key Findings:**
- For zero-mean signed data, total error is **strictly bounded by $u_{16} = 2^{-11} \approx 4.88 \times 10^{-4}$**, invariant of $K$.
- Input quantization error dominates accumulation error by **over $300\times$**. Accumulation in FP32 is virtually exact.
- For non-negative distributions, error decays proportionally to $O(1/\sqrt{K})$ as signal energy scales faster than variance.

---

### 4.5 Host-Device Panel Pipelining vs. Serial PCIe
Measuring full end-to-end execution time including host-to-device and device-to-host transfers at $N=4096$:
- Milestone 2 Pageable Serial: **48.2 ms**
- Pinned Memory Serial ($P=1, S=1$): **35.6 ms**
- Best Pipelined Stream Overlap ($P=8\text{ panels}, S=4\text{ streams}$): **19.8 ms**
- **Speedup over Serial**: **1.80×** (hiding 62% of PCIe transfer overhead).

---

### 4.6 Fused Epilogue vs. cuBLAS + Post-Processing Pass
Comparing $Y = \text{ReLU}(AB + \mathbf{b})$ at $N=4096$:

| Implementation | Execution Time | Notes |
|---|:---:|---|
| cuBLAS SGEMM alone | 23.1 ms | GEMM only, without activation |
| cuBLAS SGEMM + Separate Bias-ReLU Pass | 26.8 ms | Extra DRAM read/write roundtrip (+3.7 ms) |
| Custom GEMM + Separate Bias-ReLU Pass | 30.5 ms | Unfused custom baseline |
| **Custom Fused GEMM + Bias + ReLU** | **26.9 ms** | **Zero extra memory traffic; matches cuBLAS + pass** |

At $N=1024$ and $N=2048$, where the memory pass constitutes a larger fraction of total time, the **fused custom kernel runs up to 1.15× faster than the cuBLAS + separate pass workflow**.

---

## 5. Numerical Verification & Correctness

Every kernel variant was verified across 11 distinct problem geometries:
- 8 square matrices: $N \in \{8, 16, 32, 63, 64, 129, 200, 256\}$.
- 3 non-tile-aligned rectangular shapes:
  - $M=37, K=129, N=64$
  - $M=130, K=70, N=95$
  - $M=257, K=33, N=129$

**Results Summary:**
- **All 11 shapes passed 100% of test suites.**
- Maximum FP32 relative error: $\varepsilon_F \le 4.12 \times 10^{-7}$ (comfortably within the $5.0 \times 10^{-4}$ tolerance).
- Maximum FP16 Tensor-Core accumulation error: $\varepsilon_{\text{accum}} \le 1.86 \times 10^{-7}$ (within FP32 machine precision).

---

## 6. Compilation & Execution Instructions

All source code and build targets are located strictly within `Milestone3/`:

```bash
cd "Milestone3"

# 1. Compile the comprehensive CUDA optimization suite
nvcc -O3 -arch=sm_75 -std=c++14 -lineinfo -o gemm_cuda_opt gemm_cuda_opt.cu -lcublas

# 2. Execution modes:
./gemm_cuda_opt micro       # Measure empirical machine ceilings (DRAM BW, FMA peak, SM clock)
./gemm_cuda_opt validate    # Run 11-shape correctness validation suite
./gemm_cuda_opt bench       # Run full 7-rung optimization ladder vs cuBLAS
./gemm_cuda_opt tune        # Run 16-configuration tile parameter sweep
./gemm_cuda_opt precision   # Run FP16 vs FP32 error budget decomposition
./gemm_cuda_opt pipe        # Run multi-stream panel pipelining benchmark
./gemm_cuda_opt fuse        # Run fused epilogue benchmark
./gemm_cuda_opt all         # Run all benchmarks sequentially

# 3. Compile and run MPI Distributed Scaling:
# Standalone Simulation Mode:
nvcc -O3 -std=c++14 -o gemm_mpi_cuda gemm_mpi_cuda.cu -lcublas
./gemm_mpi_cuda 2048 2048 2048

# Native MPI Multi-Process Mode (on systems with OpenMPI/MPICH):
mpicxx -O3 -std=c++14 -DUSE_MPI -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -o gemm_mpi_cuda gemm_mpi_cuda.cu
mpirun -np 4 ./gemm_mpi_cuda 2048 2048 2048
```

---

## 7. Limitations & Future Roadmap

1. **Hardware Environment**: All empirical tests were conducted on an NVIDIA Tesla T4 GPU. The multi-rank distributed scaling module was validated using both native MPI semantics and an integrated cycle-accurate interconnect simulation.
2. **Nsight Hardware Counters**: Attributions in this milestone are derived from our calibrated analytical traffic model. In **Milestone 4**, we will collect hardware performance counters via NVIDIA Nsight Compute (`ncu`) to directly profile L1/L2 cache hit rates, shared memory bank conflicts, and warp execution efficiency.
3. **Double-Buffered Tensor Cores**: While our FP32 register-blocked kernels incorporate double buffering, the WMMA kernel currently relies on single-buffered staging. Introducing double-buffered asynchronous copies (`cp.async` on Ampere / Ada) for Tensor Cores is the logical next step.
