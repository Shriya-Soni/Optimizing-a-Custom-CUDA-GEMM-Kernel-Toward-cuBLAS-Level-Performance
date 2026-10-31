# Milestone 3: Advanced Optimizations & Distributed Scaling

This directory contains the complete source code, technical report, benchmarking harnesses, and distributed scaling implementations for **Milestone 3** of the project *"Optimizing a Custom CUDA GEMM Kernel Toward cuBLAS-Level Performance"*.

---

## Directory Contents

- **`gemm_cuda_opt.cu`**: The primary CUDA implementation file containing:
  - **S0 / S0c**: Milestone 2 naive baseline and uncoalesced negative control.
  - **S1**: Flexible shared-memory tiled GEMM ($T \in \{8, 16, 32\}$).
  - **S1db**: Double-buffered shared-memory tiled GEMM (communication hiding / 2-stage software pipeline).
  - **S2 / S3**: 2-D register-blocked GEMM ($128 \times 128 \times 8$ block tile, $8 \times 8$ micro-tile) with transposed shared memory layout and `float4` vectorization.
  - **S4**: Tensor Core GEMM via NVIDIA WMMA API ($16 \times 16 \times 16$ fragments, FP16 input, FP32 accumulate).
  - **Fused Epilogue**: Fused GEMM + bias + ReLU activation, bypassing intermediate DRAM read/write roundtrips.
  - **Microbenchmarks**: Empirical DRAM copy bandwidth and FP32 FMA peak ceilings.
  - **Harnesses**: Correctness validation across 11 regular and irregular shapes, 16-configuration tile sweep, FP16 error budget decomposition, and multi-stream panel pipelining.

- **`gemm_mpi_cuda.cu`**: Dedicated Hybrid MPI + CUDA distributed GEMM scaling module:
  - Implements 1D block-row domain decomposition across $P$ ranks.
  - Supports non-blocking communication (`MPI_Ibcast`, `MPI_Isend`, `MPI_Irecv`) to overlap network and PCIe transfers with GPU execution.
  - Includes an integrated standalone interconnect simulation mode for environments without native MPI installations.

- **`MILESTONE3_REPORT.md`**: Complete, comprehensive technical engineering report containing:
  - Initial codebase analysis and re-evaluation of the Milestone 2 bottleneck.
  - Three-resource analytical ceiling model ($P \le \min(P_{\text{FMA}}, B_{\text{smem}}/\beta_{\text{smem}}, B_{\text{dram}}/\beta_{\text{glob}})$).
  - Mathematical derivations for operational intensity.
  - Complete empirical benchmark tables (CPU vs. Naive vs. Smem vs. Double-Buffer vs. Vectorized vs. Tensor Cores vs. cuBLAS).
  - FP16 error budget analysis (quantization vs. accumulation error).
  - Validation results across 11 problem shapes.

- **`Milestone_3.ipynb`**: Jupyter Notebook designed for execution in Google Colab (Tesla T4 GPU runtime).

- **`Milestone_3.tex`**: LaTeX conference report (IEEEtran format) referencing the generated tables, figures, and macros.

---

## Compilation and Build Instructions

### 1. Main Optimization Suite (`gemm_cuda_opt.cu`)
Requires CUDA Toolkit with `nvcc` and `-lcublas`:

```bash
# Compile for Turing (Tesla T4, RTX 2080):
nvcc -O3 -arch=sm_75 -std=c++14 -lineinfo -o gemm_cuda_opt gemm_cuda_opt.cu -lcublas

# Compile for Ampere (A100, RTX 3080):
# nvcc -O3 -arch=sm_80 -std=c++14 -lineinfo -o gemm_cuda_opt gemm_cuda_opt.cu -lcublas
```

### 2. MPI Hybrid Scaling Module (`gemm_mpi_cuda.cu`)

#### Standalone Simulation Mode (No MPI required):
```bash
nvcc -O3 -std=c++14 -o gemm_mpi_cuda gemm_mpi_cuda.cu -lcublas
```

#### Native MPI Multi-Process Mode (with OpenMPI/MPICH installed):
```bash
# Using mpicxx wrapper:
mpicxx -O3 -std=c++14 -DUSE_MPI -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -lcublas -o gemm_mpi_cuda gemm_mpi_cuda.cu

# Or using nvcc with mpicxx backend:
nvcc -O3 -std=c++14 -DUSE_MPI -ccbin mpicxx -o gemm_mpi_cuda gemm_mpi_cuda.cu -lcublas
```

---

## Execution Guide

### Running `gemm_cuda_opt`
The executable accepts command-line arguments to trigger individual benchmark modes:

```bash
./gemm_cuda_opt micro       # Measure device ceilings (DRAM copy BW, FP32 FMA peak, observed clock)
./gemm_cuda_opt validate    # Run correctness validation across all 11 shapes
./gemm_cuda_opt bench       # Run full ablation ladder vs cuBLAS (N=256 to N=8192)
./gemm_cuda_opt tune        # Run 16-configuration tile sensitivity sweep
./gemm_cuda_opt precision   # Run FP16 vs FP32 numerical error budget study
./gemm_cuda_opt pipe        # Run multi-stream host-device panel pipelining
./gemm_cuda_opt fuse        # Run fused epilogue (ReLU(AB + bias)) study
./gemm_cuda_opt all         # Run all benchmark suites sequentially (default)
```

### Running `gemm_mpi_cuda`

```bash
# Standalone Simulation:
./gemm_mpi_cuda 2048 2048 2048

# Native MPI Execution (e.g., 4 processes):
mpirun -np 4 ./gemm_mpi_cuda 2048 2048 2048
```

---

## Summary of Results

- **Naive Baseline (M2)**: 423 GFLOP/s (demands 1.69 TB/s of operand traffic, proving L1/L2 LSU throughput limitation).
- **Shared-Memory Tiling ($T=16$)**: 1,420 GFLOP/s ($3.36\times$ over naive).
- **Double-Buffered Shared Memory ($T=16$)**: 1,842 GFLOP/s ($4.35\times$ over naive, successfully hiding memory latency).
- **2D Register-Blocked (`float4` Vectorized)**: **5,114 GFLOP/s** (**85.7% of cuBLAS SGEMM**, $12.09\times$ over naive).
- **Tensor Cores (WMMA FP16)**: **18,740 GFLOP/s** ($44.3\times$ over naive, 78.2% of cuBLAS Tensor-Op).
- **Fused Epilogue**: Fused ReLU + bias eliminates DRAM read/write passes, running up to **1.15× faster than cuBLAS + separate elementwise pass**.
- **Numerical Correctness**: 100% PASS across 11 matrix shapes with Frobenius relative error $\varepsilon_F \le 4.12 \times 10^{-7}$.
