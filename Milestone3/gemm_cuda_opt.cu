/* ============================================================
 * gemm_cuda_opt.cu
 *
 * Milestone 3 deliverable: "Optimizing a Custom CUDA GEMM Kernel
 * Toward cuBLAS-Level Performance" -- Advanced Optimizations &
 * Pipelined (Distributed-Style) Scaling.
 *
 * Milestone 2 left us with a naive one-thread-per-element kernel
 * (423 GFLOP/s on a Tesla T4) and a *hypothesis* that it was
 * limited by off-chip bandwidth. Milestone 3 does three things:
 *
 *   (A) builds the optimization ladder
 *         naive -> smem tiling -> 2-D register blocking
 *               -> float4-vectorized loads -> Tensor Cores (WMMA)
 *       and compares every rung against cuBLAS (SGEMM + FP16 TC);
 *
 *   (B) *measures* the machine (DRAM copy BW, achievable FP32 FMA
 *       peak, observed SM clock) so that an analytical per-level
 *       traffic model can be confronted with real numbers instead
 *       of datasheet values (the notebook does the analysis);
 *
 *   (C) studies four questions that go beyond "make it fast":
 *         - tile-configuration sensitivity (16 configs, regs/spills/
 *           occupancy recorded)                         [mode: tune]
 *         - FP16-Tensor-Core error budget (quantization vs.
 *           accumulation error, vs. K, vs. data dist.)  [mode: precision]
 *         - panel-pipelined streaming GEMM as a single-node proxy
 *           for tile-level compute/transfer overlap     [mode: pipe]
 *         - a fused GEMM+bias+ReLU epilogue, the structural reason
 *           a custom kernel can beat "cuBLAS + extra pass" [mode: fuse]
 *
 * Build (the -lcublas is new compared with Milestone 2):
 *   nvcc -O3 -arch=sm_75 -std=c++14 -lineinfo \
 *        -o gemm_cuda_opt gemm_cuda_opt.cu -lcublas
 *   (sm_75 = Turing/T4; use sm_80/sm_86/sm_89 for other GPUs)
 *
 * Usage:
 *   ./gemm_cuda_opt micro      # device ceilings (DRAM BW, FMA peak, clock)
 *   ./gemm_cuda_opt validate   # correctness of every kernel
 *   ./gemm_cuda_opt bench      # ablation ladder vs cuBLAS
 *   ./gemm_cuda_opt tune       # tile-configuration sweep
 *   ./gemm_cuda_opt precision  # FP16/FP32 error studies
 *   ./gemm_cuda_opt pipe       # panel-pipelined streaming GEMM
 *   ./gemm_cuda_opt fuse       # fused epilogue study
 *   ./gemm_cuda_opt all        # everything (default)
 *
 * Output CSVs: results_m3_{micro,validation,ablation,tuning,
 *              precision,pipeline,fusion}.csv
 * ============================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda;

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err__));                             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

#define CUBLAS_CHECK(call)                                                  \
    do {                                                                    \
        cublasStatus_t st__ = (call);                                       \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                \
            fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)st__,        \
                    __FILE__, __LINE__);                                    \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static cublasHandle_t g_cublas;

/* ============================================================
 *  PART 1 -- KERNELS
 * ============================================================ */

/* Epilogue shared by all FP32 kernels. beta == 0 never reads C, so an
 * uninitialised C cannot inject NaN (BLAS semantics). */
__device__ __forceinline__ float gemm_epilogue(float acc, float alpha,
                                               float beta, float c_old) {
    return alpha * acc + (beta != 0.0f ? beta * c_old : 0.0f);
}

/* ---- S0: Milestone-2 naive kernel (verbatim mapping) --------------- */
/* threadIdx.x -> column  => B loads are coalesced, A loads broadcast.   */
__global__ void gemm_naive_kernel(int M, int N, int K, float alpha,
                                  const float *A, const float *B,
                                  float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] =
            gemm_epilogue(acc, alpha, beta, C[(size_t)row * N + col]);
    }
}

/* ---- S0c: negative control. Same arithmetic, but threadIdx.x -> ROW,
 * so a warp reads A with stride K and B with a broadcast: this is the
 * "uncoalesced" access pattern and quantifies how much coalescing the
 * Milestone-2 kernel already enjoyed "for free". -------------------- */
__global__ void gemm_naive_uncoal_kernel(int M, int N, int K, float alpha,
                                         const float *A, const float *B,
                                         float beta, float *C) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++)
            acc += A[(size_t)row * K + k] * B[(size_t)k * N + col];
        C[(size_t)row * N + col] =
            gemm_epilogue(acc, alpha, beta, C[(size_t)row * N + col]);
    }
}

/* ---- S1: classic shared-memory tiling (one output / thread) -------- */
template <int T>
__global__ void gemm_smem_kernel(int M, int N, int K, float alpha,
                                 const float *A, const float *B,
                                 float beta, float *C) {
    __shared__ float As[T][T];
    __shared__ float Bs[T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.0f;
        Bs[ty][tx] = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.0f;
        __syncthreads();
#pragma unroll
        for (int k = 0; k < T; k++) acc += As[ty][k] * Bs[k][tx];
        __syncthreads();
    }
    if (row < M && col < N)
        C[(size_t)row * N + col] =
            gemm_epilogue(acc, alpha, beta, C[(size_t)row * N + col]);
}

/* ---- S1db: Double-buffered shared-memory GEMM (Communication Hiding) ----
 * Overlaps global-memory tile loads with shared-memory computation via
 * a 2-stage software pipeline using double buffers As[2][T][T] and Bs[2][T][T].
 * ----------------------------------------------------------------------- */
template <int T>
__global__ void gemm_smem_double_buffer_kernel(int M, int N, int K, float alpha,
                                               const float *__restrict__ A,
                                               const float *__restrict__ B,
                                               float beta, float *__restrict__ C) {
    __shared__ float As[2][T][T];
    __shared__ float Bs[2][T][T];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int row = blockIdx.y * T + ty;
    const int col = blockIdx.x * T + tx;
    float acc = 0.0f;

    // Prologue: fetch tile 0 into buffer 0
    As[0][ty][tx] = (row < M && tx < K) ? A[(size_t)row * K + tx] : 0.0f;
    Bs[0][ty][tx] = (ty < K && col < N) ? B[(size_t)ty * N + col] : 0.0f;
    __syncthreads();

    int read_b = 0;
    for (int k0 = T; k0 < K; k0 += T) {
        int write_b = 1 - read_b;
        // Prefetch next tile into register staging variables
        float next_a = (row < M && k0 + tx < K) ? A[(size_t)row * K + k0 + tx] : 0.0f;
        float next_b = (k0 + ty < K && col < N) ? B[(size_t)(k0 + ty) * N + col] : 0.0f;

        // Compute on current tile from read_b
#pragma unroll
        for (int k = 0; k < T; k++) {
            acc += As[read_b][ty][k] * Bs[read_b][k][tx];
        }

        // Store prefetched tile into shared memory buffer write_b
        As[write_b][ty][tx] = next_a;
        Bs[write_b][ty][tx] = next_b;
        __syncthreads();

        read_b = write_b;
    }

    // Epilogue: compute last remaining tile
#pragma unroll
    for (int k = 0; k < T; k++) {
        acc += As[read_b][ty][k] * Bs[read_b][k][tx];
    }

    if (row < M && col < N) {
        C[(size_t)row * N + col] =
            gemm_epilogue(acc, alpha, beta, C[(size_t)row * N + col]);
    }
}

/* ---- S2/S3: 2-D register-blocked kernel ------------------------------
 *  Block tile BM x BN, K-slab BK, per-thread micro-tile TM x TN held in
 *  registers.  A is stored TRANSPOSED in shared memory (As[k][m]) so the
 *  inner loop reads TM contiguous floats.
 *  VEC=true : float4 global loads (needs K%4==0 && N%4==0, checked on host)
 *  EPI=1    : fused epilogue  v = relu(alpha*AB + beta*C + bias[col])
 *  Any M,N,K is handled (zero-padded tile loads, guarded stores).
 * -------------------------------------------------------------------- */
template <int BM, int BN, int BK, int TM, int TN, int EPI, bool VEC>
__global__ void __launch_bounds__((BM / TM) * (BN / TN))
gemm_reg2d_kernel(int M, int N, int K, float alpha,
                  const float *__restrict__ A, const float *__restrict__ B,
                  float beta, float *__restrict__ C,
                  const float *__restrict__ bias) {
    constexpr int NT  = (BM / TM) * (BN / TN);
    constexpr int TPR = BN / TN;               /* threads per tile row */
    __shared__ __align__(16) float As[BK * BM];
    __shared__ __align__(16) float Bs[BK * BN];

    const int tid  = threadIdx.x;
    const int tr   = tid / TPR;
    const int tc   = tid % TPR;
    const int row0 = blockIdx.y * BM;
    const int col0 = blockIdx.x * BN;

    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; i++)
#pragma unroll
        for (int j = 0; j < TN; j++) acc[i][j] = 0.0f;
    float ra[TM], rb[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        if (VEC) {
            for (int idx = tid; idx < BM * BK / 4; idx += NT) {
                int r = idx / (BK / 4), c4 = idx % (BK / 4);
                int gr = row0 + r, gk = k0 + c4 * 4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (gr < M && gk < K)
                    v = *reinterpret_cast<const float4 *>(&A[(size_t)gr * K + gk]);
                As[(c4 * 4 + 0) * BM + r] = v.x;
                As[(c4 * 4 + 1) * BM + r] = v.y;
                As[(c4 * 4 + 2) * BM + r] = v.z;
                As[(c4 * 4 + 3) * BM + r] = v.w;
            }
            for (int idx = tid; idx < BK * BN / 4; idx += NT) {
                int r = idx / (BN / 4), c4 = idx % (BN / 4);
                int gk = k0 + r, gc = col0 + c4 * 4;
                float4 v = make_float4(0.f, 0.f, 0.f, 0.f);
                if (gk < K && gc < N)
                    v = *reinterpret_cast<const float4 *>(&B[(size_t)gk * N + gc]);
                *reinterpret_cast<float4 *>(&Bs[r * BN + c4 * 4]) = v;
            }
        } else {
            for (int idx = tid; idx < BM * BK; idx += NT) {
                int r = idx / BK, c = idx % BK;
                int gr = row0 + r, gk = k0 + c;
                As[c * BM + r] = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : 0.0f;
            }
            for (int idx = tid; idx < BK * BN; idx += NT) {
                int r = idx / BN, c = idx % BN;
                int gk = k0 + r, gc = col0 + c;
                Bs[r * BN + c] = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : 0.0f;
            }
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < BK; kk++) {
#pragma unroll
            for (int i = 0; i < TM; i++) ra[i] = As[kk * BM + tr * TM + i];
#pragma unroll
            for (int j = 0; j < TN; j++) rb[j] = Bs[kk * BN + tc * TN + j];
#pragma unroll
            for (int i = 0; i < TM; i++)
#pragma unroll
                for (int j = 0; j < TN; j++)
                    acc[i][j] = fmaf(ra[i], rb[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < TM; i++) {
#pragma unroll
        for (int j = 0; j < TN; j++) {
            int gr = row0 + tr * TM + i, gc = col0 + tc * TN + j;
            if (gr < M && gc < N) {
                float c_old = (beta != 0.0f) ? C[(size_t)gr * N + gc] : 0.0f;
                float v = gemm_epilogue(acc[i][j], alpha, beta, c_old);
                if (EPI == 1) v = fmaxf(v + bias[gc], 0.0f);
                C[(size_t)gr * N + gc] = v;
            }
        }
    }
}

/* ---- S4: Tensor-Core kernel (WMMA, FP16 inputs, FP32 accumulate) -----
 *  64x64 block tile, BK=32, 4 warps (2x2), each warp owns 2x2 16x16x16
 *  fragments.  Smem rows are padded (+8 halves / +4 floats) against bank
 *  conflicts; all fragment pointers are 32-byte aligned (see comments in
 *  the report).  Any M,N,K supported (zero-padded loads, guarded stores).
 * -------------------------------------------------------------------- */
#define WM_BM 64
#define WM_BN 64
#define WM_BK 32
#define WM_LDA (WM_BK + 8)
#define WM_LDB (WM_BN + 8)
#define WM_LDC (WM_BN + 4)

__global__ void __launch_bounds__(128)
gemm_wmma_kernel(int M, int N, int K, float alpha,
                 const half *__restrict__ A, const half *__restrict__ B,
                 float beta, float *__restrict__ C) {
    __shared__ __align__(128) half  As[WM_BM * WM_LDA];
    __shared__ __align__(128) half  Bs[WM_BK * WM_LDB];
    __shared__ __align__(128) float Cs[WM_BM * WM_LDC];

    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int wy = warp >> 1, wx = warp & 1;
    const int row0 = blockIdx.y * WM_BM, col0 = blockIdx.x * WM_BN;
    const half zero = __float2half(0.0f);

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(acc[i][j], 0.0f);

    for (int k0 = 0; k0 < K; k0 += WM_BK) {
        for (int idx = tid; idx < WM_BM * WM_BK; idx += 128) {
            int r = idx / WM_BK, c = idx % WM_BK;
            int gr = row0 + r, gk = k0 + c;
            As[r * WM_LDA + c] = (gr < M && gk < K) ? A[(size_t)gr * K + gk] : zero;
        }
        for (int idx = tid; idx < WM_BK * WM_BN; idx += 128) {
            int r = idx / WM_BN, c = idx % WM_BN;
            int gk = k0 + r, gc = col0 + c;
            Bs[r * WM_LDB + c] = (gk < K && gc < N) ? B[(size_t)gk * N + gc] : zero;
        }
        __syncthreads();

#pragma unroll
        for (int kk = 0; kk < WM_BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> fb[2];
#pragma unroll
            for (int i = 0; i < 2; i++)
                wmma::load_matrix_sync(fa[i], &As[(wy * 32 + i * 16) * WM_LDA + kk], WM_LDA);
#pragma unroll
            for (int j = 0; j < 2; j++)
                wmma::load_matrix_sync(fb[j], &Bs[kk * WM_LDB + wx * 32 + j * 16], WM_LDB);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(&Cs[(wy * 32 + i * 16) * WM_LDC + wx * 32 + j * 16],
                                    acc[i][j], WM_LDC, wmma::mem_row_major);
    __syncthreads();

    for (int idx = tid; idx < WM_BM * WM_BN; idx += 128) {
        int r = idx / WM_BN, c = idx % WM_BN;
        int gr = row0 + r, gc = col0 + c;
        if (gr < M && gc < N) {
            float c_old = (beta != 0.0f) ? C[(size_t)gr * N + gc] : 0.0f;
            C[(size_t)gr * N + gc] = gemm_epilogue(Cs[r * WM_LDC + c], alpha, beta, c_old);
        }
    }
}

/* ---- helpers: FP32 -> FP16 conversion, separate bias+ReLU pass ------ */
__global__ void f2h_kernel(const float *in, half *out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(in[i]);
}

/* The "extra pass" that a library GEMM forces on us: reads and writes the
 * whole of C once more.  Requires N % 4 == 0. */
__global__ void bias_relu_kernel(float4 *C, const float4 *bias4, int N4, size_t total4) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < total4) {
        float4 c = C[i];
        float4 b = bias4[i % (size_t)N4];
        c.x = fmaxf(c.x + b.x, 0.f);
        c.y = fmaxf(c.y + b.y, 0.f);
        c.z = fmaxf(c.z + b.z, 0.f);
        c.w = fmaxf(c.w + b.w, 0.f);
        C[i] = c;
    }
}

/* ---- microbenchmark kernels ---------------------------------------- */
__global__ void copy_kernel(const float4 *__restrict__ in, float4 *__restrict__ out, size_t n) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) out[i] = in[i];
}

/* 8 independent FMA chains per thread -> saturates the FP32 pipes.
 * Also reports the SM clock actually *observed* during the run
 * (clock64 cycles / globaltimer ns), exposing boost/throttle behaviour. */
__global__ void fma_peak_kernel(float *out, int iters, long long *res) {
    float a0 = threadIdx.x * 1e-3f + 1.0f, a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
    const float b = 1.0000001f, c = 1e-7f;
    long long t0 = clock64();
    unsigned long long g0, g1;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
    for (int i = 0; i < iters; i++) {
        a0 = fmaf(a0, b, c); a1 = fmaf(a1, b, c); a2 = fmaf(a2, b, c); a3 = fmaf(a3, b, c);
        a4 = fmaf(a4, b, c); a5 = fmaf(a5, b, c); a6 = fmaf(a6, b, c); a7 = fmaf(a7, b, c);
    }
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
    long long t1 = clock64();
    out[blockIdx.x * blockDim.x + threadIdx.x] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        res[0] = t1 - t0;
        res[1] = (long long)(g1 - g0);
    }
}

/* ============================================================
 *  PART 2 -- LAUNCH PLUMBING (config table, device buffers, dispatch)
 * ============================================================ */

template <int BM, int BN, int BK, int TM, int TN, int EPI, bool VEC>
static void launch_reg2d(int M, int N, int K, float alpha, const float *A,
                         const float *B, float beta, float *C,
                         const float *bias, cudaStream_t s) {
    dim3 block((BM / TM) * (BN / TN));
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    gemm_reg2d_kernel<BM, BN, BK, TM, TN, EPI, VEC>
        <<<grid, block, 0, s>>>(M, N, K, alpha, A, B, beta, C, bias);
}

template <int BM, int BN, int BK, int TM, int TN, bool VEC>
static void query_reg2d(cudaFuncAttributes *a, int *blocks_per_sm) {
    auto k = gemm_reg2d_kernel<BM, BN, BK, TM, TN, 0, VEC>;
    CUDA_CHECK(cudaFuncGetAttributes(a, k));
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        blocks_per_sm, k, (BM / TM) * (BN / TN), 0));
}

typedef void (*LaunchFn)(int, int, int, float, const float *, const float *,
                         float, float *, const float *, cudaStream_t);
typedef void (*QueryFn)(cudaFuncAttributes *, int *);

struct Cfg {
    const char *name;
    int BM, BN, BK, TM, TN;
    bool vec;
    LaunchFn fn;
    QueryFn q;
};

/* Every (BM,BN,BK,TM,TN) is instantiated as scalar-load and float4-load
 * variants.  Thread count (BM/TM)*(BN/TN) is 256 except where noted. */
#define CFG(BM, BN, BK, TM, TN)                                                         \
    {"r" #BM "x" #BN "x" #BK "_t" #TM "x" #TN, BM, BN, BK, TM, TN, false,               \
     launch_reg2d<BM, BN, BK, TM, TN, 0, false>, query_reg2d<BM, BN, BK, TM, TN, false>},\
    {"r" #BM "x" #BN "x" #BK "_t" #TM "x" #TN "_v", BM, BN, BK, TM, TN, true,           \
     launch_reg2d<BM, BN, BK, TM, TN, 0, true>, query_reg2d<BM, BN, BK, TM, TN, true>}

static const Cfg g_cfgs[] = {
    CFG(32, 32, 8, 2, 2),     /*  256 thr,   4 accumulators                */
    CFG(64, 64, 8, 4, 4),     /*  256 thr,  16 accumulators                */
    CFG(64, 64, 16, 4, 4),    /*  256 thr,  deeper K slab                   */
    CFG(128, 64, 8, 8, 4),    /*  256 thr,  32 accumulators                */
    CFG(64, 128, 8, 4, 8),    /*  256 thr,  32 accumulators                */
    CFG(128, 128, 8, 8, 8),   /*  256 thr,  64 accumulators (Ampere-style default) */
    CFG(128, 128, 16, 8, 8),  /*  256 thr,  deeper K slab                   */
    CFG(128, 128, 8, 8, 4)    /*  512 thr,  32 accumulators                */
};
static const int NCFG = (int)(sizeof(g_cfgs) / sizeof(g_cfgs[0]));

static int find_cfg(int BM, int BN, int BK, int TM, int TN, bool vec) {
    for (int i = 0; i < NCFG; i++)
        if (g_cfgs[i].BM == BM && g_cfgs[i].BN == BN && g_cfgs[i].BK == BK &&
            g_cfgs[i].TM == TM && g_cfgs[i].TN == TN && g_cfgs[i].vec == vec)
            return i;
    fprintf(stderr, "config not found\n");
    exit(1);
}

struct Dev {
    int M = 0, N = 0, K = 0;
    float *dA = nullptr, *dB = nullptr, *dC = nullptr, *dBias = nullptr;
    half *dAh = nullptr, *dBh = nullptr;
};

static void dev_alloc(Dev &d, int M, int N, int K) {
    d.M = M; d.N = N; d.K = K;
    CUDA_CHECK(cudaMalloc(&d.dA, (size_t)M * K * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dB, (size_t)K * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dC, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dBias, (size_t)N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d.dAh, (size_t)M * K * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d.dBh, (size_t)K * N * sizeof(half)));
    CUDA_CHECK(cudaMemset(d.dC, 0, (size_t)M * N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d.dBias, 0, (size_t)N * sizeof(float)));
}
static void dev_free(Dev &d) {
    cudaFree(d.dA); cudaFree(d.dB); cudaFree(d.dC); cudaFree(d.dBias);
    cudaFree(d.dAh); cudaFree(d.dBh);
    d = Dev();
}
static void dev_convert_half(Dev &d) {
    size_t nA = (size_t)d.M * d.K, nB = (size_t)d.K * d.N;
    f2h_kernel<<<(unsigned)((nA + 255) / 256), 256>>>(d.dA, d.dAh, nA);
    f2h_kernel<<<(unsigned)((nB + 255) / 256), 256>>>(d.dB, d.dBh, nB);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

enum Kind { K_NAIVE, K_UNCOAL, K_SMEM, K_SMEM8, K_SMEM16, K_SMEM32, K_SMEM_DB16, K_SMEM_DB32, K_REG, K_WMMA, K_SGEMM, K_TCGEMM };

/* Enqueue one GEMM of the requested kind. Returns false if the kernel
 * cannot legally run on this shape (float4 alignment). */
static bool launch_kind(Kind kind, int cfg, Dev &d, float alpha, float beta,
                        cudaStream_t s) {
    const int M = d.M, N = d.N, K = d.K;
    switch (kind) {
    case K_NAIVE: {
        dim3 b(16, 16), g((N + 15) / 16, (M + 15) / 16);
        gemm_naive_kernel<<<g, b, 0, s>>>(M, N, K, alpha, d.dA, d.dB, beta, d.dC);
        break;
    }
    case K_UNCOAL: {
        dim3 b(16, 16), g((M + 15) / 16, (N + 15) / 16);
        gemm_naive_uncoal_kernel<<<g, b, 0, s>>>(M, N, K, alpha, d.dA, d.dB, beta, d.dC);
        break;
    }
    case K_SMEM:
    case K_SMEM32: {
        dim3 b(32, 32), g((N + 31) / 32, (M + 31) / 32);
        gemm_smem_kernel<32><<<g, b, 0, s>>>(M, N, K, alpha, d.dA, d.dB, beta, d.dC);
        break;
    }
    case K_SMEM16: {
        dim3 b(16, 16), g((N + 15) / 16, (M + 15) / 16);
        gemm_smem_kernel<16><<<g, b, 0, s>>>(M, N, K, alpha, d.dA, d.dB, beta, d.dC);
        break;
    }
    case K_SMEM8: {
        dim3 b(8, 8), g((N + 7) / 8, (M + 7) / 8);
        gemm_smem_kernel<8><<<g, b, 0, s>>>(M, N, K, alpha, d.dA, d.dB, beta, d.dC);
        break;
    }
    case K_SMEM_DB16: {
        dim3 b(16, 16), g((N + 15) / 16, (M + 15) / 16);
        gemm_smem_double_buffer_kernel<16><<<g, b, 0, s>>>(M, N, K, alpha, d.dA, d.dB, beta, d.dC);
        break;
    }
    case K_SMEM_DB32: {
        dim3 b(32, 32), g((N + 31) / 32, (M + 31) / 32);
        gemm_smem_double_buffer_kernel<32><<<g, b, 0, s>>>(M, N, K, alpha, d.dA, d.dB, beta, d.dC);
        break;
    }
    case K_REG: {
        const Cfg &c = g_cfgs[cfg];
        if (c.vec && ((K & 3) || (N & 3))) return false;
        c.fn(M, N, K, alpha, d.dA, d.dB, beta, d.dC, d.dBias, s);
        break;
    }
    case K_WMMA: {
        dim3 b(128), g((N + WM_BN - 1) / WM_BN, (M + WM_BM - 1) / WM_BM);
        gemm_wmma_kernel<<<g, b, 0, s>>>(M, N, K, alpha, d.dAh, d.dBh, beta, d.dC);
        break;
    }
    case K_SGEMM: {
        /* Row-major C = A*B  <=>  column-major C^T = B^T * A^T */
        CUBLAS_CHECK(cublasSetStream(g_cublas, s));
        CUBLAS_CHECK(cublasSgemm(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                 &alpha, d.dB, N, d.dA, K, &beta, d.dC, N));
        break;
    }
    case K_TCGEMM: {
        CUBLAS_CHECK(cublasSetStream(g_cublas, s));
        CUBLAS_CHECK(cublasGemmEx(g_cublas, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                                  &alpha, d.dBh, CUDA_R_16F, N, d.dAh, CUDA_R_16F, K,
                                  &beta, d.dC, CUDA_R_32F, N, CUBLAS_COMPUTE_32F,
                                  CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        break;
    }
    }
    CUDA_CHECK(cudaGetLastError());
    return true;
}

/* ============================================================
 *  PART 3 -- HOST UTILITIES + REFERENCES (carried over from M1/M2)
 * ============================================================ */

static float *alloc_f(size_t n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}
static void fill_random(float *M, size_t n, unsigned int *seed) {      /* U[-1,1] */
    for (size_t i = 0; i < n; i++)
        M[i] = ((float)rand_r(seed) / (float)RAND_MAX) * 2.0f - 1.0f;
}
static void fill_positive(float *M, size_t n, unsigned int *seed) {    /* U[0,1]  */
    for (size_t i = 0; i < n; i++) M[i] = (float)rand_r(seed) / (float)RAND_MAX;
}
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}
static float q16(float x) { return __half2float(__float2half(x)); }   /* FP16 round-trip */

void ikj_gemm(int M, int N, int K, float alpha, const float *A, const float *B,
              float beta, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) C[(size_t)i * N + j] *= beta;
    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a_ik = alpha * A[(size_t)i * K + k];
            for (int j = 0; j < N; j++) C[(size_t)i * N + j] += a_ik * B[(size_t)k * N + j];
        }
}

void reference_gemm_highprec(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, const float *C_in,
                             float *C_out) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            long double acc = 0.0L;
            for (int k = 0; k < K; k++)
                acc += (long double)A[(size_t)i * K + k] * (long double)B[(size_t)k * N + j];
            long double val = (long double)alpha * acc +
                              (long double)beta * (long double)C_in[(size_t)i * N + j];
            C_out[(size_t)i * N + j] = (float)val;
        }
}

/* Same Frobenius-norm relative error as Milestones 1 and 2. */
static void cmp(const float *C, const float *R, size_t n, double *max_abs, double *fro_rel) {
    double ma = 0.0, num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double diff = fabs((double)C[i] - (double)R[i]);
        if (diff > ma) ma = diff;
        num += diff * diff;
        den += (double)R[i] * (double)R[i];
    }
    *max_abs = ma;
    *fro_rel = sqrt(num) / (sqrt(den) + 1e-30);
}

struct TimeStat { double med, mn; };

/* Timing protocol (fixes the Milestone-2 single cold launch):
 * 1 untimed warm-up -> adaptive repetitions -> median and minimum. */
template <class F>
static TimeStat time_it(F fn) {
    cudaEvent_t a, b;
    CUDA_CHECK(cudaEventCreate(&a));
    CUDA_CHECK(cudaEventCreate(&b));
    auto once = [&]() -> double {
        CUDA_CHECK(cudaEventRecord(a, 0));
        fn();
        CUDA_CHECK(cudaEventRecord(b, 0));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return (double)ms;
    };
    double t0 = once();                                  /* warm-up */
    int reps = (t0 > 500.0) ? 3 : (t0 > 50.0) ? 7 : 25;
    std::vector<double> v;
    for (int r = 0; r < reps; r++) v.push_back(once());
    std::sort(v.begin(), v.end());
    CUDA_CHECK(cudaEventDestroy(a));
    CUDA_CHECK(cudaEventDestroy(b));
    TimeStat t;
    t.med = v[v.size() / 2];
    t.mn = v[0];
    return t;
}
static double median_d(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
static double gflops_of(double flops, double ms) { return flops / (ms * 1e6); }

/* ============================================================
 *  PART 4 -- MODES
 * ============================================================ */

static int fp32_cores_per_sm(int major, int minor) {
    if (major == 7) return 64;
    if (major == 8 && minor == 0) return 64;
    if (major == 8) return 128;
    if (major == 9) return 128;
    return 128;
}

/* ---- micro: measure the machine ----------------------------------- */
static void run_micro(void) {
    printf("================================================================\n");
    printf(" DEVICE CEILINGS (measured, not datasheet)\n");
    printf("================================================================\n");
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int sm = prop.multiProcessorCount, clk_khz = 0, mclk_khz = 0, bus = 0, l2 = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&clk_khz, cudaDevAttrClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&mclk_khz, cudaDevAttrMemoryClockRate, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&bus, cudaDevAttrGlobalMemoryBusWidth, dev));
    CUDA_CHECK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, dev));
    int cores = fp32_cores_per_sm(prop.major, prop.minor);
    double nominal_fp32 = 2.0 * cores * sm * (clk_khz * 1e3) / 1e9;          /* GFLOP/s */
    double nominal_bw = 2.0 * (mclk_khz * 1e3) * (bus / 8.0) / 1e9;          /* GB/s    */

    /* DRAM copy bandwidth (float4, 256 MB in + 256 MB out per pass) */
    size_t nf4 = (size_t)16 << 20;
    float4 *din, *dout;
    CUDA_CHECK(cudaMalloc(&din, nf4 * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&dout, nf4 * sizeof(float4)));
    CUDA_CHECK(cudaMemset(din, 0, nf4 * sizeof(float4)));
    TimeStat tc = time_it([&] { copy_kernel<<<sm * 8, 256>>>(din, dout, nf4); });
    double copy_bytes = 2.0 * (double)nf4 * sizeof(float4);
    double copy_gbs = copy_bytes / (tc.mn * 1e6);
    cudaFree(din); cudaFree(dout);

    /* FP32 FMA peak + observed SM clock (one resident wave: 4 blocks x 256 thr / SM) */
    int blocks = sm * 4, threads = 256, iters = 1 << 18;
    float *dout2; long long *dres;
    CUDA_CHECK(cudaMalloc(&dout2, (size_t)blocks * threads * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dres, 2 * sizeof(long long)));
    TimeStat tf = time_it([&] { fma_peak_kernel<<<blocks, threads>>>(dout2, iters, dres); });
    long long hres[2] = {0, 1};
    CUDA_CHECK(cudaMemcpy(hres, dres, sizeof(hres), cudaMemcpyDeviceToHost));
    double obs_clock_ghz = (double)hres[0] / (double)hres[1];                 /* cycles/ns = GHz */
    double fma_flops = 2.0 * 8.0 * (double)iters * (double)blocks * threads;
    double fma_gflops = fma_flops / (tf.mn * 1e6);
    cudaFree(dout2); cudaFree(dres);

    double smem_bw_gbs = 32.0 * 4.0 * sm * obs_clock_ghz;                     /* 128 B/clk/SM */

    printf("Device                    : %s (cc %d.%d, %d SMs, %d FP32 cores/SM)\n",
           prop.name, prop.major, prop.minor, sm, cores);
    printf("L2 cache                  : %.1f MB\n", l2 / 1048576.0);
    printf("Nominal FP32 peak         : %.0f GFLOP/s (at reported max clock %.0f MHz)\n",
           nominal_fp32, clk_khz / 1e3);
    printf("Measured FP32 FMA peak    : %.0f GFLOP/s\n", fma_gflops);
    printf("Observed SM clock         : %.3f GHz (clock64 / globaltimer)\n", obs_clock_ghz);
    printf("Nominal DRAM bandwidth    : %.0f GB/s\n", nominal_bw);
    printf("Measured DRAM copy BW     : %.0f GB/s\n", copy_gbs);
    printf("Est. shared-mem BW        : %.0f GB/s (128 B/clk/SM at observed clock)\n", smem_bw_gbs);

    FILE *csv = fopen("results_m3_micro.csv", "w");
    fprintf(csv, "key,value\n");
    fprintf(csv, "device,%s\n", prop.name);
    fprintf(csv, "cc,%d.%d\n", prop.major, prop.minor);
    fprintf(csv, "sm_count,%d\n", sm);
    fprintf(csv, "fp32_cores_per_sm,%d\n", cores);
    fprintf(csv, "l2_mb,%.2f\n", l2 / 1048576.0);
    fprintf(csv, "max_clock_mhz,%.1f\n", clk_khz / 1e3);
    fprintf(csv, "nominal_fp32_gflops,%.1f\n", nominal_fp32);
    fprintf(csv, "measured_fma_gflops,%.1f\n", fma_gflops);
    fprintf(csv, "observed_clock_ghz,%.4f\n", obs_clock_ghz);
    fprintf(csv, "nominal_dram_gbs,%.1f\n", nominal_bw);
    fprintf(csv, "measured_copy_gbs,%.1f\n", copy_gbs);
    fprintf(csv, "smem_bw_gbs,%.1f\n", smem_bw_gbs);
    fclose(csv);
    printf("Raw data written to results_m3_micro.csv\n\n");
}

/* ---- validate -------------------------------------------------------- */
struct Var { const char *name; Kind kind; int cfg; };

static void run_validation(void) {
    printf("================================================================\n");
    printf(" CORRECTNESS VALIDATION (all Milestone-3 kernels)\n");
    printf(" FP32 kernels : vs sequential ikj AND long-double reference\n");
    printf("                (eps_F < 5e-4, max_abs < 5e-3  -- same as M1/M2)\n");
    printf(" FP16-TC path : total error vs original-input reference (eps_F < 2e-3,\n");
    printf("                max_abs < 5e-2) AND accumulation-only error vs the\n");
    printf("                reference on FP16-rounded inputs (M1/M2 thresholds)\n");
    printf("================================================================\n");

    struct Shape { int M, N, K; };
    Shape shapes[] = {{8,8,8},{16,16,16},{32,32,32},{63,63,63},{64,64,64},
                      {129,129,129},{200,200,200},{256,256,256},
                      {37,129,64},{130,70,95},{257,33,129}};          /* last 3: rectangular */
    int nshapes = (int)(sizeof(shapes) / sizeof(shapes[0]));

    std::vector<Var> vars;
    vars.push_back({"naive_m2", K_NAIVE, 0});
    vars.push_back({"naive_uncoalesced", K_UNCOAL, 0});
    vars.push_back({"smem_tile8", K_SMEM8, 0});
    vars.push_back({"smem_tile16", K_SMEM16, 0});
    vars.push_back({"smem_tile32", K_SMEM, 0});
    vars.push_back({"smem_double_buffer16", K_SMEM_DB16, 0});
    vars.push_back({"smem_double_buffer32", K_SMEM_DB32, 0});
    for (int i = 0; i < NCFG; i++) vars.push_back({g_cfgs[i].name, K_REG, i});
    vars.push_back({"wmma_fp16_tc", K_WMMA, 0});

    const double TOL_ABS = 5e-3, TOL_FRO = 5e-4;
    const double TOL_ABS16 = 5e-2, TOL_FRO16 = 2e-3;
    unsigned int seed = 42;
    float alpha = 1.3f, beta = 0.7f;
    int all_pass = 1;

    FILE *csv = fopen("results_m3_validation.csv", "w");
    fprintf(csv, "M,N,K,kernel,eps_vs_ikj,eps_vs_ref,maxabs_vs_ref,eps_vs_quantref,status\n");

    for (int s = 0; s < nshapes; s++) {
        int M = shapes[s].M, N = shapes[s].N, K = shapes[s].K;
        size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
        float *A = alloc_f(nA), *B = alloc_f(nB), *C0 = alloc_f(nC);
        float *Aq = alloc_f(nA), *Bq = alloc_f(nB);
        float *Cseq = alloc_f(nC), *Cref = alloc_f(nC), *Cqref = alloc_f(nC), *Ch = alloc_f(nC);
        fill_random(A, nA, &seed); fill_random(B, nB, &seed); fill_random(C0, nC, &seed);
        for (size_t i = 0; i < nA; i++) Aq[i] = q16(A[i]);
        for (size_t i = 0; i < nB; i++) Bq[i] = q16(B[i]);
        memcpy(Cseq, C0, nC * sizeof(float));
        ikj_gemm(M, N, K, alpha, A, B, beta, Cseq);
        reference_gemm_highprec(M, N, K, alpha, A, B, beta, C0, Cref);
        reference_gemm_highprec(M, N, K, alpha, Aq, Bq, beta, C0, Cqref);

        Dev d;
        dev_alloc(d, M, N, K);
        CUDA_CHECK(cudaMemcpy(d.dA, A, nA * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.dB, B, nB * sizeof(float), cudaMemcpyHostToDevice));
        dev_convert_half(d);

        int ran = 0, skipped = 0, failed = 0;
        double worst_fro = 0.0, worst_abs = 0.0;
        const char *worst_name = "-";
        for (size_t v = 0; v < vars.size(); v++) {
            CUDA_CHECK(cudaMemcpy(d.dC, C0, nC * sizeof(float), cudaMemcpyHostToDevice));
            if (!launch_kind(vars[v].kind, vars[v].cfg, d, alpha, beta, 0)) { skipped++; continue; }
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaMemcpy(Ch, d.dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
            ran++;
            double ma_ikj, fr_ikj, ma_ref, fr_ref, ma_q = 0, fr_q = 0;
            cmp(Ch, Cseq, nC, &ma_ikj, &fr_ikj);
            cmp(Ch, Cref, nC, &ma_ref, &fr_ref);
            bool ok;
            if (vars[v].kind == K_WMMA) {
                cmp(Ch, Cqref, nC, &ma_q, &fr_q);
                ok = (fr_ref < TOL_FRO16 && ma_ref < TOL_ABS16 && fr_q < TOL_FRO && ma_q < TOL_ABS);
            } else {
                fr_q = -1.0;
                ok = (fr_ikj < TOL_FRO && ma_ikj < TOL_ABS && fr_ref < TOL_FRO && ma_ref < TOL_ABS);
            }
            if (!ok) { failed++; all_pass = 0; }
            if (fr_ref > worst_fro) { worst_fro = fr_ref; worst_abs = ma_ref; worst_name = vars[v].name; }
            fprintf(csv, "%d,%d,%d,%s,%.4e,%.4e,%.4e,%.4e,%s\n", M, N, K, vars[v].name,
                    fr_ikj, fr_ref, ma_ref, fr_q, ok ? "PASS" : "FAIL");
        }
        printf("Shape %4d x %4d x %4d : %2d kernels run, %2d skipped (float4 alignment), "
               "%d failed | worst eps_F(ref)=%.2e [%s]\n",
               M, N, K, ran, skipped, failed, worst_fro, worst_name);
        (void)worst_abs;

        dev_free(d);
        free(A); free(B); free(C0); free(Aq); free(Bq);
        free(Cseq); free(Cref); free(Cqref); free(Ch);
    }
    fclose(csv);
    printf("----------------------------------------------------------------\n");
    printf("OVERALL VALIDATION RESULT: %s\n", all_pass ? "ALL TESTS PASSED" : "FAILURES DETECTED");
    printf("Per-kernel rows written to results_m3_validation.csv\n");
    printf("================================================================\n\n");
}

/* ---- bench: ablation ladder vs cuBLAS --------------------------------- */
static void run_benchmark(void) {
    printf("================================================================\n");
    printf(" ABLATION LADDER vs cuBLAS (square GEMM, alpha=1, beta=0)\n");
    printf(" timing: 1 warm-up + adaptive reps, median ms, kernel-only\n");
    printf("================================================================\n");
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int nsizes = (int)(sizeof(sizes) / sizeof(sizes[0]));
    int cfg_s = find_cfg(128, 128, 8, 8, 8, false);
    int cfg_v = find_cfg(128, 128, 8, 8, 8, true);

    struct Stage { const char *name; Kind kind; int cfg; };
    Stage stages[] = {
        {"naive_m2", K_NAIVE, 0},
        {"naive_uncoalesced", K_UNCOAL, 0},
        {"smem_tile8", K_SMEM8, 0},
        {"smem_tile16", K_SMEM16, 0},
        {"smem_tile32", K_SMEM, 0},
        {"smem_double_buffer16", K_SMEM_DB16, 0},
        {"reg2d_128x128x8_scalar", K_REG, cfg_s},
        {"reg2d_128x128x8_vec", K_REG, cfg_v},
        {"wmma_fp16_tc", K_WMMA, 0},
        {"cublas_sgemm", K_SGEMM, 0},
        {"cublas_tensorop_fp16", K_TCGEMM, 0}};
    int nstages = (int)(sizeof(stages) / sizeof(stages[0]));

    FILE *csv = fopen("results_m3_ablation.csv", "w");
    fprintf(csv, "kernel,N,median_ms,min_ms,gflops,eps_vs_cublas\n");
    unsigned int seed = 7;

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        size_t n = (size_t)N * N;
        double flops = 2.0 * (double)N * N * N;
        float *hA = alloc_f(n), *hB = alloc_f(n), *hC = alloc_f(n), *hR = alloc_f(n);
        fill_random(hA, n, &seed); fill_random(hB, n, &seed);
        Dev d;
        dev_alloc(d, N, N, N);
        CUDA_CHECK(cudaMemcpy(d.dA, hA, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.dB, hB, n * sizeof(float), cudaMemcpyHostToDevice));
        dev_convert_half(d);

        /* cuBLAS SGEMM output is the on-device correctness reference at scale */
        launch_kind(K_SGEMM, 0, d, 1.0f, 0.0f, 0);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(hR, d.dC, n * sizeof(float), cudaMemcpyDeviceToHost));

        printf("\nN = %d\n%-26s %12s %12s %14s\n", N, "kernel", "median ms", "GFLOP/s", "eps vs cuBLAS");
        for (int k = 0; k < nstages; k++) {
            if (stages[k].kind == K_UNCOAL && N > 4096) continue;     /* too slow, no new info */
            CUDA_CHECK(cudaMemset(d.dC, 0, n * sizeof(float)));
            TimeStat t = time_it([&] { launch_kind(stages[k].kind, stages[k].cfg, d, 1.0f, 0.0f, 0); });
            CUDA_CHECK(cudaMemcpy(hC, d.dC, n * sizeof(float), cudaMemcpyDeviceToHost));
            double ma, fr;
            cmp(hC, hR, n, &ma, &fr);
            double g = gflops_of(flops, t.med);
            printf("%-26s %12.4f %12.1f %14.3e\n", stages[k].name, t.med, g, fr);
            fprintf(csv, "%s,%d,%.5f,%.5f,%.3f,%.4e\n", stages[k].name, N, t.med, t.mn, g, fr);
        }
        dev_free(d);
        free(hA); free(hB); free(hC); free(hR);
    }
    fclose(csv);
    printf("================================================================\n");
    printf("Raw data written to results_m3_ablation.csv\n\n");
}

/* ---- tune: tile-configuration sweep ------------------------------------ */
static void run_tuning(void) {
    printf("================================================================\n");
    printf(" TILE-CONFIGURATION SWEEP (register-blocked kernel, FP32)\n");
    printf("================================================================\n");
    int sizes[] = {1024, 2048, 4096};
    FILE *csv = fopen("results_m3_tuning.csv", "w");
    fprintf(csv, "config,BM,BN,BK,TM,TN,vec,threads,regs,spill_bytes,smem_bytes,blocks_per_sm,N,median_ms,gflops\n");
    unsigned int seed = 11;
    for (int si = 0; si < 3; si++) {
        int N = sizes[si];
        size_t n = (size_t)N * N;
        double flops = 2.0 * (double)N * N * N;
        float *hA = alloc_f(n), *hB = alloc_f(n);
        fill_random(hA, n, &seed); fill_random(hB, n, &seed);
        Dev d;
        dev_alloc(d, N, N, N);
        CUDA_CHECK(cudaMemcpy(d.dA, hA, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.dB, hB, n * sizeof(float), cudaMemcpyHostToDevice));
        printf("\nN = %d\n%-24s %5s %6s %8s %5s %10s %10s\n", N, "config", "regs", "spillB",
               "blk/SM", "thr", "median ms", "GFLOP/s");
        for (int i = 0; i < NCFG; i++) {
            const Cfg &c = g_cfgs[i];
            cudaFuncAttributes fa;
            int bps = 0;
            c.q(&fa, &bps);
            int threads = (c.BM / c.TM) * (c.BN / c.TN);
            TimeStat t = time_it([&] { launch_kind(K_REG, i, d, 1.0f, 0.0f, 0); });
            double g = gflops_of(flops, t.med);
            printf("%-24s %5d %6zu %8d %5d %10.4f %10.1f\n", c.name, fa.numRegs,
                   (size_t)fa.localSizeBytes, bps, threads, t.med, g);
            fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%d,%d,%zu,%zu,%d,%d,%.5f,%.3f\n", c.name,
                    c.BM, c.BN, c.BK, c.TM, c.TN, c.vec ? 1 : 0, threads, fa.numRegs,
                    (size_t)fa.localSizeBytes, (size_t)fa.sharedSizeBytes, bps, N, t.med, g);
        }
        dev_free(d);
        free(hA); free(hB);
    }
    fclose(csv);
    printf("================================================================\n");
    printf("Raw data written to results_m3_tuning.csv\n\n");
}

/* ---- precision: where does the FP16 error come from? ------------------- */
static void run_precision(void) {
    printf("================================================================\n");
    printf(" PRECISION STUDY (M=N=128, alpha=1, beta=0)\n");
    printf(" eps_orig  = ||C - C_ref(original inputs)||_F / ||C_ref||_F\n");
    printf(" eps_quant = ||C - C_ref(FP16-rounded inputs)||_F / ||C_ref||_F\n");
    printf("            (isolates accumulation error from input quantization)\n");
    printf("================================================================\n");
    int Ks[] = {16, 64, 256, 1024, 4096, 8192};
    int nK = (int)(sizeof(Ks) / sizeof(Ks[0]));
    const int M = 128, N = 128;
    int cfg_s = find_cfg(128, 128, 8, 8, 8, false);
    FILE *csv = fopen("results_m3_precision.csv", "w");
    fprintf(csv, "dist,K,kernel,eps_orig,eps_quant\n");
    unsigned int seed = 99;
    const char *dists[] = {"signed", "positive"};

    for (int dist = 0; dist < 2; dist++) {
        printf("\nInput distribution: %s\n%8s %-14s %12s %12s\n", dists[dist], "K", "kernel",
               "eps_orig", "eps_quant");
        for (int ki = 0; ki < nK; ki++) {
            int K = Ks[ki];
            size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
            float *A = alloc_f(nA), *B = alloc_f(nB), *C0 = alloc_f(nC);
            float *Aq = alloc_f(nA), *Bq = alloc_f(nB);
            float *Cref = alloc_f(nC), *Cqref = alloc_f(nC), *Ch = alloc_f(nC);
            if (dist == 0) { fill_random(A, nA, &seed); fill_random(B, nB, &seed); }
            else           { fill_positive(A, nA, &seed); fill_positive(B, nB, &seed); }
            memset(C0, 0, nC * sizeof(float));
            for (size_t i = 0; i < nA; i++) Aq[i] = q16(A[i]);
            for (size_t i = 0; i < nB; i++) Bq[i] = q16(B[i]);
            reference_gemm_highprec(M, N, K, 1.0f, A, B, 0.0f, C0, Cref);
            reference_gemm_highprec(M, N, K, 1.0f, Aq, Bq, 0.0f, C0, Cqref);

            Dev d;
            dev_alloc(d, M, N, K);
            CUDA_CHECK(cudaMemcpy(d.dA, A, nA * sizeof(float), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d.dB, B, nB * sizeof(float), cudaMemcpyHostToDevice));
            dev_convert_half(d);

            for (int w = 0; w < 2; w++) {
                Kind kind = (w == 0) ? K_REG : K_WMMA;
                const char *name = (w == 0) ? "fp32_reg2d" : "fp16_tensorcore";
                CUDA_CHECK(cudaMemset(d.dC, 0, nC * sizeof(float)));
                launch_kind(kind, cfg_s, d, 1.0f, 0.0f, 0);
                CUDA_CHECK(cudaDeviceSynchronize());
                CUDA_CHECK(cudaMemcpy(Ch, d.dC, nC * sizeof(float), cudaMemcpyDeviceToHost));
                double ma, e_orig, e_q;
                cmp(Ch, Cref, nC, &ma, &e_orig);
                cmp(Ch, Cqref, nC, &ma, &e_q);
                printf("%8d %-14s %12.3e %12.3e\n", K, name, e_orig, e_q);
                fprintf(csv, "%s,%d,%s,%.4e,%.4e\n", dists[dist], K, name, e_orig, e_q);
            }
            dev_free(d);
            free(A); free(B); free(C0); free(Aq); free(Bq); free(Cref); free(Cqref); free(Ch);
        }
    }
    fclose(csv);
    printf("================================================================\n");
    printf("Raw data written to results_m3_precision.csv\n\n");
}

/* ---- pipe: panel-pipelined streaming GEMM ------------------------------ *
 * Single-node proxy for tile-level compute/communication overlap (Paper 3):
 * C is split into row panels; B stays resident; each panel does
 * H2D(A panel) -> kernel -> D2H(C panel) on one of S streams.  The T4 has
 * independent H2D and D2H copy engines, so all three phases can overlap.
 * NOTE: one GPU only -- this is NOT a multi-GPU measurement.              */
static double run_pipe_once(int N, int P, int S, float *hA, float *hB, float *hC,
                            float *dA, float *dB, float *dC, cudaStream_t *st,
                            cudaEvent_t bready) {
    const int M = N, K = N;
    int mp = (((M + P - 1) / P) + 127) / 128 * 128;
    int panels = (M + mp - 1) / mp;
    double t0 = now_seconds();
    CUDA_CHECK(cudaMemcpyAsync(dB, hB, (size_t)K * N * sizeof(float), cudaMemcpyHostToDevice, st[0]));
    CUDA_CHECK(cudaEventRecord(bready, st[0]));
    for (int s = 1; s < S; s++) CUDA_CHECK(cudaStreamWaitEvent(st[s], bready, 0));
    for (int p = 0; p < panels; p++) {
        int s = p % S;
        int r0 = p * mp;
        int rows = std::min(mp, M - r0);
        CUDA_CHECK(cudaMemcpyAsync(dA + (size_t)r0 * K, hA + (size_t)r0 * K,
                                   (size_t)rows * K * sizeof(float), cudaMemcpyHostToDevice, st[s]));
        launch_reg2d<128, 128, 8, 8, 8, 0, true>(rows, N, K, 1.0f, dA + (size_t)r0 * K, dB, 0.0f,
                                                  dC + (size_t)r0 * N, nullptr, st[s]);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(hC + (size_t)r0 * N, dC + (size_t)r0 * N,
                                   (size_t)rows * N * sizeof(float), cudaMemcpyDeviceToHost, st[s]));
    }
    for (int s = 0; s < S; s++) CUDA_CHECK(cudaStreamSynchronize(st[s]));
    return (now_seconds() - t0) * 1e3;
}

static void run_pipeline(void) {
    printf("================================================================\n");
    printf(" PANEL-PIPELINED STREAMING GEMM (end-to-end incl. PCIe transfers)\n");
    printf("================================================================\n");
    int Ns[] = {2048, 4096, 8192};
    int Ps[] = {1, 2, 4, 8, 16};
    int Ss[] = {1, 2, 4};
    FILE *csv = fopen("results_m3_pipeline.csv", "w");
    fprintf(csv, "N,variant,P,S,time_ms\n");
    cudaStream_t st[4];
    for (int i = 0; i < 4; i++) CUDA_CHECK(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking));
    cudaEvent_t bready, e0, e1;
    CUDA_CHECK(cudaEventCreateWithFlags(&bready, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    unsigned int seed = 3;

    for (int ni = 0; ni < 3; ni++) {
        int N = Ns[ni];
        size_t n = (size_t)N * N, bytes = n * sizeof(float);
        float *hA, *hB, *hC;
        CUDA_CHECK(cudaMallocHost(&hA, bytes));
        CUDA_CHECK(cudaMallocHost(&hB, bytes));
        CUDA_CHECK(cudaMallocHost(&hC, bytes));
        fill_random(hA, n, &seed); fill_random(hB, n, &seed); memset(hC, 0, bytes);
        float *pA = alloc_f(n), *pB = alloc_f(n), *pC = alloc_f(n);
        memcpy(pA, hA, bytes); memcpy(pB, hB, bytes); memset(pC, 0, bytes);
        float *dA, *dB, *dC;
        CUDA_CHECK(cudaMalloc(&dA, bytes));
        CUDA_CHECK(cudaMalloc(&dB, bytes));
        CUDA_CHECK(cudaMalloc(&dC, bytes));

        /* (1) component times, pinned memory, default stream */
        auto comp = [&](auto fn) {
            fn(); CUDA_CHECK(cudaDeviceSynchronize());
            std::vector<double> v;
            for (int r = 0; r < 5; r++) {
                CUDA_CHECK(cudaEventRecord(e0, 0));
                fn();
                CUDA_CHECK(cudaEventRecord(e1, 0));
                CUDA_CHECK(cudaEventSynchronize(e1));
                float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
                v.push_back(ms);
            }
            return median_d(v);
        };
        double t_hA = comp([&] { cudaMemcpyAsync(dA, hA, bytes, cudaMemcpyHostToDevice, 0); });
        double t_hB = comp([&] { cudaMemcpyAsync(dB, hB, bytes, cudaMemcpyHostToDevice, 0); });
        double t_k  = comp([&] { launch_reg2d<128, 128, 8, 8, 8, 0, true>(N, N, N, 1.f, dA, dB, 0.f, dC, nullptr, 0); });
        double t_dC = comp([&] { cudaMemcpyAsync(hC, dC, bytes, cudaMemcpyDeviceToHost, 0); });
        fprintf(csv, "%d,comp_h2d_A,0,0,%.4f\n", N, t_hA);
        fprintf(csv, "%d,comp_h2d_B,0,0,%.4f\n", N, t_hB);
        fprintf(csv, "%d,comp_kernel,0,0,%.4f\n", N, t_k);
        fprintf(csv, "%d,comp_d2h_C,0,0,%.4f\n", N, t_dC);
        printf("\nN=%d  components (ms): H2D(A)=%.2f  H2D(B)=%.2f  kernel=%.2f  D2H(C)=%.2f\n",
               N, t_hA, t_hB, t_k, t_dC);

        /* (2) Milestone-2-style serial end-to-end with PAGEABLE host memory */
        auto pageable = [&]() -> double {
            double t0 = now_seconds();
            CUDA_CHECK(cudaMemcpy(dA, pA, bytes, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(dB, pB, bytes, cudaMemcpyHostToDevice));
            launch_reg2d<128, 128, 8, 8, 8, 0, true>(N, N, N, 1.f, dA, dB, 0.f, dC, nullptr, 0);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(pC, dC, bytes, cudaMemcpyDeviceToHost));
            return (now_seconds() - t0) * 1e3;
        };
        pageable();
        std::vector<double> v;
        for (int r = 0; r < 5; r++) v.push_back(pageable());
        double t_pg = median_d(v);
        fprintf(csv, "%d,pageable_serial,1,1,%.4f\n", N, t_pg);
        printf("  pageable serial (M2-style)      : %9.2f ms\n", t_pg);

        /* (3) pipelined, pinned memory */
        for (int pi = 0; pi < 5; pi++)
            for (int si = 0; si < 3; si++) {
                int P = Ps[pi], S = Ss[si];
                if (S > P) continue;
                run_pipe_once(N, P, S, hA, hB, hC, dA, dB, dC, st, bready);   /* warm-up */
                std::vector<double> w;
                for (int r = 0; r < 5; r++)
                    w.push_back(run_pipe_once(N, P, S, hA, hB, hC, dA, dB, dC, st, bready));
                double t = median_d(w);
                fprintf(csv, "%d,pipelined,%d,%d,%.4f\n", N, P, S, t);
                printf("  pipelined P=%2d S=%d              : %9.2f ms\n", P, S, t);
            }
        cudaFreeHost(hA); cudaFreeHost(hB); cudaFreeHost(hC);
        free(pA); free(pB); free(pC);
        cudaFree(dA); cudaFree(dB); cudaFree(dC);
    }
    fclose(csv);
    for (int i = 0; i < 4; i++) cudaStreamDestroy(st[i]);
    printf("================================================================\n");
    printf("Raw data written to results_m3_pipeline.csv\n\n");
}

/* ---- fuse: why a custom kernel can still be worth writing -------------- */
static void run_fusion(void) {
    printf("================================================================\n");
    printf(" FUSED EPILOGUE STUDY:  Y = ReLU(A*B + bias)\n");
    printf("================================================================\n");
    int sizes[] = {1024, 2048, 4096, 8192};
    FILE *csv = fopen("results_m3_fusion.csv", "w");
    fprintf(csv, "N,variant,median_ms,gflops_eq,eps_vs_cublas_plus_pass\n");
    unsigned int seed = 5;
    for (int si = 0; si < 4; si++) {
        int N = sizes[si];
        size_t n = (size_t)N * N;
        double flops = 2.0 * (double)N * N * N;
        float *hA = alloc_f(n), *hB = alloc_f(n), *hBias = alloc_f(N);
        float *hY1 = alloc_f(n), *hY2 = alloc_f(n);
        fill_random(hA, n, &seed); fill_random(hB, n, &seed); fill_random(hBias, N, &seed);
        Dev d;
        dev_alloc(d, N, N, N);
        CUDA_CHECK(cudaMemcpy(d.dA, hA, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.dB, hB, n * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d.dBias, hBias, N * sizeof(float), cudaMemcpyHostToDevice));
        int N4 = N / 4;
        size_t total4 = n / 4;
        unsigned blocks = (unsigned)((total4 + 255) / 256);
        auto epi = [&] {
            bias_relu_kernel<<<blocks, 256>>>(reinterpret_cast<float4 *>(d.dC),
                                              reinterpret_cast<const float4 *>(d.dBias), N4, total4);
        };
        int cfg_v = find_cfg(128, 128, 8, 8, 8, true);

        TimeStat t_fused = time_it([&] {
            launch_reg2d<128, 128, 8, 8, 8, 1, true>(N, N, N, 1.f, d.dA, d.dB, 0.f, d.dC, d.dBias, 0);
        });
        CUDA_CHECK(cudaMemcpy(hY1, d.dC, n * sizeof(float), cudaMemcpyDeviceToHost));

        TimeStat t_cb_epi = time_it([&] {
            launch_kind(K_SGEMM, 0, d, 1.f, 0.f, 0);
            epi();
        });
        CUDA_CHECK(cudaMemcpy(hY2, d.dC, n * sizeof(float), cudaMemcpyDeviceToHost));
        double ma, eps;
        cmp(hY1, hY2, n, &ma, &eps);

        TimeStat t_cu_epi = time_it([&] {
            launch_kind(K_REG, cfg_v, d, 1.f, 0.f, 0);
            epi();
        });
        TimeStat t_cu = time_it([&] { launch_kind(K_REG, cfg_v, d, 1.f, 0.f, 0); });
        TimeStat t_cb = time_it([&] { launch_kind(K_SGEMM, 0, d, 1.f, 0.f, 0); });

        struct R { const char *n; TimeStat t; };
        R rows[] = {{"custom_fused", t_fused}, {"cublas_plus_epilogue_pass", t_cb_epi},
                    {"custom_plus_epilogue_pass", t_cu_epi}, {"custom_gemm_only", t_cu},
                    {"cublas_gemm_only", t_cb}};
        printf("\nN = %d  (fused vs cuBLAS+pass: eps_F = %.2e)\n", N, eps);
        for (int i = 0; i < 5; i++) {
            printf("  %-28s %10.4f ms\n", rows[i].n, rows[i].t.med);
            fprintf(csv, "%d,%s,%.5f,%.3f,%.4e\n", N, rows[i].n, rows[i].t.med,
                    gflops_of(flops, rows[i].t.med), eps);
        }
        dev_free(d);
        free(hA); free(hB); free(hBias); free(hY1); free(hY2);
    }
    fclose(csv);
    printf("================================================================\n");
    printf("Raw data written to results_m3_fusion.csv\n\n");
}

int main(int argc, char **argv) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) { fprintf(stderr, "No CUDA-capable device found.\n"); return 1; }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device 0: %s (compute capability %d.%d)\n\n", prop.name, prop.major, prop.minor);
    if (prop.major < 7) { fprintf(stderr, "Tensor-Core path needs compute capability >= 7.0\n"); return 1; }
    CUBLAS_CHECK(cublasCreate(&g_cublas));

    const char *mode = (argc > 1) ? argv[1] : "all";
    bool all = (strcmp(mode, "all") == 0);
    if (all || !strcmp(mode, "micro"))     run_micro();
    if (all || !strcmp(mode, "validate"))  run_validation();
    if (all || !strcmp(mode, "bench"))     run_benchmark();
    if (all || !strcmp(mode, "tune"))      run_tuning();
    if (all || !strcmp(mode, "precision")) run_precision();
    if (all || !strcmp(mode, "pipe"))      run_pipeline();
    if (all || !strcmp(mode, "fuse"))      run_fusion();

    cublasDestroy(g_cublas);
    return 0;
}
