/* ============================================================
 * gemm_cuda_naive.cu
 *
 * Milestone 2 deliverable for the project "Optimizing a Custom
 * CUDA GEMM Kernel Toward cuBLAS-Level Performance".
 *
 * This is the FIRST (naive) CUDA parallelization of the GEMM
 * kernel validated sequentially in Milestone 1
 * (gemm_baseline.c). Each CUDA thread is responsible for
 * exactly one output element C[i][j], directly mirroring the
 * innermost accumulation loop of ijk_gemm from Milestone 1 --
 * this is the most literal possible parallelization of that
 * loop nest, with NO shared-memory tiling, NO register
 * blocking, and NO explicit coalescing optimization yet. Those
 * are deliberately left for later milestones (see the roadmap
 * in the Literature Review, Eq. 9), so that their individual
 * contribution can be measured against THIS baseline.
 *
 * Provides:
 *   1. gemm_naive_kernel   - one-thread-per-output-element CUDA
 *                            GEMM kernel (global memory only)
 *   2. ikj_gemm             - the Milestone-1 sequential
 *                            reference kernel (ported verbatim),
 *                            used as the correctness baseline
 *                            and as the speedup denominator
 *   3. reference_gemm_highprec - the same long-double ground
 *                            truth used in Milestone 1
 *   4. A correctness-validation harness comparing the CUDA
 *                            kernel's output against BOTH the
 *                            sequential ikj_gemm output and the
 *                            high-precision reference, using the
 *                            same Frobenius-norm relative-error
 *                            criterion established in Milestone 1
 *   5. A performance/scaling harness that sweeps problem size
 *                            and reports achieved GFLOP/s and
 *                            speedup over the Milestone-1
 *                            sequential ikj_gemm baseline
 *
 * Build:
 *   nvcc -O3 -arch=sm_70 -std=c++14 -o gemm_cuda_naive gemm_cuda_naive.cu
 *   (replace sm_70 with the compute capability of your target GPU,
 *    e.g. sm_75 Turing, sm_80/sm_86 Ampere, sm_89 Ada, sm_90 Hopper)
 *
 * Usage:
 *   ./gemm_cuda_naive validate     # correctness validation only
 *   ./gemm_cuda_naive bench        # performance/scaling sweep only
 *   ./gemm_cuda_naive all          # both (default)
 * ============================================================ */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <ctime>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err__ = (call);                                         \
        if (err__ != cudaSuccess) {                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,\
                    cudaGetErrorString(err__));                             \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

/* Thread-block tile dimensions for the naive kernel launch grid.
 * NOTE: this is only the launch geometry (how output elements are
 * partitioned across threads) -- it is NOT shared-memory tiling.
 * Each thread still reads its A-row and B-column directly from
 * global memory on every access. */
#define BLOCK_DIM 16

/* ---------------------------------------------------------- */
/* 1. Naive CUDA GEMM kernel: one thread per output element     */
/*    C[i][j]. Global-memory reads only, no shared memory.      */
/* ---------------------------------------------------------- */
__global__ void gemm_naive_kernel(int M, int N, int K, float alpha,
                                   const float *A, const float *B,
                                   float beta, float *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;   /* i */
    int col = blockIdx.x * blockDim.x + threadIdx.x;   /* j */

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; k++) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * acc + beta * C[row * N + col];
    }
}

/* Host wrapper: allocates device memory, copies in, launches the
 * kernel, copies the result back, and (optionally) times the
 * kernel-only execution with CUDA events. */
static float gemm_naive_launch(int M, int N, int K, float alpha,
                                const float *hA, const float *hB,
                                float beta, float *hC, bool time_it) {
    float *dA, *dB, *dC;
    size_t sizeA = (size_t)M * K * sizeof(float);
    size_t sizeB = (size_t)K * N * sizeof(float);
    size_t sizeC = (size_t)M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dA, sizeA));
    CUDA_CHECK(cudaMalloc(&dB, sizeB));
    CUDA_CHECK(cudaMalloc(&dC, sizeC));

    CUDA_CHECK(cudaMemcpy(dA, hA, sizeA, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB, sizeB, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dC, hC, sizeC, cudaMemcpyHostToDevice));

    dim3 blockDim(BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim((N + BLOCK_DIM - 1) / BLOCK_DIM,
                 (M + BLOCK_DIM - 1) / BLOCK_DIM);

    float ms = 0.0f;
    cudaEvent_t start, stop;
    if (time_it) {
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
    }

    gemm_naive_kernel<<<gridDim, blockDim>>>(M, N, K, alpha, dA, dB, beta, dC);
    CUDA_CHECK(cudaGetLastError());

    if (time_it) {
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
    } else {
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(hC, dC, sizeC, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return ms; /* milliseconds, 0 if time_it == false */
}

/* ---------------------------------------------------------- */
/* 2. Sequential ikj_gemm, ported verbatim from Milestone 1     */
/*    (gemm_baseline.c). Used as (a) a correctness cross-check  */
/*    and (b) the speedup denominator, since it was already     */
/*    established as the faster of the two Milestone-1 kernels. */
/* ---------------------------------------------------------- */
void ikj_gemm(int M, int N, int K, float alpha,
              const float *A, const float *B, float beta, float *C) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            C[i * N + j] *= beta;

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = alpha * A[i * K + k];
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/* ---------------------------------------------------------- */
/* 3. High-precision reference, ported verbatim from Milestone 1*/
/* ---------------------------------------------------------- */
void reference_gemm_highprec(int M, int N, int K, float alpha,
                              const float *A, const float *B,
                              float beta, const float *C_in,
                              float *C_out) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            long double acc = 0.0L;
            for (int k = 0; k < K; k++) {
                acc += (long double)A[i * K + k] * (long double)B[k * N + j];
            }
            long double val = (long double)alpha * acc +
                               (long double)beta * (long double)C_in[i * N + j];
            C_out[i * N + j] = (float)val;
        }
    }
}

/* ---------------------------------------------------------- */
/* Utilities                                                     */
/* ---------------------------------------------------------- */
static float *alloc_f(size_t n) {
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) { fprintf(stderr, "alloc failed\n"); exit(1); }
    return p;
}
static void fill_random(float *M, size_t n, unsigned int *seed) {
    for (size_t i = 0; i < n; i++)
        M[i] = ((float)rand_r(seed) / (float)RAND_MAX) * 2.0f - 1.0f;
}
static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Same Frobenius-norm relative-error metric used in Milestone 1,
 * chosen there specifically because per-element relative error is
 * numerically unstable near zero-valued reference entries. */
static void compare(const char *label, const float *C, const float *Cref,
                     int M, int N, double *max_abs, double *fro_rel) {
    double ma = 0.0, num = 0.0, den = 0.0;
    for (int i = 0; i < M * N; i++) {
        double diff = fabs((double)C[i] - (double)Cref[i]);
        if (diff > ma) ma = diff;
        num += diff * diff;
        den += (double)Cref[i] * (double)Cref[i];
    }
    *max_abs = ma;
    *fro_rel = sqrt(num) / (sqrt(den) + 1e-30);
    printf("  %-32s max_abs_err = %.3e   frobenius_rel_err = %.3e\n",
           label, ma, *fro_rel);
}

/* ---------------------------------------------------------- */
/* 4. Correctness validation: CUDA kernel vs sequential ikj_gemm */
/*    vs high-precision reference, same sizes/tolerances/seed as */
/*    Milestone 1 so the two reports are directly comparable.    */
/* ---------------------------------------------------------- */
void run_validation(void) {
    printf("================================================================\n");
    printf(" CORRECTNESS VALIDATION (naive CUDA kernel)\n");
    printf(" (checked against sequential ikj_gemm AND long-double reference)\n");
    printf("================================================================\n");

    int sizes[] = {8, 16, 32, 63, 64, 129, 200, 256};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    unsigned int seed = 42;
    float alpha = 1.3f, beta = 0.7f;

    const double TOL_ABS = 5e-3;
    const double TOL_FRO = 5e-4;
    int all_pass = 1;

    for (int s = 0; s < n_sizes; s++) {
        int M = sizes[s], K = sizes[s], N = sizes[s];
        float *A = alloc_f((size_t)M * K);
        float *B = alloc_f((size_t)K * N);
        float *C0 = alloc_f((size_t)M * N);
        float *C_cuda = alloc_f((size_t)M * N);
        float *C_seq  = alloc_f((size_t)M * N);
        float *C_ref  = alloc_f((size_t)M * N);

        fill_random(A, (size_t)M * K, &seed);
        fill_random(B, (size_t)K * N, &seed);
        fill_random(C0, (size_t)M * N, &seed);

        memcpy(C_cuda, C0, (size_t)M * N * sizeof(float));
        memcpy(C_seq,  C0, (size_t)M * N * sizeof(float));

        gemm_naive_launch(M, N, K, alpha, A, B, beta, C_cuda, false);
        ikj_gemm(M, N, K, alpha, A, B, beta, C_seq);
        reference_gemm_highprec(M, N, K, alpha, A, B, beta, C0, C_ref);

        printf("Problem size %d x %d x %d:\n", M, K, N);
        double ma1, mr1, ma2, mr2;
        compare("CUDA naive vs sequential ikj", C_cuda, C_seq, M, N, &ma1, &mr1);
        compare("CUDA naive vs highprec ref",   C_cuda, C_ref, M, N, &ma2, &mr2);

        if (ma1 > TOL_ABS || mr1 > TOL_FRO || ma2 > TOL_ABS || mr2 > TOL_FRO) {
            printf("  --> FAIL at size %d\n", M);
            all_pass = 0;
        } else {
            printf("  --> PASS\n");
        }

        free(A); free(B); free(C0); free(C_cuda); free(C_seq); free(C_ref);
    }

    printf("----------------------------------------------------------------\n");
    printf("OVERALL VALIDATION RESULT: %s\n", all_pass ? "ALL TESTS PASSED" : "FAILURES DETECTED");
    printf("================================================================\n\n");
}

/* ---------------------------------------------------------- */
/* 5. Performance / scaling benchmark                            */
/*    Sweeps problem size, reports CUDA kernel-only GFLOP/s      */
/*    (host<->device transfer time excluded, kernel time only,   */
/*    measured via cudaEvent) and speedup over the Milestone-1   */
/*    sequential ikj_gemm baseline at the same size.              */
/* ---------------------------------------------------------- */
void run_benchmark(void) {
    printf("================================================================\n");
    printf(" CUDA NAIVE GEMM PERFORMANCE / SCALING SWEEP (square GEMM, M=N=K)\n");
    printf("================================================================\n");
    printf("%8s %14s %16s %14s %12s\n",
           "N", "seq ikj (s)", "seq GFLOP/s", "cuda GFLOP/s", "speedup");

    /* Extend beyond the Milestone-1 sweep (up to N=1024) since the
     * GPU can handle larger problems -- this is what lets the
     * "scaling" part of the milestone title actually be shown. */
    int sizes[] = {64, 128, 256, 384, 512, 768, 1024, 2048, 4096};
    int n_sizes = sizeof(sizes) / sizeof(sizes[0]);
    unsigned int seed = 7;

    FILE *csv = fopen("results_milestone2.csv", "w");
    fprintf(csv, "N,seq_ikj_time_s,seq_ikj_gflops,cuda_naive_time_s,cuda_naive_gflops,speedup\n");

    for (int s = 0; s < n_sizes; s++) {
        int N = sizes[s];
        double flops = 2.0 * (double)N * (double)N * (double)N;

        float *A  = alloc_f((size_t)N * N);
        float *B  = alloc_f((size_t)N * N);
        float *C1 = alloc_f((size_t)N * N);
        float *C2 = alloc_f((size_t)N * N);

        fill_random(A, (size_t)N * N, &seed);
        fill_random(B, (size_t)N * N, &seed);
        fill_random(C1, (size_t)N * N, &seed);
        memcpy(C2, C1, (size_t)N * N * sizeof(float));

        /* Sequential baseline (skip for the largest sizes if it
         * becomes impractically slow -- ijk/ikj at N=4096 can take
         * a long time on CPU; adjust the cutoff to your machine). */
        double t_seq = -1.0, gflops_seq = -1.0;
        if (N <= 1024) {
            double t0 = now_seconds();
            ikj_gemm(N, N, N, 1.0f, A, B, 0.0f, C1);
            double t1 = now_seconds();
            t_seq = t1 - t0;
            gflops_seq = flops / t_seq / 1e9;
        }

        /* CUDA naive kernel, kernel-only timing via cudaEvent */
        float ms = gemm_naive_launch(N, N, N, 1.0f, A, B, 0.0f, C2, true);
        double t_cuda = ms / 1000.0;
        double gflops_cuda = flops / t_cuda / 1e9;
        double speedup = (t_seq > 0) ? (t_seq / t_cuda) : -1.0;

        if (t_seq > 0)
            printf("%8d %14.4f %16.3f %14.3f %12.2fx\n",
                   N, t_seq, gflops_seq, gflops_cuda, speedup);
        else
            printf("%8d %14s %16s %14.3f %12s\n",
                   N, "skipped", "skipped", gflops_cuda, "n/a");

        fprintf(csv, "%d,%.6f,%.4f,%.6f,%.4f,%.4f\n",
                N, t_seq, gflops_seq, t_cuda, gflops_cuda, speedup);

        free(A); free(B); free(C1); free(C2);
    }
    fclose(csv);
    printf("================================================================\n");
    printf("Raw data written to results_milestone2.csv\n\n");
}

int main(int argc, char **argv) {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable device found.\n");
        return 1;
    }
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Using device 0: %s (compute capability %d.%d)\n\n",
           prop.name, prop.major, prop.minor);

    const char *mode = (argc > 1) ? argv[1] : "all";
    if (strcmp(mode, "validate") == 0) {
        run_validation();
    } else if (strcmp(mode, "bench") == 0) {
        run_benchmark();
    } else {
        run_validation();
        run_benchmark();
    }
    return 0;
}
