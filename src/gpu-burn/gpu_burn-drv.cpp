/*************************************************************************
 * gpu_burn-drv.cpp — GPU stress test driver (bundled version)
 * Based on https://github.com/wilicc/gpu-burn
 *
 * Performs continuous DGEMM operations on all GPUs simultaneously,
 * comparing results against a reference to detect hardware faults.
 * Uses CUBLAS for matrix multiply and a PTX kernel for comparison.
 *************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <signal.h>
#include <unistd.h>
#include <pthread.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define MATRIX_N 2048
#define USEMEM   0.9  /* fraction of GPU memory to use */

static volatile int g_running = 1;

static void sigHandler(int sig) { g_running = 0; }

/* Each GPU runs in its own thread */
typedef struct {
    int gpuId;
    int duration;
    int faults;
    double gflops;
    int maxTemp;
    int ok;
} GpuWork;

static void checkCuda(cudaError_t err, const char *msg, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s:%d: %s (%s)\n",
                msg, line, cudaGetErrorString(err), msg);
    }
}
#define CUDA_CHK(x) checkCuda((x), __FILE__, __LINE__)

static void checkCublas(cublasStatus_t s, const char *msg) {
    if (s != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "CUBLAS error: %s (code %d)\n", msg, (int)s);
    }
}

/* Load PTX compare kernel */
static CUfunction loadCompareKernel(void) {
    CUmodule module;
    CUfunction func;
    CUresult r = cuModuleLoad(&module, "compare.ptx");
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "Warning: Could not load compare.ptx (code %d), "
                "error checking disabled\n", (int)r);
        return NULL;
    }
    r = cuModuleGetFunction(&func, module, "compare");
    if (r != CUDA_SUCCESS) {
        fprintf(stderr, "Warning: compare kernel not found\n");
        return NULL;
    }
    return func;
}

static void *gpuThread(void *arg) {
    GpuWork *w = (GpuWork*)arg;
    w->ok = 1;
    w->faults = 0;
    w->gflops = 0;
    w->maxTemp = 0;

    CUDA_CHK(cudaSetDevice(w->gpuId));

    /* Query free memory */
    size_t freeMem, totalMem;
    CUDA_CHK(cudaMemGetInfo(&freeMem, &totalMem));
    size_t useMem = (size_t)((double)freeMem * USEMEM);

    /* How many MATRIX_N x MATRIX_N doubles can we fit? */
    size_t matBytes = (size_t)MATRIX_N * MATRIX_N * sizeof(double);
    int numMat = (int)(useMem / matBytes);
    if (numMat < 3) numMat = 3;  /* A, B, C minimum */

    /* Allocate matrices */
    double *d_A, *d_B, *d_C;
    CUDA_CHK(cudaMalloc(&d_A, matBytes));
    CUDA_CHK(cudaMalloc(&d_B, matBytes));
    CUDA_CHK(cudaMalloc(&d_C, matBytes));

    /* Fill A and B with random-ish data */
    double *h_init = (double*)malloc(matBytes);
    srand48(42 + w->gpuId);
    for (int i = 0; i < MATRIX_N * MATRIX_N; i++)
        h_init[i] = drand48();
    CUDA_CHK(cudaMemcpy(d_A, h_init, matBytes, cudaMemcpyHostToDevice));
    for (int i = 0; i < MATRIX_N * MATRIX_N; i++)
        h_init[i] = drand48();
    CUDA_CHK(cudaMemcpy(d_B, h_init, matBytes, cudaMemcpyHostToDevice));
    free(h_init);

    /* Create reference result */
    double *d_Cref;
    CUDA_CHK(cudaMalloc(&d_Cref, matBytes));

    cublasHandle_t handle;
    checkCublas(cublasCreate(&handle), "cublasCreate");

    double alpha = 1.0, beta = 0.0;
    /* Compute reference: Cref = A * B */
    checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        MATRIX_N, MATRIX_N, MATRIX_N,
        &alpha, d_A, MATRIX_N, d_B, MATRIX_N,
        &beta, d_Cref, MATRIX_N), "dgemm-ref");
    CUDA_CHK(cudaDeviceSynchronize());

    /* Load compare kernel */
    CUfunction compareFunc = loadCompareKernel();
    int *d_faultyElems = NULL;
    if (compareFunc) {
        CUDA_CHK(cudaMalloc(&d_faultyElems, sizeof(int)));
    }

    /* Stress loop */
    long long iters = 0;
    time_t startTime = time(NULL);
    double opsPerIter = 2.0 * MATRIX_N * MATRIX_N * MATRIX_N;

    while (g_running) {
        time_t now = time(NULL);
        if ((now - startTime) >= w->duration) break;

        /* C = A * B */
        beta = 0.0;
        checkCublas(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            MATRIX_N, MATRIX_N, MATRIX_N,
            &alpha, d_A, MATRIX_N, d_B, MATRIX_N,
            &beta, d_C, MATRIX_N), "dgemm");
        CUDA_CHK(cudaDeviceSynchronize());
        iters++;

        /* Compare every 10 iterations to reduce overhead */
        if (compareFunc && d_faultyElems && (iters % 10 == 0)) {
            int zero = 0;
            CUDA_CHK(cudaMemcpy(d_faultyElems, &zero, sizeof(int),
                                cudaMemcpyHostToDevice));
            int N = MATRIX_N * MATRIX_N;
            int blockSize = 256;
            int gridSize = (N + blockSize - 1) / blockSize;
            void *args[] = { &d_C, &d_Cref, &d_faultyElems, &N };
            cuLaunchKernel(compareFunc, gridSize, 1, 1,
                           blockSize, 1, 1, 0, 0, args, 0);
            CUDA_CHK(cudaDeviceSynchronize());
            int faults = 0;
            CUDA_CHK(cudaMemcpy(&faults, d_faultyElems, sizeof(int),
                                cudaMemcpyDeviceToHost));
            if (faults > 0) {
                fprintf(stderr, "GPU %d: %d faulty elements detected!\n",
                        w->gpuId, faults);
                w->faults += faults;
                w->ok = 0;
            }
        }
    }

    time_t endTime = time(NULL);
    double elapsed = (double)(endTime - startTime);
    if (elapsed < 1.0) elapsed = 1.0;
    w->gflops = (iters * opsPerIter) / elapsed / 1e9;

    /* Cleanup */
    if (d_faultyElems) cudaFree(d_faultyElems);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_Cref);
    cublasDestroy(handle);

    return NULL;
}

int main(int argc, char *argv[]) {
    int duration = 10;  /* default seconds */
    if (argc > 1) duration = atoi(argv[1]);
    if (duration <= 0) duration = 10;

    signal(SIGINT, sigHandler);
    signal(SIGTERM, sigHandler);

    /* Initialize CUDA driver API for PTX loading */
    cuInit(0);

    int nGpus;
    CUDA_CHK(cudaGetDeviceCount(&nGpus));
    if (nGpus <= 0) {
        fprintf(stderr, "No GPUs found!\n");
        return 1;
    }

    printf("gpu-burn: Burning %d GPU(s) for %d seconds\n", nGpus, duration);

    GpuWork *work = (GpuWork*)calloc(nGpus, sizeof(GpuWork));
    pthread_t *threads = (pthread_t*)calloc(nGpus, sizeof(pthread_t));

    for (int i = 0; i < nGpus; i++) {
        work[i].gpuId = i;
        work[i].duration = duration;
        pthread_create(&threads[i], NULL, gpuThread, &work[i]);
    }

    for (int i = 0; i < nGpus; i++) {
        pthread_join(threads[i], NULL);
    }

    /* Print results in format expected by gpu-burn.sh parser */
    int allOk = 1;
    for (int i = 0; i < nGpus; i++) {
        const char *status = work[i].ok ? "OK" : "FAULTY";
        if (!work[i].ok) allOk = 0;
        printf("GPU %d: %s - %.1f GFLOPS - faults: %d\n",
               i, status, work[i].gflops, work[i].faults);
    }

    printf("\n%s\n", allOk ? "All GPUs OK" : "ERRORS DETECTED");

    free(work);
    free(threads);
    return allOk ? 0 : 1;
}
