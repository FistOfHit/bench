/*************************************************************************
 * common.h — Shared helpers for NCCL performance tests (bundled version)
 * Based on https://github.com/NVIDIA/nccl-tests
 *************************************************************************/
#ifndef NCCL_TESTS_COMMON_H
#define NCCL_TESTS_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <getopt.h>
#include <sys/time.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define CUDACHECK(cmd) do {                         \
  cudaError_t err = cmd;                            \
  if (err != cudaSuccess) {                         \
    fprintf(stderr, "CUDA error %s:%d '%s'\n",      \
        __FILE__,__LINE__,cudaGetErrorString(err));  \
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

#define NCCLCHECK(cmd) do {                         \
  ncclResult_t res = cmd;                           \
  if (res != ncclSuccess) {                         \
    fprintf(stderr, "NCCL error %s:%d '%s'\n",      \
        __FILE__,__LINE__,ncclGetErrorString(res));  \
    exit(EXIT_FAILURE);                              \
  }                                                  \
} while(0)

static double getTime(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

/* Default parameters */
typedef struct {
    size_t minBytes;
    size_t maxBytes;
    int stepFactor;
    int nGpus;
    int nIters;
    int nWarmup;
} TestParams;

static void parseArgs(int argc, char* argv[], TestParams* p) {
    p->minBytes   = 8;
    p->maxBytes   = (size_t)8 * 1024 * 1024 * 1024;  /* 8 GB */
    p->stepFactor = 2;
    p->nGpus      = 0; /* auto-detect */
    p->nIters     = 20;
    p->nWarmup    = 5;

    int opt;
    while ((opt = getopt(argc, argv, "b:e:f:g:n:w:")) != -1) {
        switch (opt) {
            case 'b': p->minBytes   = strtoull(optarg, NULL, 0); break;
            case 'e': {
                char *end;
                p->maxBytes = strtoull(optarg, &end, 0);
                if (*end == 'K' || *end == 'k') p->maxBytes *= 1024;
                else if (*end == 'M' || *end == 'm') p->maxBytes *= 1024ULL*1024;
                else if (*end == 'G' || *end == 'g') p->maxBytes *= 1024ULL*1024*1024;
                break;
            }
            case 'f': p->stepFactor = atoi(optarg); break;
            case 'g': p->nGpus     = atoi(optarg); break;
            case 'n': p->nIters    = atoi(optarg); break;
            case 'w': p->nWarmup   = atoi(optarg); break;
        }
    }
    if (p->nGpus <= 0) {
        CUDACHECK(cudaGetDeviceCount(&p->nGpus));
    }
    if (p->minBytes < 4) p->minBytes = 4;
}

/* Print header matching upstream format so our shell parser works */
static void printHeader(const char* testName) {
    printf("#\n");
    printf("# %s\n", testName);
    printf("#\n");
    printf("# %7s  %12s  %6s  %7s  %6s  %7s  %5s\n",
           "size", "count", "type", "time", "algbw", "busbw", "error");
    printf("# %7s  %12s  %6s  %7s  %6s  %7s  %5s\n",
           "(B)", "(elements)", "", "(us)", "(GB/s)", "(GB/s)", "");
}

/* Print a result row */
static void printRow(size_t bytes, size_t count, double timeUs,
                     double algBw, double busBw) {
    printf("%8zu  %12zu  float  %7.1f  %6.2f  %7.2f  0e+00\n",
           bytes, count, timeUs, algBw, busBw);
}

/* Setup NCCL communicator for single-node multi-GPU */
static void setupNccl(int nGpus, ncclComm_t** comms, cudaStream_t** streams) {
    *comms   = (ncclComm_t*)malloc(nGpus * sizeof(ncclComm_t));
    *streams = (cudaStream_t*)malloc(nGpus * sizeof(cudaStream_t));
    int* devs = (int*)malloc(nGpus * sizeof(int));
    for (int i = 0; i < nGpus; i++) devs[i] = i;
    NCCLCHECK(ncclCommInitAll(*comms, nGpus, devs));
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamCreate(&(*streams)[i]));
    }
    free(devs);
}

static void teardownNccl(int nGpus, ncclComm_t* comms, cudaStream_t* streams) {
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaStreamDestroy(streams[i]));
        ncclCommDestroy(comms[i]);
    }
    free(comms);
    free(streams);
}

#endif /* NCCL_TESTS_COMMON_H */
