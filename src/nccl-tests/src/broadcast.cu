/*************************************************************************
 * broadcast_perf.cu — Broadcast NCCL benchmark (bundled version)
 * Based on https://github.com/NVIDIA/nccl-tests
 *************************************************************************/
#include "common.h"

int main(int argc, char* argv[]) {
    TestParams params;
    parseArgs(argc, argv, &params);

    int nGpus = params.nGpus;
    ncclComm_t* comms;
    cudaStream_t* streams;
    setupNccl(nGpus, &comms, &streams);

    float** buff = (float**)malloc(nGpus * sizeof(float*));
    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&buff[i], params.maxBytes));
        CUDACHECK(cudaMemset(buff[i], i == 0 ? 1 : 0, params.maxBytes));
    }

    printHeader("nccl-tests Broadcast");

    double totalBusBw = 0;
    int rowCount = 0;
    int root = 0;

    for (size_t bytes = params.minBytes; bytes <= params.maxBytes;
         bytes *= params.stepFactor) {
        size_t count = bytes / sizeof(float);
        if (count < 1) count = 1;

        for (int iter = 0; iter < params.nWarmup; iter++) {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus; i++) {
                NCCLCHECK(ncclBroadcast(buff[i], buff[i], count,
                    ncclFloat, root, comms[i], streams[i]));
            }
            NCCLCHECK(ncclGroupEnd());
            for (int i = 0; i < nGpus; i++) {
                CUDACHECK(cudaSetDevice(i));
                CUDACHECK(cudaStreamSynchronize(streams[i]));
            }
        }

        double start = getTime();
        for (int iter = 0; iter < params.nIters; iter++) {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus; i++) {
                NCCLCHECK(ncclBroadcast(buff[i], buff[i], count,
                    ncclFloat, root, comms[i], streams[i]));
            }
            NCCLCHECK(ncclGroupEnd());
            for (int i = 0; i < nGpus; i++) {
                CUDACHECK(cudaSetDevice(i));
                CUDACHECK(cudaStreamSynchronize(streams[i]));
            }
        }
        double elapsed = getTime() - start;
        double timeUs = (elapsed / params.nIters) * 1e6;
        double algBw = (double)bytes / (elapsed / params.nIters) / 1e9;
        /* Broadcast bus bw: data * 1 (ring algorithm) */
        double busBw = algBw;

        printRow(bytes, count, timeUs, algBw, busBw);
        totalBusBw += busBw;
        rowCount++;
    }

    if (rowCount > 0)
        printf("# Avg bus bandwidth: %.2f\n", totalBusBw / rowCount);

    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(buff[i]));
    }
    free(buff);
    teardownNccl(nGpus, comms, streams);
    return 0;
}
