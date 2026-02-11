/*************************************************************************
 * reduce_scatter_perf.cu — ReduceScatter NCCL benchmark (bundled version)
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

    float** sendbuff = (float**)malloc(nGpus * sizeof(float*));
    float** recvbuff = (float**)malloc(nGpus * sizeof(float*));

    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&sendbuff[i], params.maxBytes));
        CUDACHECK(cudaMalloc(&recvbuff[i], params.maxBytes));
        CUDACHECK(cudaMemset(sendbuff[i], 1, params.maxBytes));
        CUDACHECK(cudaMemset(recvbuff[i], 0, params.maxBytes));
    }

    printHeader("nccl-tests ReduceScatter");

    double totalBusBw = 0;
    int rowCount = 0;

    for (size_t bytes = params.minBytes; bytes <= params.maxBytes;
         bytes *= params.stepFactor) {
        /* ReduceScatter: each GPU receives count/nGpus elements */
        size_t count = bytes / sizeof(float);
        if (count < (size_t)nGpus) count = nGpus;
        size_t recvCount = count / nGpus;

        for (int iter = 0; iter < params.nWarmup; iter++) {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus; i++) {
                NCCLCHECK(ncclReduceScatter(sendbuff[i], recvbuff[i], recvCount,
                    ncclFloat, ncclSum, comms[i], streams[i]));
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
                NCCLCHECK(ncclReduceScatter(sendbuff[i], recvbuff[i], recvCount,
                    ncclFloat, ncclSum, comms[i], streams[i]));
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
        /* ReduceScatter bus bw: data * (n-1)/n */
        double busBw = algBw * ((double)(nGpus - 1) / nGpus);

        printRow(bytes, count, timeUs, algBw, busBw);
        totalBusBw += busBw;
        rowCount++;
    }

    if (rowCount > 0)
        printf("# Avg bus bandwidth: %.2f\n", totalBusBw / rowCount);

    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaFree(sendbuff[i]));
        CUDACHECK(cudaFree(recvbuff[i]));
    }
    free(sendbuff);
    free(recvbuff);
    teardownNccl(nGpus, comms, streams);
    return 0;
}
