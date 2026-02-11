/*************************************************************************
 * all_reduce_perf.cu — AllReduce NCCL benchmark (bundled version)
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

    /* Allocate send/recv buffers on each GPU */
    float** sendbuff = (float**)malloc(nGpus * sizeof(float*));
    float** recvbuff = (float**)malloc(nGpus * sizeof(float*));
    size_t maxCount = params.maxBytes / sizeof(float);

    for (int i = 0; i < nGpus; i++) {
        CUDACHECK(cudaSetDevice(i));
        CUDACHECK(cudaMalloc(&sendbuff[i], params.maxBytes));
        CUDACHECK(cudaMalloc(&recvbuff[i], params.maxBytes));
        CUDACHECK(cudaMemset(sendbuff[i], 1, params.maxBytes));
        CUDACHECK(cudaMemset(recvbuff[i], 0, params.maxBytes));
    }

    printHeader("nccl-tests AllReduce");

    double totalBusBw = 0;
    int rowCount = 0;

    for (size_t bytes = params.minBytes; bytes <= params.maxBytes;
         bytes *= params.stepFactor) {
        size_t count = bytes / sizeof(float);
        if (count < 1) count = 1;

        /* Warmup */
        for (int iter = 0; iter < params.nWarmup; iter++) {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus; i++) {
                NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count,
                    ncclFloat, ncclSum, comms[i], streams[i]));
            }
            NCCLCHECK(ncclGroupEnd());
            for (int i = 0; i < nGpus; i++) {
                CUDACHECK(cudaSetDevice(i));
                CUDACHECK(cudaStreamSynchronize(streams[i]));
            }
        }

        /* Timed iterations */
        double start = getTime();
        for (int iter = 0; iter < params.nIters; iter++) {
            NCCLCHECK(ncclGroupStart());
            for (int i = 0; i < nGpus; i++) {
                NCCLCHECK(ncclAllReduce(sendbuff[i], recvbuff[i], count,
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
        /* AllReduce bus bandwidth: data * 2*(n-1)/n */
        double busBw = algBw * (2.0 * (nGpus - 1) / nGpus);

        printRow(bytes, count, timeUs, algBw, busBw);
        totalBusBw += busBw;
        rowCount++;
    }

    if (rowCount > 0)
        printf("# Avg bus bandwidth: %.2f\n", totalBusBw / rowCount);

    /* Cleanup */
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
