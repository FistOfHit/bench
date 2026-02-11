/*************************************************************************
 * compare.cu — GPU matrix comparison kernel for gpu-burn (bundled version)
 * Based on https://github.com/wilicc/gpu-burn
 *
 * This kernel compares two matrices and counts differences exceeding
 * a threshold — used to detect GPU computation errors under stress.
 *************************************************************************/

extern "C" __global__ void compare(double *C, double *Cref,
                                    int *faultyElems, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N) {
        double diff = C[tid] - Cref[tid];
        if (diff < 0) diff = -diff;
        /* Relative error threshold */
        double ref = Cref[tid];
        if (ref < 0) ref = -ref;
        if (ref < 1e-15) ref = 1e-15;
        if (diff / ref > 1e-6) {
            atomicAdd(faultyElems, 1);
        }
    }
}
