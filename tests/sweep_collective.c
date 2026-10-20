#include "nccl.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    int count = 1024;
    if (argc > 1) count = atoi(argv[1]);
    if (count < 1) count = 1;

    int nr = 1;
    const char *e = getenv("FAKEGPU_NUM_DEVICES");
    if (e && *e) { nr = atoi(e); if (nr < 1) nr = 1; }
    if (cudaGetDeviceCount(&nr) == cudaSuccess) { /* keep env if set */ }

    float *sendbuff = NULL;
    float *recvbuff = NULL;
    if (cudaMalloc((void **)&sendbuff, sizeof(float) * count) != cudaSuccess) return 1;
    if (cudaMalloc((void **)&recvbuff, sizeof(float) * count * (size_t)nr) != cudaSuccess) return 1;

    for (int r = 0; r < nr; r++) {
        ncclComm_t comm;
        if (ncclCommInitRank(&comm, nr, r, 0x1234) != ncclSuccess) return 1;
        ncclGroupStart();
        ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, NULL);
        ncclGroupEnd();
        ncclCommDestroy(comm);
    }

    printf("sweep: count=%d ranks=%d ok\n", count, nr);
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    return 0;
}