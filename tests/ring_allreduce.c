/*
 * ring_allreduce.c — reference NCCL workload compiled against shim/nccl.h.
 *
 * Simulates a multi-rank NCCL ring allreduce in a single process: each "rank"
 * is a fake GPU. Under LD_PRELOAD=fakegpu.so the collectives return random
 * data and never leave the process.
 *
 * Run:  LD_PRELOAD=./libfakegpu.so FAKEGPU_NUM_DEVICES=4 ./build/ring_allreduce
 */
#include "nccl.h"
#include "cuda_runtime.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define COUNT_DEFAULT 1024

static int run(int nranks, int count)
{
    /* Allocate a fake device buffer per "rank" (just one contiguous region is
       fine since there is no real device memory). */
    float *sendbuff = NULL;
    float *recvbuff = NULL;
    if (cudaMalloc((void **)&sendbuff, sizeof(float) * count) != cudaSuccess) {
        fprintf(stderr, "ring_allreduce: cudaMalloc send failed\n");
        return 1;
    }
    if (cudaMalloc((void **)&recvbuff, sizeof(float) * count * (size_t)nranks)
        != cudaSuccess) {
        fprintf(stderr, "ring_allreduce: cudaMalloc recv failed\n");
        return 1;
    }

    /* In a real ring allreduce each rank calls ncclAllReduce on its comm.
       Here we create nranks fake comms and call AllReduce on each. */
    for (int r = 0; r < nranks; r++) {
        ncclComm_t comm;
        if (ncclCommInitRank(&comm, nranks, r, 0x1234) != ncclSuccess) {
            fprintf(stderr, "ring_allreduce: ncclCommInitRank rank=%d failed\n", r);
            return 1;
        }

        ncclGroupStart();
        ncclAllReduce(sendbuff, recvbuff, count, ncclFloat32, ncclSum, comm, NULL);
        ncclGroupEnd();

        ncclCommDestroy(comm);
    }

    printf("ring_allreduce: ranks=%d count=%d -> allreduce ok\n", nranks, count);
    cudaFree(sendbuff);
    cudaFree(recvbuff);
    return 0;
}

int main(int argc, char **argv)
{
    int nr = 1;
    if (cudaGetDeviceCount(&nr) != cudaSuccess) {
        fprintf(stderr, "ring_allreduce: cudaGetDeviceCount failed\n");
        return 1;
    }
    int count = (argc > 1) ? atoi(argv[1]) : 1024;
    printf("ring_allreduce: fake GPUs visible = %d count=%d\n", nr, count);
    return run(nr, count);
}
