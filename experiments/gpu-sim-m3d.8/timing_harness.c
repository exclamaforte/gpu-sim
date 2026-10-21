/* timing_harness.c — per-iteration collective timing harness for gpu-sim.
 * Usage: timing_harness <rank> <nranks> <uidfile> <iters>
 * Times one ncclAllReduce per iteration, writes per-iteration durations to
 * the file named by GPUSIM_TIMING (one line per iteration per rank).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>
#include <nccl.h>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    int rank = atoi(argv[1]);
    int nranks = atoi(argv[2]);
    const char* uidfile = argv[3];
    int iters = (argc > 4) ? atoi(argv[4]) : 10;
    ncclUniqueId uid;
    if (rank == 0) {
        ncclGetUniqueId(&uid);
        FILE* f = fopen(uidfile, "wb");
        fwrite(&uid, sizeof(uid), 1, f);
        fclose(f);
        usleep(200000);
    } else {
        for (int i = 0; i < 100 && access(uidfile, F_OK) != 0; i++) usleep(100000);
        FILE* f = fopen(uidfile, "rb");
        fread(&uid, sizeof(uid), 1, f);
        fclose(f);
        usleep(200000);
    }
    ncclComm_t comm;
    ncclResult_t r = ncclCommInitRank(&comm, nranks, uid, rank);
    if (r != ncclSuccess) { printf("rank %d init failed: %s\n", rank, ncclGetErrorString(r)); return 1; }
    printf("rank %d init ok\n", rank);
    fflush(stdout);

    cudaSetDevice(0);
    int n = (argc > 5) ? atoi(argv[5]) : 1024;
    float *sendbuf, *recvbuf;
    cudaMalloc((void**)&sendbuf, n*sizeof(float));
    cudaMalloc((void**)&recvbuf, n*sizeof(float));
    float val = (rank == 0) ? 1.0f : 2.0f;
    float *h = malloc(n*sizeof(float));
    for (int i = 0; i < n; i++) h[i] = val;
    cudaMemcpy(sendbuf, h, n*sizeof(float), cudaMemcpyHostToDevice);
    free(h);

    const char* tf = getenv("GPUSIM_TIMING");
    /* warmup: 1 iteration not timed */
    r = ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, 0);
    if (r == ncclSuccess) cudaDeviceSynchronize();

    for (int iter = 0; iter < iters; iter++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        r = ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, 0);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        if (r != ncclSuccess) { printf("rank %d iter %d failed: %s\n", rank, iter, ncclGetErrorString(r)); break; }
        cudaDeviceSynchronize();
        double dt = (t1.tv_sec - t0.tv_sec) * 1e6 + (t1.tv_nsec - t0.tv_nsec) / 1e3;
        printf("rank %d iter=%d enq=%.1f us\n", rank, iter, dt);
        fflush(stdout);
        if (tf) {
            FILE* f = fopen(tf, "a");
            if (f) { fprintf(f, "rank=%d iter=%d enq_us=%.1f\n", rank, iter, dt); fclose(f); }
        }
    }

    ncclCommDestroy(comm);
    cudaFree(sendbuf); cudaFree(recvbuf);
    return 0;
}
