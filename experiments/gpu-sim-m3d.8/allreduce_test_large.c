#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <nccl.h>
#include <cuda_runtime.h>

int main(int argc, char** argv) {
    int rank = atoi(argv[1]);
    int nranks = atoi(argv[2]);
    const char* uidfile = argv[3];
    /* argv[4] (optional): file to write our own pid into, so the harness can
       kill THIS rank rather than a `timeout` wrapper. */
    if (argc > 4 && argv[4] && *argv[4]) {
        FILE* pf = fopen(argv[4], "w");
        if (pf) { fprintf(pf, "%d\n", (int)getpid()); fclose(pf); }
    }
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
    int n = 1 << 24;
    float *sendbuf, *recvbuf;
    cudaMalloc((void**)&sendbuf, n*sizeof(float));
    cudaMalloc((void**)&recvbuf, n*sizeof(float));
    float val = (rank == 0) ? 1.0f : 2.0f;
    /* fill only first few to save time */
    float h[1024];
    for (int i=0;i<1024;i++) h[i]=val;
    cudaMemcpy(sendbuf, h, 1024*sizeof(float), cudaMemcpyHostToDevice);

    /* Several iterations so the harness has a long, deterministic window in
       which the rank is genuinely inside the collective when it is killed. */
    int iters = 4;
    if (argc > 5) iters = atoi(argv[5]);
    printf("rank %d allreduce start (%d iters)\n", rank, iters);
    fflush(stdout);
    for (int it = 0; it < iters; it++) {
        r = ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, 0);
        if (r != ncclSuccess) { printf("rank %d allreduce failed: %s\n", rank, ncclGetErrorString(r)); return 2; }
    }
    /* ncclAllReduce is asynchronous: completion only happens at the sync, so
       this marker is printed while the collective is genuinely in flight. */
    printf("rank %d allreduce enqueued\n", rank);
    fflush(stdout);
    cudaDeviceSynchronize();
    printf("rank %d allreduce OK\n", rank);
    fflush(stdout);

    ncclCommDestroy(comm);
    cudaFree(sendbuf); cudaFree(recvbuf);
    return 0;
}