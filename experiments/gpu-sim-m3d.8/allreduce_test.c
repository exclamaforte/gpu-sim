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
    ncclUniqueId uid;
    if (rank == 0) {
        ncclGetUniqueId(&uid);
        FILE* f = fopen(uidfile, "wb");
        fwrite(&uid, sizeof(uid), 1, f);
        fclose(f);
        /* wait for rank1 to read */
        sleep(1);
    } else {
        for (int i = 0; i < 100 && access(uidfile, F_OK) != 0; i++) usleep(100000);
        FILE* f = fopen(uidfile, "rb");
        fread(&uid, sizeof(uid), 1, f);
        fclose(f);
        sleep(1);
    }
    ncclComm_t comm;
    ncclResult_t r = ncclCommInitRank(&comm, nranks, uid, rank);
    if (r != ncclSuccess) { printf("rank %d init failed: %s\n", rank, ncclGetErrorString(r)); return 1; }
    printf("rank %d init ok\n", rank);
    fflush(stdout);

    cudaSetDevice(0);
    int n = 1024;
    float *sendbuf, *recvbuf;
    cudaMalloc((void**)&sendbuf, n*sizeof(float));
    cudaMalloc((void**)&recvbuf, n*sizeof(float));
    float val = (rank == 0) ? 1.0f : 2.0f;
    float h[n];
    for (int i=0;i<n;i++) h[i]=val;
    cudaMemcpy(sendbuf, h, n*sizeof(float), cudaMemcpyHostToDevice);

    r = ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, 0);
    if (r != ncclSuccess) { printf("rank %d allreduce failed: %s\n", rank, ncclGetErrorString(r)); return 2; }
    float out[n];
    cudaMemcpy(out, recvbuf, n*sizeof(float), cudaMemcpyDeviceToHost);
    float expected = 3.0f;
    int ok = 1;
    for (int i=0;i<n;i++) if (out[i] != expected) { ok=0; break; }
    printf("rank %d allreduce sum[0]=%.1f (expect %.1f) %s\n", rank, out[0], expected, ok?"OK":"FAIL");
    fflush(stdout);

    ncclCommDestroy(comm);
    cudaFree(sendbuf); cudaFree(recvbuf);
    return ok ? 0 : 3;
}