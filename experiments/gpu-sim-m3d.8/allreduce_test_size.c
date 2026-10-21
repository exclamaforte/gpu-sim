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
    int n = (argc > 4) ? atoi(argv[4]) : 1024;
    if (n < 1) n = 1024;
    ncclUniqueId uid;
    if (rank == 0) {
        ncclGetUniqueId(&uid);
        FILE* f = fopen(uidfile, "wb");
        fwrite(&uid, sizeof(uid), 1, f);
        fclose(f);
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
    printf("rank %d init ok n=%d\n", rank, n);
    fflush(stdout);

    cudaSetDevice(0);
    float *sendbuf, *recvbuf;
    cudaMalloc((void**)&sendbuf, n*sizeof(float));
    cudaMalloc((void**)&recvbuf, n*sizeof(float));
    float val = (rank == 0) ? 1.0f : 2.0f;
    /* fill the ENTIRE buffer so value correctness can be checked */
    float *h = (float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++) h[i] = val;
    cudaMemcpy(sendbuf, h, n*sizeof(float), cudaMemcpyHostToDevice);

    r = ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, 0);
    if (r != ncclSuccess) { printf("rank %d allreduce failed: %s\n", rank, ncclGetErrorString(r)); return 2; }
    printf("rank %d allreduce done\n", rank);
    fflush(stdout);
    /* value correctness check */
    cudaMemcpy(h, recvbuf, n*sizeof(float), cudaMemcpyDeviceToHost);
    int bad=0, first=-1; float got=0;
    for (int i=0;i<n;i++){ if (h[i]!=3.0f){ if(first<0){first=i;got=h[i];} bad++; } }
    if (bad==0) printf("rank %d VALUE OK (all %d elements == 3.0)\n", rank, n);
    else printf("rank %d VALUE MISMATCH: %d/%d wrong, first idx=%d got=%f (expect 3.0)\n", rank, bad, n, first, got);
    fflush(stdout);
    /* let async isends flush to trace files */
    usleep(5000000);
    _exit(0);
}
