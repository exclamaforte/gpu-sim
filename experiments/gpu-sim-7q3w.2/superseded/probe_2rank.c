/* gpu-sim-7q3w.2: two ranks in one process tree, so NCCL has to set up a
 * connection. Run with NCCL_P2P_DISABLE=1 NCCL_SHM_DISABLE=1 to force the
 * transport onto the NET plugin (intra-node would otherwise pick P2P/SHM and
 * never reach accept).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/wait.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define N 8

static int run_rank(int rank, ncclUniqueId uid) {
    cudaSetDevice(0);
    ncclComm_t comm;
    ncclResult_t r = ncclCommInitRank(&comm, 2, uid, rank);
    if (r != ncclSuccess) { printf("RANK%d init failed: %s\n", rank, ncclGetErrorString(r)); return 1; }
    void *s, *d;
    cudaMalloc(&s, N * sizeof(float));
    cudaMalloc(&d, N * sizeof(float));
    float h[N];
    for (int i = 0; i < N; i++) h[i] = 1.0f + rank;
    cudaMemcpy(s, h, N * sizeof(float), cudaMemcpyHostToDevice);
    r = ncclAllReduce(s, d, N, ncclFloat, ncclSum, comm, 0);
    if (r != ncclSuccess) { printf("RANK%d allreduce failed: %s\n", rank, ncclGetErrorString(r)); return 2; }
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("RANK%d sync failed: %s\n", rank, cudaGetErrorString(e)); return 3; }
    float out[N] = {0};
    cudaMemcpy(out, d, N * sizeof(float), cudaMemcpyDeviceToHost);
    printf("RANK%d sum[0]=%.1f (expect 3.0 for 2 ranks of 1+2)\n", rank, out[0]);
    ncclCommDestroy(comm);
    return 0;
}

int main(void) {
    ncclUniqueId uid;
    ncclGetUniqueId(&uid);
    pid_t pid = fork();
    if (pid == 0) { int rc = run_rank(1, uid); _exit(rc); }
    int rc0 = run_rank(0, uid);
    int st = 0; waitpid(pid, &st, 0);
    int rc1 = WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st);
    printf("PROBE rank0=%d rank1=%d\n", rc0, rc1);
    return rc0 ? rc0 : rc1;
}
