/* gpu-sim-465.1 phase 0 go/no-go, driven directly.
 *
 * The first probe used torch but never issued a COLLECTIVE, so the net-plugin
 * init path was never entered and the run reported success while testing
 * nothing -- a green result from a program that did not do the thing. This one
 * calls ncclAllReduce for real, so NCCL must bring up its transports.
 */
#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>

int main(int argc, char **argv) {
    static ncclComm_t comm;
    /* ncclCommInitRank(comm, nranks, commId, rank) -- the third argument is a
     * ncclUniqueId, not the rank. My first version passed the rank there and the
     * compiler said so, which is a better outcome than a probe that silently
     * never brought a communicator up. */
    ncclUniqueId uid;
    ncclGetUniqueId(&uid);
    ncclResult_t r = ncclCommInitRank(&comm, 1, uid, 0);
    if (r != ncclSuccess) {
        printf("PROBE ncclCommInitRank failed: %s\n", ncclGetErrorString(r));
        return 1;
    }
    printf("PROBE init ok\n");
    cudaSetDevice(0);
    void *s, *d;
    cudaMalloc(&s, 8 * sizeof(float));
    cudaMalloc(&d, 8 * sizeof(float));
    float h[8] = {1,1,1,1,1,1,1,1};
    cudaMemcpy(s, h, 8 * sizeof(float), cudaMemcpyHostToDevice);
    r = ncclAllReduce(s, d, 8, ncclFloat, ncclSum, comm, 0);
    if (r != ncclSuccess) {
        printf("PROBE ncclAllReduce failed: %s\n", ncclGetErrorString(r));
        return 2;
    }
    float out[8] = {0};
    cudaMemcpy(out, d, 8 * sizeof(float), cudaMemcpyDeviceToHost);
    printf("PROBE allreduce sum[0]=%.1f (expect 1.0 for world=1)\n", out[0]);
    ncclCommDestroy(comm);
    return 0;
}
