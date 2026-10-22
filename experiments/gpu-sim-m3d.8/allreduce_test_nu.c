/* allreduce_test_nu.c — non-uniform fill allreduce value-correctness test.
 *
 * Unlike allreduce_test_size.c (uniform fill, blind to shifted copies), this
 * fills h[i] = (rank==0 ? 1.0 : 2.0) + i*0.001.  After SUM allreduce the
 * expected value at index i is 3.0 + i*0.002.  A shifted or truncated copy
 * produces a non-zero residual at the first differing index, which is decoded
 * and printed so the log states the offset rather than only pass/fail.
 *
 * Usage: allreduce_test_nu <rank> <nranks> <uidfile> <n_floats>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
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
    printf("rank %d init ok n=%d (non-uniform fill)\n", rank, n);
    fflush(stdout);

    cudaSetDevice(0);
    float *sendbuf, *recvbuf;
    cudaMalloc((void**)&sendbuf, n*sizeof(float));
    cudaMalloc((void**)&recvbuf, n*sizeof(float));
    float base = (rank == 0) ? 1.0f : 2.0f;
    float *h = (float*)malloc(n*sizeof(float));
    for (int i = 0; i < n; i++) h[i] = base + i * 0.001f;
    cudaMemcpy(sendbuf, h, n*sizeof(float), cudaMemcpyHostToDevice);

    r = ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, 0);
    if (r != ncclSuccess) { printf("rank %d allreduce failed: %s\n", rank, ncclGetErrorString(r)); return 2; }
    printf("rank %d allreduce done\n", rank);
    fflush(stdout);
    cudaDeviceSynchronize();
    cudaMemcpy(h, recvbuf, n*sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0, first = -1; float got = 0, expected0 = 0;
    for (int i = 0; i < n; i++) {
        float expected = 3.0f + i * 0.002f;
        if (fabsf(h[i] - expected) > 1e-4f) {
            if (first < 0) { first = i; got = h[i]; expected0 = expected; }
            bad++;
        }
    }
    if (bad == 0) {
        printf("rank %d VALUE OK (all %d elements match 3.0 + i*0.002)\n", rank, n);
    } else {
        /* decode offset: residual / 0.001 gives how many elements shifted */
        float residual = got - expected0;
        int offset_elems = (int)lroundf(residual / 0.001f);
        /* check if bad count is a clean fraction of n */
        const char *frac = "none";
        if (bad == n) frac = "n (all)";
        else if (bad == n/2) frac = "n/2";
        else if (bad == n/4) frac = "n/4";
        else if (bad == n/8) frac = "n/8";
        else if (bad == n/16) frac = "n/16";
        printf("rank %d VALUE MISMATCH: %d/%d wrong, first idx=%d got=%.6f expect=%.6f residual=%.6f offset_elems=%d bad_fraction=%s\n",
               rank, bad, n, first, got, expected0, residual, offset_elems, frac);
    }
    fflush(stdout);
    printf("META: size_bytes=%zu n=%d nranks=%d protocol=%s fill0=1.0+i*0.001 fill1=2.0+i*0.001 expected=3.0+i*0.002\n",
           (size_t)n * 4, n, nranks, getenv("NCCL_PROTO") ? getenv("NCCL_PROTO") : "(default)");
    fflush(stdout);
    usleep(5000000);
    _exit(0);
}
