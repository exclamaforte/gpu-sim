/* allreduce_test_selfdesc.c — self-describing payload allreduce test.
 *
 * Rank 0 fills h[i] = (float)i (global index). Rank 1 fills h[i] = 0.
 * After SUM allreduce, expected[i] = i. Any misplaced chunk shows up as
 * positions containing values from a different range, directly revealing
 * the source->destination permutation.
 *
 * Usage: allreduce_test_selfdesc <rank> <nranks> <uidfile> <n_floats>
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
    printf("rank %d init ok n=%d (self-describing)\n", rank, n);
    fflush(stdout);

    cudaSetDevice(0);
    float *sendbuf, *recvbuf;
    cudaMalloc((void**)&sendbuf, n*sizeof(float));
    cudaMalloc((void**)&recvbuf, n*sizeof(float));
    float *h = (float*)malloc(n*sizeof(float));
    if (rank == 0) {
        for (int i = 0; i < n; i++) h[i] = (float)i;
    } else {
        for (int i = 0; i < n; i++) h[i] = 0.0f;
    }
    cudaMemcpy(sendbuf, h, n*sizeof(float), cudaMemcpyHostToDevice);

    r = ncclAllReduce(sendbuf, recvbuf, n, ncclFloat, ncclSum, comm, 0);
    if (r != ncclSuccess) { printf("rank %d allreduce failed: %s\n", rank, ncclGetErrorString(r)); return 2; }
    printf("rank %d allreduce done\n", rank);
    fflush(stdout);
    cudaDeviceSynchronize();
    cudaMemcpy(h, recvbuf, n*sizeof(float), cudaMemcpyDeviceToHost);

    /* Self-describing check: expected[i] = i (rank0 data + rank1 zeros) */
    int bad = 0, first = -1; float got = 0;
    /* Print permutation: for each position, what source index is there? */
    /* Only print mismatches and a summary */
    for (int i = 0; i < n; i++) {
        if (fabsf(h[i] - (float)i) > 0.5f) {
            if (first < 0) { first = i; got = h[i]; }
            bad++;
        }
    }
    if (bad == 0) {
        printf("rank %d VALUE OK (all %d elements == i)\n", rank, n);
    } else {
        printf("rank %d VALUE MISMATCH: %d/%d wrong, first idx=%d got=%.1f (expect %d)\n",
               rank, bad, n, first, got, first);
        /* Print source-offset mapping for first 20 mismatches */
        int printed = 0;
        for (int i = 0; i < n && printed < 20; i++) {
            if (fabsf(h[i] - (float)i) > 0.5f) {
                printf("  pos=%d got=%.1f (source offset %d)\n", i, h[i], (int)lroundf(h[i]));
                printed++;
            }
        }
        /* Print chunk-level summary: 32768-byte chunks = 8192 floats */
        int chunk_size = 8192;
        int nchunks = (n + chunk_size - 1) / chunk_size;
        printf("  chunk summary (chunk_size=%d floats):\n", chunk_size);
        for (int c = 0; c < nchunks; c++) {
            int start = c * chunk_size;
            int end = start + chunk_size;
            if (end > n) end = n;
            /* what source offset is at position start? */
            int src = (int)lroundf(h[start]);
            printf("    chunk %d: pos %d..%d contains source offset %d (%s)\n",
                   c, start, end-1, src,
                   (src == start) ? "OK" : "MISPLACED");
        }
    }
    fflush(stdout);
    printf("META: size_bytes=%zu n=%d nranks=%d fill0=i fill1=0 expected=i\n",
           (size_t)n * 4, n, nranks);
    fflush(stdout);
    usleep(5000000);
    _exit(0);
}
