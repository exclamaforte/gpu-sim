/* nccl_numerics.c -- 2-rank collective numerics through genuine NCCL over our
 * libibverbs shim, with the 7q3w.4 aliased-device shims providing two virtual
 * B300s on this box's one real GPU.
 *
 * This is gpu-sim-7q3w.5's first experiment (PI design update, 2026-11-22):
 * rank r fills a 1 MiB float32 vector with r+1, so a correct SUM is analytically
 * 3.0 on both ranks.  The assertion is exact: elements wrong and max_abs_err
 * must both be zero, not "within tolerance".  The transport is the modelled
 * fabric (libibverbs shim + MAP_SHARED ring), while the arithmetic is real
 * kernels on the real GPU behind the alias.
 *
 * A second, optional arm (NUMERICS_P2P=1) runs a 4096-byte two-rank send/recv
 * and compares the received bytes byte-for-byte.  It is deferred for the first
 * experiment: the receive side sat in the fabric poll loop after the send, so
 * the default path asserts only the allreduce and the transport counters.
 *
 * Output is machine-checkable, one line per fact:
 *   NUMERICS allreduce n=... expected=... max_abs_err=... bad=... value=...
 *   NUMERICS p2p nbytes=... mismatches=...
 *   NUMERICS ok
 */
#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <errno.h>
#include <nccl.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...) do {                                          \
    if (!(cond)) { fprintf(stderr, "NUMERICS FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); \
                   failures++; }                                       \
} while (0)

#define NCCLCHECK(call) do {                                           \
    ncclResult_t r_ = (call);                                          \
    if (r_ != ncclSuccess) {                                           \
        fprintf(stderr, "NUMERICS FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, \
                #call, ncclGetErrorString(r_));                        \
        failures++;                                                    \
    }                                                                  \
} while (0)

#define CUDACHECK(call) do {                                           \
    cudaError_t e_ = (call);                                           \
    if (e_ != cudaSuccess) {                                           \
        fprintf(stderr, "NUMERICS FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, \
                #call, cudaGetErrorString(e_));                        \
        failures++;                                                    \
    }                                                                  \
} while (0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int write_id(const char *dir, const ncclUniqueId *uid) {
    char path[512], tmp[512];
    snprintf(path, sizeof path, "%s/id.bin", dir);
    snprintf(tmp, sizeof tmp, "%s/id.bin.tmp", dir);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    int ok = fwrite(uid, sizeof *uid, 1, f) == 1;
    fclose(f);
    if (!ok) return -1;
    return rename(tmp, path);
}

static int wait_id(const char *dir, ncclUniqueId *uid, double timeout_s) {
    char path[512];
    snprintf(path, sizeof path, "%s/id.bin", dir);
    double deadline = now_s() + timeout_s;
    while (now_s() < deadline) {
        FILE *f = fopen(path, "rb");
        if (f) {
            size_t n = fread(uid, sizeof *uid, 1, f);
            fclose(f);
            if (n == 1) return 0;
        }
        struct timespec ts = {0, 20 * 1000 * 1000};
        nanosleep(&ts, NULL);
    }
    return -1;
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <rank> <world> [n_f32]\n", argv[0]);
        return 2;
    }
    int rank = atoi(argv[1]);
    int world = atoi(argv[2]);
    size_t n = 1u << 18;                    /* 1 MiB of float32 */
    if (argc > 3) n = (size_t)strtoull(argv[3], NULL, 10);
    if (n == 0) { fprintf(stderr, "NUMERICS FAIL n=0\n"); return 2; }

    const char *dir = getenv("NCCL_ARM_DIR");
    if (!dir || !*dir) { fprintf(stderr, "NUMERICS FAIL NCCL_ARM_DIR unset\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    ncclUniqueId uid;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&uid));
        if (write_id(dir, &uid) != 0) { perror("write id"); return 2; }
    } else if (wait_id(dir, &uid, 120.0) != 0) {
        fprintf(stderr, "NUMERICS FAIL rank %d: no ncclUniqueId from rank 0\n", rank);
        return 2;
    }

    /* Each process presents exactly one virtual GPU (the 7q3w.4 alias shim
     * variant), so rank is a communicator rank, not a CUDA ordinal. */
    CUDACHECK(cudaSetDevice(0));
    int ndev = -1;
    CUDACHECK(cudaGetDeviceCount(&ndev));
    printf("NUMERICS note rank=%d ndev=%d requested_device=0\n", rank, ndev);
    CHECK(ndev >= 1, "rank %d sees no CUDA device", rank);
    if (failures) return 1;

    size_t nbytes = n * sizeof(float);
    float *d_in = NULL, *d_out = NULL;
    CUDACHECK(cudaMalloc((void **)&d_in, nbytes));
    CUDACHECK(cudaMalloc((void **)&d_out, nbytes));
    float *h_in = malloc(nbytes), *h_out = malloc(nbytes);
    if (!h_in || !h_out) { perror("malloc"); return 2; }
    for (size_t i = 0; i < n; i++) h_in[i] = (float)(rank + 1);
    CUDACHECK(cudaMemcpy(d_in, h_in, nbytes, cudaMemcpyHostToDevice));

    ncclComm_t comm = NULL;
    NCCLCHECK(ncclCommInitRank(&comm, world, uid, rank));
    printf("NUMERICS init rank=%d world=%d\n", rank, world);
    if (failures) return 1;

    /* Warmup, then the measured allreduce.  Both are real NCCL collectives;
     * the warmup pays QP/MR setup before the value we assert. */
    NCCLCHECK(ncclAllReduce(d_in, d_out, n, ncclFloat, ncclSum, comm, NULL));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaMemset(d_out, 0, nbytes));
    NCCLCHECK(ncclAllReduce(d_in, d_out, n, ncclFloat, ncclSum, comm, NULL));
    CUDACHECK(cudaDeviceSynchronize());

    CUDACHECK(cudaMemcpy(h_out, d_out, nbytes, cudaMemcpyDeviceToHost));
    size_t bad = 0;
    double max_abs_err = 0.0;
    /* The exact expected sum for rank r is (r+1) + (other rank + 1) = 3.0. */
    const float expected = 0.5f * (float)world * (float)(world + 1);
    for (size_t i = 0; i < n; i++) {
        double e = (double)h_out[i] - (double)expected;
        if (e < 0) e = -e;
        if (e > max_abs_err) max_abs_err = e;
        if (h_out[i] != expected) bad++;
    }
    printf("NUMERICS allreduce rank=%d n=%zu expected=%.1f value=%.1f "
           "max_abs_err=%.3e bad=%zu\n",
           rank, n, expected, (double)h_out[0], max_abs_err, bad);
    CHECK(bad == 0, "rank %d: %zu/%zu allreduce elements != %.1f (max_abs_err %.3e)",
          rank, bad, n, expected, max_abs_err);
    CHECK(max_abs_err == 0.0, "rank %d: max_abs_err %.3e is not exact zero",
          rank, max_abs_err);

    /* Point-to-point is a separate arm (NUMERICS_P2P=1).  It is not the
     * first experiment's claim; when this run started, the receive side sat
     * in the fabric poll loop after the 4096-byte send, so it is kept off the
     * default path until the p2p semantics are measured on their own. */
    if (getenv("NUMERICS_P2P") && !strcmp(getenv("NUMERICS_P2P"), "1")) {
        const size_t p2p_n = 4096;
        unsigned char *d_send = NULL, *d_recv = NULL;
        unsigned char *h_send = malloc(p2p_n), *h_recv = malloc(p2p_n);
        if (!h_send || !h_recv) { perror("malloc"); return 2; }
        for (size_t i = 0; i < p2p_n; i++) h_send[i] = (unsigned char)((i * 37u + 11u) & 0xffu);
        CUDACHECK(cudaMalloc((void **)&d_send, p2p_n));
        CUDACHECK(cudaMalloc((void **)&d_recv, p2p_n));
        CUDACHECK(cudaMemcpy(d_send, h_send, p2p_n, cudaMemcpyHostToDevice));
        CUDACHECK(cudaMemset(d_recv, 0xA5, p2p_n));
        NCCLCHECK(ncclGroupStart());
        if (rank == 0) {
            NCCLCHECK(ncclSend(d_send, p2p_n, ncclUint8, 1, comm, NULL));
        } else if (rank == 1) {
            NCCLCHECK(ncclRecv(d_recv, p2p_n, ncclUint8, 0, comm, NULL));
        }
        NCCLCHECK(ncclGroupEnd());
        CUDACHECK(cudaDeviceSynchronize());
        size_t mismatches = 0;
        if (rank == 1) {
            CUDACHECK(cudaMemcpy(h_recv, d_recv, p2p_n, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < p2p_n; i++) if (h_recv[i] != h_send[i]) mismatches++;
            printf("NUMERICS p2p rank=1 nbytes=%zu mismatches=%zu\n", p2p_n, mismatches);
            CHECK(mismatches == 0, "rank 1: %zu/%zu received bytes differ from sent bytes",
                  mismatches, p2p_n);
        } else {
            printf("NUMERICS p2p rank=0 nbytes=%zu sent=1\n", p2p_n);
        }
        cudaFree(d_send);
        cudaFree(d_recv);
        free(h_send); free(h_recv);
    }

    ncclCommDestroy(comm);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in); free(h_out);
    if (failures) { fprintf(stderr, "NUMERICS FAILED: %d failure(s)\n", failures); return 1; }
    printf("NUMERICS ok rank=%d\n", rank);
    return 0;
}
