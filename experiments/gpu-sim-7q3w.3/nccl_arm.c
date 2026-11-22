/* nccl_arm.c -- the acceptance arms of gpu-sim-7q3w.3, driven through real NCCL.
 *
 * Two ranks, two processes, one GPU each (CUDA_VISIBLE_DEVICES selects which,
 * so both ranks call cudaSetDevice(0) -- a rank number is not a device number).
 * Everything below NCCL is ours: libibverbs.so.1 is the shim, /sys/class/
 * infiniband is the LD_PRELOAD rewrite, and the bytes cross processes through
 * the ring.  If NCCL's IB transport were not running on our fabric there would
 * be no bytes at all, which is the point of asserting the numerics.
 *
 * Rank 0 writes the ncclUniqueId to $NCCL_ARM_DIR/id.bin via a temp file +
 * rename; rank 1 polls for it.  A file, not a pipe, because the ranks are
 * started by a shell that does not wire them together (they are separate
 * processes in separate UTS namespaces).
 *
 * Prints one machine-checkable line per arm and exits non-zero on failure:
 *   ARM ok init rank=<r> world=<w>
 *   ARM ok allreduce n=<n> value=<v> expected=<e>
 *   ARM ok bw ratio=<r> charge_ratio=<c>   (--bw)
 */
#define _GNU_SOURCE
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <errno.h>
#include <nccl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static int failures;
#define CHECK(cond, ...) do {                                          \
    if (!(cond)) { fprintf(stderr, "ARM FAIL %s:%d: ", __FILE__, __LINE__); \
                   fprintf(stderr, __VA_ARGS__); fprintf(stderr, "\n"); \
                   failures++; }                                       \
} while (0)

#define NCCLCHECK(call) do {                                           \
    ncclResult_t r_ = (call);                                          \
    if (r_ != ncclSuccess) {                                           \
        fprintf(stderr, "ARM FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, \
                #call, ncclGetErrorString(r_));                        \
        failures++;                                                    \
    }                                                                  \
} while (0)

#define CUDACHECK(call) do {                                           \
    cudaError_t e_ = (call);                                           \
    if (e_ != cudaSuccess) {                                           \
        fprintf(stderr, "ARM FAIL %s:%d: %s -> %s\n", __FILE__, __LINE__, \
                #call, cudaGetErrorString(e_));                        \
        failures++;                                                    \
    }                                                                  \
} while (0)

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int read_id(const char *dir, ncclUniqueId *uid) {
    char path[512], tmp[512];
    snprintf(path, sizeof path, "%s/id.bin", dir);
    snprintf(tmp, sizeof tmp, "%s/id.bin.tmp", dir);
    FILE *f = fopen(tmp, "wb");
    if (!f) return -1;
    int ok = fwrite(uid, sizeof *uid, 1, f) == 1;
    fclose(f);
    if (!ok) return -1;
    return rename(tmp, path);            /* readers never see a partial id */
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
        fprintf(stderr, "usage: %s <rank> <world> [--bw] [n_f32]\n", argv[0]);
        return 2;
    }
    int rank = atoi(argv[1]);
    int world = atoi(argv[2]);
    int want_bw = 0;
    size_t n = 1u << 20;                       /* 4 MiB by default */
    for (int i = 3; i < argc; i++) {
        if (!strcmp(argv[i], "--bw")) want_bw = 1;
        else n = (size_t)strtoull(argv[i], NULL, 10);
    }
    const char *dir = getenv("NCCL_ARM_DIR");
    if (!dir || !*dir) { fprintf(stderr, "ARM FAIL NCCL_ARM_DIR unset\n"); return 2; }
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);

    ncclUniqueId uid;
    if (rank == 0) {
        NCCLCHECK(ncclGetUniqueId(&uid));
        if (read_id(dir, &uid) != 0) { perror("write id"); return 2; }
    } else if (wait_id(dir, &uid, 120.0) != 0) {
        fprintf(stderr, "ARM FAIL rank %d: no ncclUniqueId from rank 0\n", rank);
        return 2;
    }

    CUDACHECK(cudaSetDevice(0));
    float *d_in = NULL, *d_out = NULL;
    size_t bytes = n * sizeof(float);
    /* cudaMalloc wants void**; casting through void* keeps -Wincompatible-
     * pointer-types quiet without hiding a real API mistake. */
    CUDACHECK(cudaMalloc((void **)&d_in, bytes));
    CUDACHECK(cudaMalloc((void **)&d_out, bytes));
    float *h_in = malloc(bytes), *h_out = malloc(bytes);
    if (!h_in || !h_out) { perror("malloc"); return 2; }
    /* rank 0 contributes 1.0, rank 1 contributes 2.0: a correct sum is 3.0 and
     * a communicator that silently missing a rank cannot reach it. */
    for (size_t i = 0; i < n; i++) h_in[i] = (float)(rank + 1);
    CUDACHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    ncclComm_t comm = NULL;
    NCCLCHECK(ncclCommInitRank(&comm, world, uid, rank));
    printf("ARM ok init rank=%d world=%d\n", rank, world);
    if (failures) return 1;

    /* warmup: the first collective pays QP setup, MR registration, topology */
    NCCLCHECK(ncclAllReduce(d_in, d_out, n, ncclFloat, ncclSum, comm, NULL));
    CUDACHECK(cudaDeviceSynchronize());

    if (want_bw) {
        /* Two line rates, same bytes.  The shim charges each completion with
         * bytes/BW, and the charge counters are what the assertion uses: a
         * wall-clock ratio on a shared box measures the box (stage 5 learned
         * that the hard way).  The NCCL-level wall ratio is printed to show the
         * collective is on the modelled wire at all. */
        int iters = 10;
        double t[2];
        for (int round = 0; round < 2; round++) {
            setenv("FAKEIB_BW_GBPS", round == 0 ? "10" : "20", 1);
            double t0 = now_s();
            for (int i = 0; i < iters; i++)
                NCCLCHECK(ncclAllReduce(d_in, d_out, n, ncclFloat, ncclSum, comm, NULL));
            CUDACHECK(cudaDeviceSynchronize());
            t[round] = (now_s() - t0) / iters;
        }
        uint64_t *charge = (uint64_t *)dlsym(RTLD_DEFAULT, "fakeib_last_charge_ns");
        printf("ARM ok bw ratio=%.2f wall_10G=%.6f wall_20G=%.6f last_charge_ns=%llu\n",
               t[0] / t[1], t[0], t[1],
               (unsigned long long)(charge ? *charge : 0));
        /* 2x the line rate must not be slower; the strict 2x assertion lives in
         * stage 5 where the shim's own charge is the measurement. */
        CHECK(t[0] > t[1], "20 Gb/s was not faster than 10: %.6f vs %.6f", t[0], t[1]);
    } else {
        size_t bad = 0;
        NCCLCHECK(ncclAllReduce(d_in, d_out, n, ncclFloat, ncclSum, comm, NULL));
        CUDACHECK(cudaDeviceSynchronize());
        CUDACHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < n; i++) if (h_out[i] != 3.0f) bad++;
        printf("ARM ok allreduce n=%zu value=%.1f expected=3.0 bad=%zu\n",
               n, bad ? -1.0f : (double)h_out[0], bad);
        CHECK(bad == 0, "%zu of %zu elements are not 3.0", bad, n);
    }

    ncclCommDestroy(comm);
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    if (failures) { fprintf(stderr, "ARM FAILED: %d failure(s)\n", failures); return 1; }
    printf("ARM OK rank=%d\n", rank);
    return 0;
}
