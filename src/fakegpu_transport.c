/* Data-correct all_reduce through /dev/shm (gpu-sim-n5h).
 *
 * The constraint that decides this implementation: fake NCCL is handed DEVICE
 * pointers (torch allocates on the GPU). Host code cannot read or write them
 * directly -- that is precisely why fg_fill on a device pointer SIGSEGVs.
 * So the path is:
 *
 *   cudaDeviceSynchronize -> cudaMemcpy(d2h) -> write /dev/shm slot ->
 *   barrier on an atomic counter -> read all slots -> reduce on host ->
 *   cudaMemcpy(h2d) into recvbuff
 *
 * The CUDA runtime here is REAL (only nccl symbols are preloaded), so
 * cudaMemcpy/cudaDeviceSynchronize resolve from the process's own libcudart
 * via dlsym(RTLD_DEFAULT, ...). This is correctness infrastructure, not
 * speed: the simulated-timing path stays the default for traffic measurement.
 *
 * GATE: transport runs iff FAKEGPU_NCCL_SIMULATE_ONLY=="0". Unset or any other
 * value (the default) keeps today's simulate-only behaviour, so every
 * published traffic number stays reproducible. (The bead text says "UNSET",
 * but unset-means-transport would flip the default for every existing sweep;
 * unset-means-simulate is the reading that keeps the default simulate-only
 * promise. Flagged for the PI; changing it is one strcmp.)
 *
 * KEY: /dev/shm/fgcoll_nr<nr>_c<count>_d<dtype>_o<op>_k<k>, where k is a
 * per-process collective counter. Ranks call collectives in the same order,
 * so the k-th transported collective agrees across ranks without rendezvous
 * (the same trick as the uniqueId fix). nranks+count+dtype+op are in the key
 * AND re-validated from the segment header, so a stale segment from a
 * different run aborts loudly instead of corrupting silently.
 *
 * LIMIT (minimal scope): one transported-collective sequence shared by all
 * ranks. Different subgroups interleaving different collective sequences
 * would diverge the per-process counter and hit the barrier timeout -- which
 * dumps (op, k, rank, nranks, arrived) and aborts instead of hanging. That
 * loud failure is the design, not an accident.
 */
#define _POSIX_C_SOURCE 200809L

#include "nccl.h"

#include <stdatomic.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sched.h>
#include <signal.h>
#include <time.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <dlfcn.h>

#define FG_COLL_MAGIC 0x4647434f4c4c5955ULL /* "FGCOLL" + pad */
#define FG_COLL_TIMEOUT_DEFAULT_S 30.0

typedef int fg_cuda_err_t;
typedef enum {
    FG_CPY_H2H = 0, FG_CPY_H2D = 1, FG_CPY_D2H = 2, FG_CPY_D2D = 3
} fg_cpy_kind_t;

static fg_cuda_err_t (*fg_pMemcpy)(void *, const void *, size_t, int) = NULL;
static fg_cuda_err_t (*fg_pDevSync)(void) = NULL;
static unsigned long fg_coll_k = 0;

struct fg_coll_hdr {
    uint64_t magic;
    uint32_t nranks;
    uint32_t op;
    uint64_t count;
    int32_t datatype;
    int32_t creator_pid;
    _Atomic uint32_t arrived;
    _Atomic uint32_t done;
};

static void fg_coll_die(const char *what, const char *op, unsigned long k,
                        int rank, int nranks, uint32_t arrived, uint32_t done) {
    fprintf(stderr,
            "[fg-transport] FATAL %s: op=%s k=%lu rank=%d nranks=%d "
            "arrived=%u done=%u (timeout or state mismatch; refusing to hang)\n",
            what, op, k, rank, nranks, arrived, done);
    fflush(stderr);
    abort();
}

static double fg_now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

static double fg_coll_timeout_s(void) {
    const char *s = getenv("FAKEGPU_COLL_TIMEOUT_S");
    if (s && *s) {
        double v = strtod(s, NULL);
        if (v > 0) return v;
    }
    return FG_COLL_TIMEOUT_DEFAULT_S;
}

/* bfloat16 <-> float bit conversion (no native type needed). */
static float fg_bf16_to_f32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    memcpy(&f, &u, sizeof f);
    return f;
}

static uint16_t fg_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof u);
    /* Round-to-nearest-even on the discarded 16 bits. NaN stays NaN. */
    if ((u & 0x7fffffffU) > 0x7f800000U) return 0x7fc0U;
    return (uint16_t)((u + 0x7fffU + ((u >> 16) & 1U)) >> 16);
}

/* Returns 1 if the collective was moved with correct data (caller skips the
 * timing-only fill but still emits its trace line), 0 to take the simulate
 * path exactly as before. Never returns on barrier timeout (aborts). */
int fg_transport_allreduce(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int nranks, int rank) {
    const char *gate = getenv("FAKEGPU_NCCL_SIMULATE_ONLY");
    if (!(gate && gate[0] == '0' && gate[1] == '\0')) return 0;
    if (op != 0 /* ncclSum */) return 0;
    if (datatype != 7 /* ncclFloat32 */ && datatype != 9 /* ncclBfloat16 */)
        return 0;
    if (count == 0 || !sendbuff || !recvbuff) return 0;
    if (nranks < 1 || rank < 0 || rank >= nranks) return 0;

    if (!fg_pMemcpy) {
        fg_pMemcpy = (fg_cuda_err_t (*)(void *, const void *, size_t, int))
            dlsym(RTLD_DEFAULT, "cudaMemcpy");
        fg_pDevSync = (fg_cuda_err_t (*)(void))dlsym(RTLD_DEFAULT,
                                                     "cudaDeviceSynchronize");
        if (!fg_pMemcpy || !fg_pDevSync)
            fg_coll_die("cudaMemcpy/cudaDeviceSynchronize not found",
                        "allreduce", fg_coll_k, rank, nranks, 0, 0);
    }

    size_t elsize = (datatype == 7) ? 4 : 2;
    size_t payload = count * elsize;
    unsigned long k = fg_coll_k++;

    char name[128];
    snprintf(name, sizeof name, "/fgcoll_nr%d_c%zu_d%d_o%d_k%lu",
             nranks, count, datatype, op, k);
    size_t total = sizeof(struct fg_coll_hdr) + (size_t)nranks * payload;

    /* Fresh-or-stale open: a crashed run can leave a segment whose key AND
     * header fields match ours (same shape, same k). Inheriting its arrival
     * counter is silent corruption -- observed as arrived=3 with nranks=2.
     * Rule: O_EXCL-create when possible; on EEXIST, join only if the header
     * validates AND the creator process is still alive, else unlink and
     * recreate. PID reuse inside the millisecond live window is accepted
     * and documented; the barrier timeout remains the backstop. */
    struct fg_coll_hdr *h = NULL;
    char *slots = NULL;
    void *base = NULL;
    for (;;) {
        int fd = shm_open(name, O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd >= 0) {
            if (ftruncate(fd, (off_t)total) != 0)
        fg_coll_die("ftruncate failed", "allreduce", k, rank, nranks, 0, 0);
            base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                          MAP_SHARED, fd, 0);
            close(fd);
            if (base == MAP_FAILED)
        fg_coll_die("mmap failed", "allreduce", k, rank, nranks, 0, 0);
            h = (struct fg_coll_hdr *)base;
            slots = (char *)base + sizeof(struct fg_coll_hdr);
            h->magic = FG_COLL_MAGIC;
            h->nranks = (uint32_t)nranks;
            h->op = (uint32_t)op;
            h->count = (uint64_t)count;
            h->datatype = (int32_t)datatype;
            h->creator_pid = (int32_t)getpid();
            atomic_store(&h->arrived, 0);
            atomic_store(&h->done, 0);
            break;
        }
        if (errno != EEXIST)
        fg_coll_die("shm_open failed", "allreduce", k, rank, nranks, 0, 0);
        fd = shm_open(name, O_RDWR, 0600);
        if (fd < 0) continue; /* unlinked between: retry create */
        void *hdr = mmap(NULL, sizeof(struct fg_coll_hdr),
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (hdr == MAP_FAILED) continue;
        struct fg_coll_hdr *eh = (struct fg_coll_hdr *)hdr;
        /* EPERM means the creator is alive but unsignallable: join, do not
         * reclaim. Only ESRCH (no such process) proves death. */
        int creator_dead = (kill((pid_t)eh->creator_pid, 0) != 0 &&
                            errno == ESRCH);
        int stale = (eh->magic != FG_COLL_MAGIC ||
                     eh->nranks != (uint32_t)nranks ||
                     eh->count != (uint64_t)count ||
                     eh->datatype != (int32_t)datatype ||
                     eh->op != (uint32_t)op || creator_dead);
        munmap(hdr, sizeof(struct fg_coll_hdr));
        if (stale) {
            shm_unlink(name); /* dead creator or bad header: reclaim */
            continue;
        }
        fd = shm_open(name, O_RDWR, 0600);
        if (fd < 0) continue;
        base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
        close(fd);
        if (base == MAP_FAILED) continue;
        h = (struct fg_coll_hdr *)base;
        slots = (char *)base + sizeof(struct fg_coll_hdr);
        break;
    }

    if (fg_pDevSync() != 0)
        fg_coll_die("cudaDeviceSynchronize failed", "allreduce", k, rank,
                    nranks, 0, 0);
    char *mine = slots + (size_t)rank * payload;
    if (fg_pMemcpy(mine, sendbuff, payload, FG_CPY_D2H) != 0)
        fg_coll_die("cudaMemcpy d2h failed", "allreduce", k, rank, nranks,
                    0, 0);

    atomic_fetch_add(&h->arrived, 1);
    double t0 = fg_now_s(), timeout = fg_coll_timeout_s();
    for (;;) {
        if (atomic_load(&h->arrived) == (uint32_t)nranks) break;
        if (fg_now_s() - t0 > timeout) {
            uint32_t a = atomic_load(&h->arrived);
            munmap(base, total);
            fg_coll_die("barrier timeout", "allreduce", k, rank, nranks, a, 0);
        }
        sched_yield();
    }

    /* Reduce on host. Double accumulator; tolerance in validation covers it. */
    if (datatype == 7) {
        double *acc = (double *)calloc(count ? count : 1, sizeof(double));
        if (!acc) fg_coll_die("host calloc failed", "allreduce", k, rank,
                              nranks, (uint32_t)nranks, 0);
        for (int r = 0; r < nranks; r++) {
            float *s = (float *)(slots + (size_t)r * payload);
            for (size_t i = 0; i < count; i++) acc[i] += (double)s[i];
        }
        float *out = (float *)malloc(payload ? payload : 1);
        if (!out) fg_coll_die("host malloc failed", "allreduce", k, rank,
                              nranks, (uint32_t)nranks, 0);
        for (size_t i = 0; i < count; i++) out[i] = (float)acc[i];
        free(acc);
        if (fg_pMemcpy(recvbuff, out, payload, FG_CPY_H2D) != 0)
            fg_coll_die("cudaMemcpy h2d failed", "allreduce", k, rank, nranks,
                        (uint32_t)nranks, 0);
        free(out);
    } else {
        double *acc = (double *)calloc(count ? count : 1, sizeof(double));
        if (!acc) fg_coll_die("host calloc failed", "allreduce", k, rank,
                              nranks, (uint32_t)nranks, 0);
        for (int r = 0; r < nranks; r++) {
            uint16_t *s = (uint16_t *)(slots + (size_t)r * payload);
            for (size_t i = 0; i < count; i++)
                acc[i] += (double)fg_bf16_to_f32(s[i]);
        }
        uint16_t *out = (uint16_t *)malloc(payload ? payload : 1);
        if (!out) fg_coll_die("host malloc failed", "allreduce", k, rank,
                              nranks, (uint32_t)nranks, 0);
        for (size_t i = 0; i < count; i++)
            out[i] = fg_f32_to_bf16((float)acc[i]);
        free(acc);
        if (fg_pMemcpy(recvbuff, out, payload, FG_CPY_H2D) != 0)
            fg_coll_die("cudaMemcpy h2d failed", "allreduce", k, rank, nranks,
                        (uint32_t)nranks, 0);
        free(out);
    }

    atomic_fetch_add(&h->done, 1);
    t0 = fg_now_s();
    for (;;) {
        if (atomic_load(&h->done) == (uint32_t)nranks) break;
        if (fg_now_s() - t0 > timeout) {
            uint32_t d = atomic_load(&h->done);
            munmap(base, total);
            fg_coll_die("release timeout", "allreduce", k, rank, nranks,
                        (uint32_t)nranks, d);
        }
        sched_yield();
    }
    if (rank == 0) shm_unlink(name);
    munmap(base, total);
    return 1;
}
