/* Data-correct collectives through /dev/shm (gpu-sim-n5h, gpu-sim-f50).
 *
 * The constraint that decides this implementation: fake NCCL is handed DEVICE
 * pointers (torch allocates on the GPU). Host code cannot read or write them
 * directly -- that is precisely why fg_fill on a device pointer SIGSEGVs.
 * So every path is:
 *
 *   cudaDeviceSynchronize -> cudaMemcpy(d2h) -> write /dev/shm slot ->
 *   barrier on an atomic counter -> read peers -> reduce/gather on host ->
 *   cudaMemcpy(h2d) into recvbuff
 *
 * The CUDA runtime here is REAL (only nccl symbols are preloaded), so
 * cudaMemcpy/cudaDeviceSynchronize resolve from the process's own libcudart
 * via dlsym(RTLD_DEFAULT, ...). This is correctness infrastructure, not
 * speed: the simulated-timing path stays the default for traffic measurement.
 *
 * GATE: transport runs iff FAKEGPU_NCCL_SIMULATE_ONLY=="0". Unset or any other
 * value (the default) keeps today's simulate-only behaviour, so every
 * published traffic number stays reproducible.
 *
 * KEY: /dev/shm/fgcoll_<kind>_n<nslots>c<count>d<dtype>o<op>k<k>[_s<src>t<dst>],
 * where k is a per-process collective counter. Ranks call collectives in the
 * same order, so the k-th transported collective agrees across ranks without
 * rendezvous (the same trick as the uniqueId fix). kind separates the shapes
 * (ar/ag/rs/p2p); for p2p the (src,dst) pair is in the key because a send has
 * exactly one peer and must not barrier the world.
 *
 * PUBLISH RULE (tbp race): the segment is built fully under a pid-unique temp
 * name, then link()ed to the canonical key (atomic publish). A joiner only
 * ever sees an initialized header. Re-check every new shape against this:
 * a peer opening between create and header publish forks a ghost segment and
 * both sides time out with arrived=1 each (seen at tbp k=26).
 *
 * STALE RULE: on EEXIST, join only if the header validates AND the creator is
 * alive (kill ESRCH proves death; EPERM joins). Else unlink and recreate.
 *
 * LIMIT: one transported-call sequence shared by all ranks per kind. Ranks
 * that do not participate in a p2p exchange do not consume its k, so mixed
 * subgroup patterns can diverge the counter and hit the timeout -- which
 * dumps (op, k, rank, nranks, arrived, done) and aborts instead of hanging.
 * That loud failure is the design, not an accident. Validated scope: dp/tp
 * groups where every rank issues the same order, pp=2 pairs.
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

#define FG_KIND_AR 0
#define FG_KIND_AG 1
#define FG_KIND_RS 2
#define FG_KIND_P2P 3

typedef int fg_cuda_err_t;
typedef enum {
    FG_CPY_H2H = 0, FG_CPY_H2D = 1, FG_CPY_D2H = 2, FG_CPY_D2D = 3
} fg_cpy_kind_t;

static fg_cuda_err_t (*fg_pMemcpy)(void *, const void *, size_t, int) = NULL;
static fg_cuda_err_t (*fg_pDevSync)(void) = NULL;
static unsigned long fg_coll_k = 0;

struct fg_coll_hdr {
    uint64_t magic;
    uint32_t kind;
    uint32_t nslots;
    uint32_t nranks;
    uint32_t op;
    uint64_t count;
    int32_t datatype;
    int32_t src;
    int32_t dst;
    int32_t creator_pid;
    _Atomic uint32_t arrived;
    _Atomic uint32_t done;
};

struct fg_seg {
    char name[160];
    void *base;
    size_t total;
    struct fg_coll_hdr *h;
    char *slots;
    unsigned long k;
};

static const char *fg_kind_name(int kind) {
    switch (kind) {
    case FG_KIND_AR: return "allreduce";
    case FG_KIND_AG: return "allgather";
    case FG_KIND_RS: return "reducescatter";
    default: return "sendrecv";
    }
}

static void fg_coll_die(const char *what, int kind, unsigned long k,
                        int rank, int nranks, uint32_t arrived, uint32_t done) {
    fprintf(stderr,
            "[fg-transport] FATAL %s: op=%s k=%lu rank=%d nranks=%d "
            "arrived=%u done=%u (timeout or state mismatch; refusing to hang)\n",
            what, fg_kind_name(kind), k, rank, nranks, arrived, done);
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

static int fg_gate_open(void) {
    const char *gate = getenv("FAKEGPU_NCCL_SIMULATE_ONLY");
    return gate && gate[0] == '0' && gate[1] == '\0';
}

static int fg_dtype_ok(int datatype) {
    return datatype == 7 /* ncclFloat32 */ || datatype == 9 /* ncclBfloat16 */;
}

static void fg_cuda_resolve(int kind, unsigned long k, int rank, int nranks) {
    if (fg_pMemcpy) return;
    fg_pMemcpy = (fg_cuda_err_t (*)(void *, const void *, size_t, int))
        dlsym(RTLD_DEFAULT, "cudaMemcpy");
    fg_pDevSync = (fg_cuda_err_t (*)(void))dlsym(RTLD_DEFAULT,
                                                 "cudaDeviceSynchronize");
    if (!fg_pMemcpy || !fg_pDevSync)
        fg_coll_die("cudaMemcpy/cudaDeviceSynchronize not found",
                    kind, k, rank, nranks, 0, 0);
}

/* Publish-or-join a segment: build fully under a temp name, link() to the
 * canonical key. Returns with s->h/slots mapped. Never returns a
 * half-initialized header to a joiner (see PUBLISH RULE above). */
static void fg_seg_open(struct fg_seg *s, int kind, uint32_t nslots,
                        size_t slot_bytes, size_t count, int datatype, int op,
                        int rank, int nranks, int src, int dst) {
    size_t total = sizeof(struct fg_coll_hdr) + (size_t)nslots * slot_bytes;
    unsigned long k = fg_coll_k++;
    if (kind == FG_KIND_P2P)
        snprintf(s->name, sizeof s->name,
                 "/fgcoll_p2p_n%us%zu_c%zu_d%d_o%d_k%lu_s%dt%d",
                 nslots, slot_bytes, count, datatype, op, k, src, dst);
    else
        snprintf(s->name, sizeof s->name, "/fgcoll_%s_n%us%zu_c%zu_d%d_o%d_k%lu",
                 fg_kind_name(kind), nslots, slot_bytes, count, datatype, op, k);
    s->total = total;
    s->k = k;
    s->base = NULL;
    char keypath[192], tmppath[224];
    snprintf(keypath, sizeof keypath, "/dev/shm%s", s->name);
    snprintf(tmppath, sizeof tmppath, "/dev/shm%s.tmp.%d.%u",
             s->name, (int)getpid(), (unsigned)rand());
    for (;;) {
        int fd = shm_open(tmppath + 8, O_CREAT | O_EXCL | O_RDWR, 0600);
        if (fd < 0) {
            if (errno == EEXIST) continue; /* pid+rand collision: retry */
            fg_coll_die("shm_open tmp failed", kind, k, rank, nranks, 0, 0);
        }
        if (ftruncate(fd, (off_t)total) != 0)
            fg_coll_die("ftruncate failed", kind, k, rank, nranks, 0, 0);
        void *base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                           MAP_SHARED, fd, 0);
        close(fd);
        if (base == MAP_FAILED)
            fg_coll_die("mmap failed", kind, k, rank, nranks, 0, 0);
        struct fg_coll_hdr *h = (struct fg_coll_hdr *)base;
        h->magic = FG_COLL_MAGIC;
        h->kind = (uint32_t)kind;
        h->nslots = nslots;
        h->nranks = (uint32_t)nranks;
        h->op = (uint32_t)op;
        h->count = (uint64_t)count;
        h->datatype = (int32_t)datatype;
        h->src = (int32_t)src;
        h->dst = (int32_t)dst;
        h->creator_pid = (int32_t)getpid();
        atomic_store(&h->arrived, 0);
        atomic_store(&h->done, 0);
        if (link(tmppath, keypath) == 0) {
            shm_unlink(tmppath + 8); /* published; drop the temp name */
            s->base = base;
            s->h = h;
            s->slots = (char *)base + sizeof(struct fg_coll_hdr);
            return;
        }
        /* Someone else published first: drop ours and join theirs. */
        munmap(base, total);
        shm_unlink(tmppath + 8);
        fd = shm_open(s->name, O_RDWR, 0600);
        if (fd < 0) continue; /* unlinked between: retry */
        void *hdr = mmap(NULL, sizeof(struct fg_coll_hdr),
                           PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        close(fd);
        if (hdr == MAP_FAILED) continue;
        struct fg_coll_hdr *eh = (struct fg_coll_hdr *)hdr;
        int creator_dead = (kill((pid_t)eh->creator_pid, 0) != 0 &&
                            errno == ESRCH);
        int stale = (eh->magic != FG_COLL_MAGIC ||
                     eh->kind != (uint32_t)kind ||
                     eh->nslots != nslots ||
                     eh->count != (uint64_t)count ||
                     eh->datatype != (int32_t)datatype ||
                     eh->op != (uint32_t)op || creator_dead);
        munmap(hdr, sizeof(struct fg_coll_hdr));
        if (stale) {
            shm_unlink(s->name); /* dead creator or bad header: reclaim */
            continue;
        }
        fd = shm_open(s->name, O_RDWR, 0600);
        if (fd < 0) continue;
        base = mmap(NULL, total, PROT_READ | PROT_WRITE,
                      MAP_SHARED, fd, 0);
        close(fd);
        if (base == MAP_FAILED) continue;
        s->base = base;
        s->h = (struct fg_coll_hdr *)base;
        s->slots = (char *)base + sizeof(struct fg_coll_hdr);
        return;
    }
}

/* Wait until `want` participants arrived; rank 0 reaps the segment after
 * everyone reports done. Used by the symmetric collectives only -- p2p uses
 * its own ready/consumed handshake (no world barrier by design). */
static void fg_seg_barrier(struct fg_seg *s, int kind, uint32_t want,
                           int rank, int nranks) {
    atomic_fetch_add(&s->h->arrived, 1);
    double t0 = fg_now_s(), timeout = fg_coll_timeout_s();
    for (;;) {
        if (atomic_load(&s->h->arrived) == want) break;
        if (fg_now_s() - t0 > timeout) {
            uint32_t a = atomic_load(&s->h->arrived);
            munmap(s->base, s->total);
            fg_coll_die("barrier timeout", kind, s->k, rank, nranks, a, 0);
        }
        sched_yield();
    }
}

static void fg_seg_release(struct fg_seg *s, int rank) {
    atomic_fetch_add(&s->h->done, 1);
    double t0 = fg_now_s(), timeout = fg_coll_timeout_s();
    uint32_t want = atomic_load(&s->h->arrived);
    for (;;) {
        if (atomic_load(&s->h->done) == want) break;
        if (fg_now_s() - t0 > timeout) {
            uint32_t d = atomic_load(&s->h->done);
            munmap(s->base, s->total);
            fg_coll_die("release timeout", 0, s->k, rank, 0, want, d);
        }
        sched_yield();
    }
    if (rank == 0) shm_unlink(s->name);
    munmap(s->base, s->total);
}

/* Sum srcs[r][0..count) (float32 or bf16) into a host double accumulator. */
static void fg_host_sum(char **srcs, int nsrc, size_t count, int datatype,
                        double *acc) {
    if (datatype == 7) {
        for (int r = 0; r < nsrc; r++) {
            float *s = (float *)srcs[r];
            for (size_t i = 0; i < count; i++) acc[i] += (double)s[i];
        }
    } else {
        for (int r = 0; r < nsrc; r++) {
            uint16_t *s = (uint16_t *)srcs[r];
            for (size_t i = 0; i < count; i++)
                acc[i] += (double)fg_bf16_to_f32(s[i]);
        }
    }
}

static void fg_host_write(const double *acc, size_t count, int datatype,
                          void *out) {
    if (datatype == 7) {
        float *o = (float *)out;
        for (size_t i = 0; i < count; i++) o[i] = (float)acc[i];
    } else {
        uint16_t *o = (uint16_t *)out;
        for (size_t i = 0; i < count; i++)
            o[i] = fg_f32_to_bf16((float)acc[i]);
    }
}

/* Returns 1 if the collective was moved with correct data (caller skips the
 * timing-only fill but still emits its trace line), 0 to take the simulate
 * path exactly as before. Never returns on barrier timeout (aborts). */
int fg_transport_allreduce(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int op, int nranks, int rank) {
    if (!fg_gate_open()) return 0;
    if (op != 0 /* ncclSum */) return 0;
    if (!fg_dtype_ok(datatype)) return 0;
    if (count == 0 || !sendbuff || !recvbuff) return 0;
    if (nranks < 1 || rank < 0 || rank >= nranks) return 0;
    fg_cuda_resolve(FG_KIND_AR, fg_coll_k, rank, nranks);

    size_t elsize = (datatype == 7) ? 4 : 2;
    size_t payload = count * elsize;
    struct fg_seg s;
    fg_seg_open(&s, FG_KIND_AR, (uint32_t)nranks, payload, count, datatype,
                op, rank, nranks, -1, -1);

    if (fg_pDevSync() != 0)
        fg_coll_die("cudaDeviceSynchronize failed", FG_KIND_AR, s.k, rank,
                    nranks, 0, 0);
    if (fg_pMemcpy(s.slots + (size_t)rank * payload, sendbuff, payload,
                   FG_CPY_D2H) != 0)
        fg_coll_die("cudaMemcpy d2h failed", FG_KIND_AR, s.k, rank, nranks,
                    0, 0);
    fg_seg_barrier(&s, FG_KIND_AR, (uint32_t)nranks, rank, nranks);

    char *srcs[64];
    if (nranks > 64)
        fg_coll_die("nranks too large", FG_KIND_AR, s.k, rank, nranks, 0, 0);
    for (int r = 0; r < nranks; r++) srcs[r] = s.slots + (size_t)r * payload;
    double *acc = (double *)calloc(count ? count : 1, sizeof(double));
    if (!acc)
        fg_coll_die("host calloc failed", FG_KIND_AR, s.k, rank, nranks,
                    (uint32_t)nranks, 0);
    fg_host_sum(srcs, nranks, count, datatype, acc);
    void *out = malloc(payload ? payload : 1);
    if (!out)
        fg_coll_die("host malloc failed", FG_KIND_AR, s.k, rank, nranks,
                    (uint32_t)nranks, 0);
    fg_host_write(acc, count, datatype, out);
    free(acc);
    if (fg_pMemcpy(recvbuff, out, payload, FG_CPY_H2D) != 0)
        fg_coll_die("cudaMemcpy h2d failed", FG_KIND_AR, s.k, rank, nranks,
                    (uint32_t)nranks, 0);
    free(out);
    fg_seg_release(&s, rank);
    return 1;
}

int fg_transport_allgather(const void *sendbuff, void *recvbuff, size_t count,
                           int datatype, int nranks, int rank) {
    if (!fg_gate_open()) return 0;
    if (!fg_dtype_ok(datatype)) return 0;
    if (count == 0 || !sendbuff || !recvbuff) return 0;
    if (nranks < 1 || rank < 0 || rank >= nranks) return 0;
    fg_cuda_resolve(FG_KIND_AG, fg_coll_k, rank, nranks);

    size_t elsize = (datatype == 7) ? 4 : 2;
    size_t payload = count * elsize;
    struct fg_seg s;
    fg_seg_open(&s, FG_KIND_AG, (uint32_t)nranks, payload, count, datatype,
                0, rank, nranks, -1, -1);

    if (fg_pDevSync() != 0)
        fg_coll_die("cudaDeviceSynchronize failed", FG_KIND_AG, s.k, rank,
                    nranks, 0, 0);
    if (fg_pMemcpy(s.slots + (size_t)rank * payload, sendbuff, payload,
                   FG_CPY_D2H) != 0)
        fg_coll_die("cudaMemcpy d2h failed", FG_KIND_AG, s.k, rank, nranks,
                    0, 0);
    fg_seg_barrier(&s, FG_KIND_AG, (uint32_t)nranks, rank, nranks);

    size_t total = payload * (size_t)nranks;
    if (fg_pMemcpy(recvbuff, s.slots, total, FG_CPY_H2D) != 0)
        fg_coll_die("cudaMemcpy h2d failed", FG_KIND_AG, s.k, rank, nranks,
                    (uint32_t)nranks, 0);
    fg_seg_release(&s, rank);
    return 1;
}

int fg_transport_reducescatter(const void *sendbuff, void *recvbuff,
                               size_t recvcount, int datatype, int op,
                               int nranks, int rank) {
    if (!fg_gate_open()) return 0;
    if (op != 0 /* ncclSum */) return 0;
    if (!fg_dtype_ok(datatype)) return 0;
    if (recvcount == 0 || !sendbuff || !recvbuff) return 0;
    if (nranks < 1 || rank < 0 || rank >= nranks) return 0;
    fg_cuda_resolve(FG_KIND_RS, fg_coll_k, rank, nranks);

    size_t elsize = (datatype == 7) ? 4 : 2;
    size_t chunk = recvcount * elsize;
    size_t payload = chunk * (size_t)nranks;
    struct fg_seg s;
    fg_seg_open(&s, FG_KIND_RS, (uint32_t)nranks, payload, recvcount,
                datatype, op, rank, nranks, -1, -1);

    if (fg_pDevSync() != 0)
        fg_coll_die("cudaDeviceSynchronize failed", FG_KIND_RS, s.k, rank,
                    nranks, 0, 0);
    if (fg_pMemcpy(s.slots + (size_t)rank * payload, sendbuff, payload,
                   FG_CPY_D2H) != 0)
        fg_coll_die("cudaMemcpy d2h failed", FG_KIND_RS, s.k, rank, nranks,
                    0, 0);
    fg_seg_barrier(&s, FG_KIND_RS, (uint32_t)nranks, rank, nranks);

    /* Rank r reduces everyone's r-th chunk. */
    char *srcs[64];
    if (nranks > 64)
        fg_coll_die("nranks too large", FG_KIND_RS, s.k, rank, nranks, 0, 0);
    for (int r = 0; r < nranks; r++)
        srcs[r] = s.slots + (size_t)r * payload + (size_t)rank * chunk;
    double *acc = (double *)calloc(recvcount ? recvcount : 1, sizeof(double));
    if (!acc)
        fg_coll_die("host calloc failed", FG_KIND_RS, s.k, rank, nranks,
                    (uint32_t)nranks, 0);
    fg_host_sum(srcs, nranks, recvcount, datatype, acc);
    void *out = malloc(chunk ? chunk : 1);
    if (!out)
        fg_coll_die("host malloc failed", FG_KIND_RS, s.k, rank, nranks,
                    (uint32_t)nranks, 0);
    fg_host_write(acc, recvcount, datatype, out);
    free(acc);
    if (fg_pMemcpy(recvbuff, out, chunk, FG_CPY_H2D) != 0)
        fg_coll_die("cudaMemcpy h2d failed", FG_KIND_RS, s.k, rank, nranks,
                    (uint32_t)nranks, 0);
    free(out);
    fg_seg_release(&s, rank);
    return 1;
}

/* Point-to-point: no world barrier by design. Sender publishes data and
 * signals ready (arrived=1); the receiver spins for ready, copies, signals
 * consumed (done=1); the sender spins for consumed, then unlinks. Either side
 * timing out dumps and aborts -- a PP stage waiting on a dead neighbour must
 * never hang silently. Either side may publish first (the join path only ever
 * sees an initialized header); the SENDER always writes the slot before
 * signalling ready, so order is safe regardless of who created. */
static int fg_transport_p2p(const void *sendbuff, void *recvbuff, size_t count,
                            int datatype, int self, int peer, int is_send) {
    if (!fg_gate_open()) return 0;
    if (!fg_dtype_ok(datatype)) return 0;
    if (count == 0 || (!sendbuff && is_send) || (!recvbuff && !is_send))
        return 0;
    fg_cuda_resolve(FG_KIND_P2P, fg_coll_k, self, 2);

    size_t elsize = (datatype == 7) ? 4 : 2;
    size_t payload = count * elsize;
    int src = is_send ? self : peer, dst = is_send ? peer : self;
    struct fg_seg s;
    fg_seg_open(&s, FG_KIND_P2P, 1, payload, count, datatype, 0, self, 2,
                src, dst);

    if (is_send) {
        if (fg_pDevSync() != 0)
            fg_coll_die("cudaDeviceSynchronize failed", FG_KIND_P2P, s.k,
                        self, 2, 0, 0);
        if (fg_pMemcpy(s.slots, sendbuff, payload, FG_CPY_D2H) != 0)
            fg_coll_die("cudaMemcpy d2h failed", FG_KIND_P2P, s.k, self,
                        2, 0, 0);
        atomic_fetch_add(&s.h->arrived, 1); /* ready */
        double t0 = fg_now_s(), timeout = fg_coll_timeout_s();
        for (;;) {
            if (atomic_load(&s.h->done) == 1) break;
            if (fg_now_s() - t0 > timeout) {
                munmap(s.base, s.total);
                fg_coll_die("send-consumed timeout (peer dead?)",
                            FG_KIND_P2P, s.k, self, 2, 1, 0);
            }
            sched_yield();
        }
        shm_unlink(s.name);
        munmap(s.base, s.total);
    } else {
        double t0 = fg_now_s(), timeout = fg_coll_timeout_s();
        for (;;) {
            if (atomic_load(&s.h->arrived) == 1) break;
            if (fg_now_s() - t0 > timeout) {
                munmap(s.base, s.total);
                fg_coll_die("recv-ready timeout (peer dead?)",
                            FG_KIND_P2P, s.k, self, 2, 0, 0);
            }
            sched_yield();
        }
        if (fg_pMemcpy(recvbuff, s.slots, payload, FG_CPY_H2D) != 0)
            fg_coll_die("cudaMemcpy h2d failed", FG_KIND_P2P, s.k, self,
                        2, 1, 0);
        atomic_fetch_add(&s.h->done, 1); /* consumed */
        munmap(s.base, s.total);
    }
    return 1;
}

int fg_transport_send(const void *sendbuff, size_t count, int datatype,
                      int self, int peer) {
    return fg_transport_p2p(sendbuff, NULL, count, datatype, self, peer, 1);
}

int fg_transport_recv(void *recvbuff, size_t count, int datatype,
                      int self, int peer) {
    return fg_transport_p2p(NULL, recvbuff, count, datatype, self, peer, 0);
}
