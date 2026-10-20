#define _GNU_SOURCE
#include "fakegpu_fabric.h"

#include <errno.h>
#include <fcntl.h>
#include <signal.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define FG_FAB_DEFAULT_TIMEOUT_NS (1000000000ULL) /* 1s */

typedef struct {
    int32_t pid;
    int32_t rank;
    atomic_int attached;
    atomic_int ack;
    atomic_uint consumed_seq;
} fg_rank_slot_t;

typedef struct {
    atomic_uint seq;
    atomic_size_t len;
    atomic_int ready;
    char data[FG_FAB_MAX_PAYLOAD];
} fg_mailbox_t;

typedef struct {
    uint64_t magic;
    uint32_t version;
    uint32_t nranks;
    uint64_t epoch;
    atomic_int state; /* 0 init, 1 ready */
    uint64_t deadline_ns;
    atomic_uint barrier_arrived;
    atomic_uint barrier_gen;
    /* rank slots follow in memory, then mailboxes */
} fg_header_t;

struct fg_fab {
    char name[256];
    int shm_fd;
    size_t total_size;
    void *base;
    fg_header_t *hdr;
    fg_rank_slot_t *slots;
    fg_mailbox_t *mailboxes; /* nranks*nranks */
    int rank;
    int nranks;
    uint64_t epoch;
    uint64_t timeout_ns;
    int is_creator;
};

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void shm_name(char *out, size_t outsz, const char *run_id) {
    snprintf(out, outsz, "/gpusim-%s", run_id);
}

static size_t total_size_for(int nranks) {
    size_t s = sizeof(fg_header_t);
    s += (size_t)nranks * sizeof(fg_rank_slot_t);
    s += (size_t)nranks * (size_t)nranks * sizeof(fg_mailbox_t);
    /* align up to page */
    size_t page = 4096;
    if (s % page) s += page - (s % page);
    return s;
}

fg_fab_t *fg_fab_attach(const char *run_id, uint64_t epoch, int rank, int nranks,
                        uint64_t timeout_ns, int *err) {
    if (!run_id || nranks < 1 || rank < 0 || rank >= nranks) {
        if (err) *err = FG_FAB_EINVAL;
        return NULL;
    }
    if (timeout_ns == 0) timeout_ns = FG_FAB_DEFAULT_TIMEOUT_NS;

    fg_fab_t *f = calloc(1, sizeof(*f));
    if (!f) { if (err) *err = FG_FAB_ENOMEM; return NULL; }
    f->rank = rank;
    f->nranks = nranks;
    f->epoch = epoch;
    f->timeout_ns = timeout_ns;
    shm_name(f->name, sizeof(f->name), run_id);

    /* try create */
    int fd = shm_open(f->name, O_CREAT | O_EXCL | O_RDWR, 0666);
    int created = (fd >= 0);
    if (!created) {
        if (errno != EEXIST) {
            free(f);
            if (err) *err = FG_FAB_EIO;
            return NULL;
        }
        fd = shm_open(f->name, O_RDWR, 0666);
        if (fd < 0) {
            free(f);
            if (err) *err = FG_FAB_EIO;
            return NULL;
        }
    }
    f->shm_fd = fd;
    f->is_creator = created;

    size_t tsz = total_size_for(nranks);

    if (created) {
        if (ftruncate(fd, (off_t)tsz) != 0) {
            close(fd);
            shm_unlink(f->name);
            free(f);
            if (err) *err = FG_FAB_EIO;
            return NULL;
        }
        void *base = mmap(NULL, tsz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            close(fd);
            shm_unlink(f->name);
            free(f);
            if (err) *err = FG_FAB_ENOMEM;
            return NULL;
        }
        f->base = base;
        f->total_size = tsz;
        f->hdr = base;
        memset(f->hdr, 0, tsz);
        f->hdr->magic = FG_FAB_MAGIC;
        f->hdr->version = 1;
        f->hdr->nranks = nranks;
        f->hdr->epoch = epoch;
        atomic_store(&f->hdr->state, 1);
        f->slots = (fg_rank_slot_t *)((char *)base + sizeof(fg_header_t));
        f->mailboxes = (fg_mailbox_t *)((char *)base + sizeof(fg_header_t) +
                                         (size_t)nranks * sizeof(fg_rank_slot_t));
        /* zero slots/mailboxes already via memset */
    } else {
        /* wait for ready */
        /* mmap with known size; need determine size: use tsz based on our nranks, but verify */
        /* first map a page to read header */
        void *base = mmap(NULL, tsz, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (base == MAP_FAILED) {
            close(fd);
            free(f);
            if (err) *err = FG_FAB_ENOMEM;
            return NULL;
        }
        f->base = base;
        f->total_size = tsz;
        f->hdr = base;
        uint64_t deadline = now_ns() + timeout_ns;
        while (atomic_load(&f->hdr->state) != 1 || f->hdr->magic != FG_FAB_MAGIC) {
            if (now_ns() > deadline) {
                munmap(base, tsz);
                close(fd);
                free(f);
                if (err) *err = FG_FAB_ETIMEDOUT;
                return NULL;
            }
            usleep(1000);
        }
        if (f->hdr->version != 1 || f->hdr->nranks != (uint32_t)nranks) {
            munmap(base, tsz);
            close(fd);
            free(f);
            if (err) *err = FG_FAB_EINVAL;
            return NULL;
        }
        if (f->hdr->epoch != epoch) {
            munmap(base, tsz);
            close(fd);
            free(f);
            if (err) *err = FG_FAB_ESTALE;
            return NULL;
        }
        f->slots = (fg_rank_slot_t *)((char *)base + sizeof(fg_header_t));
        f->mailboxes = (fg_mailbox_t *)((char *)base + sizeof(fg_header_t) +
                                         (size_t)nranks * sizeof(fg_rank_slot_t));
    }

    /* register slot */
    if (rank < 0 || rank >= f->hdr->nranks) {
        /* cleanup */
        if (f->is_creator) shm_unlink(f->name);
        munmap(f->base, f->total_size);
        close(f->shm_fd);
        free(f);
        if (err) *err = FG_FAB_EINVAL;
        return NULL;
    }
    f->slots[rank].pid = (int32_t)getpid();
    f->slots[rank].rank = rank;
    atomic_store(&f->slots[rank].attached, 1);
    atomic_store(&f->slots[rank].ack, 0);
    atomic_store(&f->slots[rank].consumed_seq, 0);

    if (err) *err = FG_FAB_OK;
    return f;
}

int fg_fab_barrier(fg_fab_t *f) {
    if (!f) return FG_FAB_EINVAL;
    fg_header_t *h = f->hdr;
    unsigned int my_gen = atomic_load_explicit(&h->barrier_gen, memory_order_acquire);
    atomic_fetch_add_explicit(&h->barrier_arrived, 1, memory_order_acq_rel);

    uint64_t deadline = now_ns() + f->timeout_ns;
    while (atomic_load_explicit(&h->barrier_gen, memory_order_acquire) == my_gen &&
           atomic_load_explicit(&h->barrier_arrived, memory_order_acquire) < (uint32_t)f->nranks) {
        if (now_ns() > deadline) {
            return FG_FAB_ETIMEDOUT;
        }
        usleep(100);
    }
    if (atomic_load_explicit(&h->barrier_gen, memory_order_acquire) == my_gen) {
        unsigned int expected = my_gen;
        if (atomic_compare_exchange_strong_explicit(&h->barrier_gen, &expected, my_gen + 1,
                                                    memory_order_acq_rel, memory_order_acquire)) {
            atomic_store_explicit(&h->barrier_arrived, 0, memory_order_release);
        }
    }
    deadline = now_ns() + f->timeout_ns;
    while (atomic_load_explicit(&h->barrier_gen, memory_order_acquire) == my_gen) {
        if (now_ns() > deadline) return FG_FAB_ETIMEDOUT;
        usleep(100);
    }
    return FG_FAB_OK;
}

static int peer_alive(fg_fab_t *f, int peer) {
    if (peer < 0 || peer >= f->nranks) return 0;
    pid_t pid = (pid_t)f->slots[peer].pid;
    if (pid <= 0) return 0;
    if (kill(pid, 0) == 0) return 1;
    if (errno == EPERM) return 1;
    return 0;
}

int fg_fab_send(fg_fab_t *f, int peer, const void *buf, size_t n) {
    if (!f || !buf || peer < 0 || peer >= f->nranks || n == 0 || n > FG_FAB_MAX_PAYLOAD) {
        return FG_FAB_EINVAL;
    }
    fg_mailbox_t *m = &f->mailboxes[f->rank * f->nranks + peer];
    uint64_t deadline = now_ns() + f->timeout_ns;
    while (atomic_load_explicit(&m->ready, memory_order_acquire)) {
        if (now_ns() > deadline) {
            if (!peer_alive(f, peer)) return FG_FAB_EPEERDEAD;
            return FG_FAB_ETIMEDOUT;
        }
        usleep(100);
    }
    memcpy(m->data, buf, n);
    atomic_store_explicit(&m->len, n, memory_order_release);
    atomic_store_explicit(&m->ready, 1, memory_order_release);
    atomic_fetch_add_explicit(&m->seq, 1, memory_order_relaxed);
    return FG_FAB_OK;
}

int fg_fab_recv(fg_fab_t *f, int from, void *buf, size_t n) {
    if (!f || !buf || from < 0 || from >= f->nranks || n == 0 || n > FG_FAB_MAX_PAYLOAD) {
        return FG_FAB_EINVAL;
    }
    fg_mailbox_t *m = &f->mailboxes[from * f->nranks + f->rank];
    uint64_t deadline = now_ns() + f->timeout_ns;
    while (!atomic_load_explicit(&m->ready, memory_order_acquire)) {
        if (now_ns() > deadline) {
            if (!peer_alive(f, from)) return FG_FAB_EPEERDEAD;
            return FG_FAB_ETIMEDOUT;
        }
        usleep(100);
    }
    size_t mlen = atomic_load_explicit(&m->len, memory_order_acquire);
    if (mlen != n) {
        /* reject mismatched size; leave message? consume to avoid stuck */
        atomic_store_explicit(&m->ready, 0, memory_order_release);
        return FG_FAB_ESIZE;
    }
    memcpy(buf, m->data, n);
    atomic_store_explicit(&m->ready, 0, memory_order_release);
    atomic_fetch_add_explicit(&f->slots[from].consumed_seq, 1, memory_order_relaxed);
    return FG_FAB_OK;
}

int fg_fab_transfer(fg_fab_t *f, int dir, size_t nbytes, double bw_gbps,
                    double latency_us, double jitter_us) {
    (void)dir;
    if (!f) return FG_FAB_EINVAL;
    (void)nbytes; (void)bw_gbps; (void)latency_us; (void)jitter_us;
    /* best-effort: tiny sleep proportional; keep simple */
    if (latency_us > 0) {
        usleep((useconds_t)latency_us);
    }
    return FG_FAB_OK;
}

void fg_fab_detach(fg_fab_t *f) {
    if (!f) return;
    if (f->rank >= 0 && f->rank < f->nranks) {
        atomic_store(&f->slots[f->rank].attached, 0);
        f->slots[f->rank].pid = 0;
    }
    if (f->base && f->base != MAP_FAILED) munmap(f->base, f->total_size);
    if (f->shm_fd >= 0) close(f->shm_fd);
    free(f);
}

void fg_fab_unlink(const char *run_id) {
    char name[256];
    shm_name(name, sizeof(name), run_id);
    shm_unlink(name);
}

int fg_fab_nranks(const fg_fab_t *f) { return f ? f->nranks : 0; }
int fg_fab_rank(const fg_fab_t *f) { return f ? f->rank : -1; }

const char *fg_fab_errstr(int err) {
    switch (err) {
        case FG_FAB_OK: return "OK";
        case FG_FAB_ETIMEDOUT: return "ETIMEDOUT";
        case FG_FAB_EINVAL: return "EINVAL";
        case FG_FAB_ESIZE: return "ESIZE";
        case FG_FAB_EPEERDEAD: return "EPEERDEAD";
        case FG_FAB_ESTALE: return "ESTALE";
        case FG_FAB_ENOMEM: return "ENOMEM";
        case FG_FAB_EBUSY: return "EBUSY";
        default: return "UNKNOWN";
    }
}