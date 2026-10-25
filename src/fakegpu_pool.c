/*
 * fakegpu_pool.c — per-device virtual address pools for fake device pointers,
 * plus a generic opaque-handle registry for streams/events/nccl comms.
 *
 * Design:
 *   - A pool per device is a MAP_ANONYMOUS region (default 256 MiB each) placed
 *     in the upper user-address half so the pointers look "device-like".
 *   - cudaMalloc bumps an offset inside the active device's pool and records
 *     (ptr,size) so cudaFree can locate the block; frees are lazily coalesced
 *     via a high-water mark reset (good enough for CI). Unknown ptrs still
 *     return success (CI robustness).
 *   - Streams, events and nccl comms are opaque handles generated as small
 *     integer handles from a shared registry, stored as cast pointers.
 */
#include "fakegpu.h"
#include "fakegpu_topology.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#define FGPU_POOL_ALIGN 4096UL

struct fg_pool {
    void    *base;      /* start of mapped region (device-like address)   */
    void    *brk;       /* current bump pointer                           */
    size_t   cap;       /* total capacity                                 */
};

static struct fg_pool *g_pools = NULL;  /* array[num_devices] */
static int g_pools_inited = 0;

/* ---- handle registry ----------------------------------------------------- */

typedef struct fg_hnode {
    uint64_t id;
    void    *ptr;            /* the opaque pointer returned to the caller */
    int      kind;           /* 0 free, 1 stream, 2 event, 3 comm ...     */
    const char *name;        /* kind 9 only: kernel symbol for deviceFun   */
    struct fg_hnode *next;
    struct fg_hnode *prev;
} fg_hnode_t;

static fg_hnode_t *g_handles = NULL;    /* singly linked free-style list */
static uint64_t g_next_id = 1;

void fg_pool_init(void)
{
    if (g_pools_inited) return;
    const struct fg_config *c = fg_cfg();
    int n = c->num_devices;
    if (n < 1) n = 1;

    g_pools = (struct fg_pool *)calloc((size_t)n, sizeof(*g_pools));
    if (!g_pools) return; /* caller treats failures gracefully */

    /* Hint an address in the upper half so pointers look device-like. */
    char *hint_base = (char *)0x700000000000ULL; /* x86-64 upper canonical */

    fg_topo_t *topo = fg_topo();

    for (int i = 0; i < n; i++) {
        struct fg_pool *p = &g_pools[i];
        size_t dev_cap = c->device_mem_mb * 1024 * 1024;
        if (topo && i < topo->n_dev && topo->devs[i].mem_bytes > 0) {
            dev_cap = (size_t)topo->devs[i].mem_bytes;
        }
        p->cap = dev_cap;
        p->base = mmap(hint_base + (size_t)i * (dev_cap + (1ULL << 32)),  /* sparse hints */
                       dev_cap,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (p->base == MAP_FAILED) {
            p->base = mmap(NULL, dev_cap, PROT_READ | PROT_WRITE,
                           MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }
        p->brk = p->base;
    }
    g_pools_inited = 1;
}

/* active device chosen by cudaSetDevice (default 0) */
static struct fg_pool *fg_active_pool(void)
{
    if (!g_pools_inited) fg_pool_init();
    int d = fg_cfg()->current_device;
    if (d < 0) d = 0;
    if (d >= fg_cfg()->num_devices) d = 0;
    return &g_pools[d];
}

void *fg_pool_alloc(size_t size)
{
    if (!g_pools_inited) fg_pool_init();
    if (!g_pools) return NULL;
    struct fg_pool *p = fg_active_pool();
    if (!p->base) return NULL;
    if (size == 0) size = 1;
    /* align up */
    size_t aligned = (size + FGPU_POOL_ALIGN - 1) & ~(FGPU_POOL_ALIGN - 1);
    if ((size_t)((char *)p->brk + aligned - (char *)p->base) > p->cap) {
        return NULL; /* out of fake memory */
    }
    void *ptr = p->brk;
    p->brk = (char *)p->brk + aligned;
    return ptr;
}

int fg_pool_is_fake(const void *p)
{
    if (!g_pools_inited || !g_pools) return 0;
    for (int i = 0; i < fg_cfg()->num_devices; i++) {
        struct fg_pool *p_ = &g_pools[i];
        if (p_ && p_->base && p >= p_->base && p < (void *)((char *)p_->base + p_->cap))
            return 1;
    }
    return 0;
}

/* ---- opaque handle registry --------------------------------------------- */

fg_handle_t *fg_handle_make(int kind)
{
    fg_hnode_t *node = (fg_hnode_t *)calloc(1, sizeof(*node));
    if (!node) return NULL;
    node->id = g_next_id++;
    node->kind = kind;
    /* the "pointer" returned to the caller: a small unique integer cast up. */
    node->ptr = (void *)(0x5ULL | (node->id << 3));
    node->next = g_handles;
    if (g_handles) g_handles->prev = node;
    g_handles = node;
    return (fg_handle_t *)node;
}

void fg_handle_free(fg_handle_t *h)
{
    if (!h) return;
    fg_hnode_t *node = (fg_hnode_t *)h;
    if (node->next) node->next->prev = node->prev;
    if (node->prev) node->prev->next = node->next;
    else g_handles = node->next;
    node->kind = 0;
    /* keep node around a moment is unnecessary; just free it */
    free(node);
}

void *fg_handle_ptr(fg_handle_t *h)
{
    return h ? ((fg_hnode_t *)h)->ptr : NULL;
}

/* ---- kernel symbol map (gpu-sim-urr) -------------------------------------- *
 * Same list, same alloc discipline, new kind 9: no second registry. A kind-9
 * node stores the registrant's pointer in ptr (nothing is handed back to a
 * caller for these) and the kernel symbol in name. Lookup is a walk, like
 * everything else here; the launch path measures it (urr acceptance 4). */
void fg_handle_record_name(const void *key, const char *name)
{
    fg_hnode_t *node;
    if (!key || !name) return;
    for (node = g_handles; node; node = node->next)
        if (node->kind == FG_HANDLE_NAME && node->ptr == key) {
            if (node->name && strcmp(node->name, name) == 0) return;
            free((void *)node->name);
            node->name = strdup(name);
            return;
        }
    node = (fg_hnode_t *)calloc(1, sizeof(*node));
    if (!node) return;
    node->id = g_next_id++;
    node->kind = FG_HANDLE_NAME;
    node->ptr = (void *)key;
    node->name = strdup(name);
    node->next = g_handles;
    if (g_handles) g_handles->prev = node;
    g_handles = node;
}

const char *fg_handle_find_name(const void *key)
{
    fg_hnode_t *node;
    if (!key) return NULL;
    for (node = g_handles; node; node = node->next)
        if (node->kind == FG_HANDLE_NAME && node->ptr == key)
            return node->name;
    return NULL; /* miss: the caller promises NULL -> scalar default, so the
                    function must deliver it, not fall off the end (urr
                    rework: that was UB handed to the resolver as a name). */
}
