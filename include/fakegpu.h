#ifndef FAKEGPU_INTERNAL_H
#define FAKEGPU_INTERNAL_H

#include <stddef.h>
#include <stdint.h>

#include "cuda_runtime.h"
#include "nccl.h"

/*
 * Shared internal definitions for libfakegpu.so.
 * This header is NOT a public CUDA/NCCL API; it is only included by the
 * interposer's own source files. Public API shapes come from the shim headers.
 */

/* ---- config (populated once from env on first use) ----------------------- */

struct fg_config {
    int    num_devices;        /* FAKEGPU_NUM_DEVICES, default 1            */
    size_t device_mem_mb;      /* FAKEGPU_DEVICE_MEM_MB, default 256        */
    int    current_device;     /* last cudaSetDevice result                 */
    /* latencies (nanoseconds unless noted) */
    long   memcpy_latency_ns;   /* FAKEGPU_MEMCPY_LATENCY_US * 1000          */
    long   kernel_latency_ms;   /* FAKEGPU_KERNEL_LATENCY_MS                 */
    long   sync_latency_ms;     /* FAKEGPU_DEVICE_SYNC_LATENCY_MS            */
    long   hostfunc_latency_us; /* FAKEGPU_HOST_FUNC_LATENCY_US              */
    double bandwidth_gbps;     /* FAKEGPU_BANDWIDTH_GBPS, 0 = fixed latency */
    uint64_t seed;              /* FAKEGPU_SEED, 0 = time based              */
    /* gpu-sim-urr: learned-cost-model attachment. The resolver maps a
     * launch to nanoseconds; NULL (the default: static zero-init, never
     * set from env) reproduces today's scalar path exactly, so CI stays
     * fast. It lives HERE, never in cudaDeviceProp (m64's sizeof guards
     * fail loudly on an insertion there). Deliberately NO problem_size:
     * cudaLaunchKernel has no such parameter, and grid*block does not
     * determine covered work for under/over-covered configs -- the
     * resolver derives it from its model artifact, keyed by kernel_name.
     * D_GPUSIM: the resolver CONSUMES predictions; it has no channel back
     * (takes inputs, returns ns; no out-params, no record access). */
    long (*cost_resolver)(const char *kernel_name,
                          int gridX, int gridY, int gridZ,
                          int blockX, int blockY, int blockZ,
                          unsigned int sharedMem);
};

void   fg_config_reload(void);          /* (re)read env into the config cache */
struct fg_config *fg_cfg(void);         /* accessor (lazy init) */

/* ---- random -------------------------------------------------------------- */

uint64_t fg_rng_next(void);             /* next 64-bit from thread-local xorshift */
void     fg_fill(void *buf, size_t nbytes);   /* write nbytes of random bytes */

/* ---- fake pointer pool --------------------------------------------------- */

void    fg_pool_init(void);             /* one-time: mmap pools for all devices */
void   *fg_pool_alloc(size_t size);     /* allocate from active device pool   */
int     fg_pool_is_fake(const void *p); /* non-zero if p is a fake device ptr */

/* ---- opaque handle registry (streams/events/nccl comms) ------------------ */

typedef struct fg_handle { struct fg_handle *next; uint64_t id; void *ptr; } fg_handle_t;

fg_handle_t *fg_handle_make(int kind);  /* alloc a registry entry */
void         fg_handle_free(fg_handle_t *h);
void        *fg_handle_ptr(fg_handle_t *h);

#define FG_HANDLE_NAME 9  /* kernel symbol map: ptr = registrant key */
void         fg_handle_record_name(const void *key, const char *name);
const char  *fg_handle_find_name(const void *key);

/* ---- latency ------------------------------------------------------------- */

void fg_sleep_ns(long ns);              /* busy or nanosleep(0 => fast) */
void fg_simulate_memcpy(size_t bytes);  /* apply memcpy latency/bandwidth */
void fg_simulate_kernel(void);
/* gpu-sim-urr: launch-aware entry. name==NULL (or no resolver configured)
 * takes today's scalar path exactly. Resolver output flows ONLY into the
 * sleep below -- it has no channel back into any record (D_GPUSIM). */
void fg_simulate_kernel_resolved(const char *kernel_name,
                                 int gridX, int gridY, int gridZ,
                                 int blockX, int blockY, int blockZ,
                                 unsigned int sharedMem);
void fg_simulate_sync(void);
long fg_latency_active(void);

/* ---- module handle (runtime module == opaque fake handle) ----------------
   Stores a fake handle pointer in *out. Returns 0 (== cudaSuccess). */
int fg_module_load(void **out, const char *name);

/* ---- NCCL comm handle (internal to the NCCL interposer) ------------------ */
struct fg_nccl_comm { int nrank; int rank; };

#endif /* FAKEGPU_INTERNAL_H */
