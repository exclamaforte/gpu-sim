/*
 * fakegpu_cuda.c — CUDA runtime + driver API interposer.
 *
 * Every symbol declared in shim/cuda_runtime.h is defined here. None of these
 * forward to a real CUDA library (RTLD_NEXT); they unconditionally return
 * success and, where a buffer is produced at an API boundary, fill it with
 * random bytes. Latency is injected through fg_simulate_*.
 */
#include "fakegpu.h"
#include "cuda_runtime.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- memory allocations -------------------------------------------------- */

cudaError_t cudaMalloc(void **devPtr, size_t size)
{
    if (!devPtr) return cudaErrorInvalidValue;
    void *p = fg_pool_alloc(size ? size : 1);
    if (!p) return (cudaError_t)cudaErrorInvalidDevice;
    *devPtr = p;
    return cudaSuccess;
}

cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream)
{
    (void)stream;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags)
{
    (void)flags;
    return cudaMalloc(devPtr, size);
}

cudaError_t cudaMallocHost(void **ptr, size_t size)
{
    if (!ptr) return cudaErrorInvalidValue;
    void *p = fg_pool_alloc(size ? size : 1);
    if (!p) return (cudaError_t)cudaErrorInvalidDevice;
    *ptr = p;
    return cudaSuccess;
}

cudaError_t cudaHostAlloc(void **pHost, size_t bytes, unsigned int flags)
{
    (void)flags;
    return cudaMallocHost(pHost, bytes);
}

cudaError_t cudaFree(void *devPtr)
{
    (void)devPtr; /* always succeed; CI robustness for unknown ptrs */
    return cudaSuccess;
}

cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream)
{
    (void)devPtr; (void)stream;
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void *ptr)
{
    (void)ptr;
    return cudaSuccess;
}

/* ---- memset (device memory) --------------------------------------------- */

cudaError_t cudaMemset(void *devPtr, int value, size_t count)
{
    fg_simulate_memcpy(count);
    if (devPtr && count) memset(devPtr, value, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemGetInfo(size_t *free, size_t *total)
{
    /* gpu-sim-u2o.8dw: the repo's own generated driver
     * (epic_u2o/kernel_driver.py, u2o.6.3) emits a cudaMemGetInfo call to
     * report the device-side footprint. libfakegpu did not export it, so any
     * -cudart=none link of such a driver failed with 'undefined reference to
     * cudaMemGetInfo'. Found while implementing the 8dw preflight smoke test,
     * which tripped over the same gap. */
    if (!free || !total) return (cudaError_t)cudaErrorInvalidValue;
    const /* ==========================================================================
 * SINGLE SOURCE OF TRUTH for the device quantities the fake reports.
 *
 * gpu-sim-u2o.19, structural rule taken from its review: a quantity the fake
 * reports through MORE THAN ONE CHANNEL must be single-sourced, and the test
 * must pin the channels to EACH OTHER rather than pinning one to a literal.
 *
 * Why that is not tidiness here. maxThreadsPerMultiProcessor was reaching
 * callers three ways -- the shim struct, the real-layout overlay, and
 * cudaDeviceGetAttribute -- and the three disagreed, so the fake was a
 * DIFFERENT PART depending on which channel the caller happened to use. That
 * is worse than any one value being wrong, because a caller cannot detect it.
 * l2CacheSize was worse still: the overlay said 96 MB while the shim-only
 * `totalL2Cache` said 4 MB, a 24x disagreement, and PyTorch reads exactly
 * this family of numbers when it computes occupancy.
 *
 * Same lesson 6.6.2 learned as "one byte convention in the file". A value
 * written in three places is three values the moment one of them is edited.
 *
 * Every number below was READ OFF A REAL RTX 5090, ts 20260928t105453-2dc7af7c.
 * The previous values (2048 / 233472 / 128 MB) described a part that does not
 * exist, and the reviewer's bead blamed the wrong thing: the byte OFFSETS were
 * correct (verified with offsetof against CUDA 13.3) and the VALUES were wrong.
 * ========================================================================== */
#define FG_MAX_THREADS_PER_SM      1536       /* was 2048: own probe + re-probe, no attr exists */
#define FG_SHMEM_PER_SM          102400       /* was 233472 (228 KB): own probe + megakern-37e.2 */
#define FG_L2_BYTES            100663296      /* was 134217728 (128 MB): own probe + megakern-37e.2 */
/* gpu-sim-u2o.19 C5: max shared memory per block (opt-in), measured on the
 * real 5090 by gpu-sim's own probe (ts 20261010t211802-ff147780: 101376) AND
 * megakern-37e.2 (101376 to the byte). Set from the two-source agreement,
 * not borrowed. Served through cudaDeviceGetAttribute only: the real
 * cudaDeviceProp has no opt-in field. */
#define FG_OPTIN_PER_BLOCK       101376
#define FG_SMEM_PER_BLOCK         49152
#define FG_REGS_PER_BLOCK         65536
#define FG_REGS_PER_SM            65536
#define FG_SM_COUNT                170
#define FG_WARP                     32

/* ========================================================================== */

struct fg_config *c = fg_cfg();
    *total = (size_t)c->device_mem_mb * 1024u * 1024u;
    /* the fake pool hands out from this budget; report what is notionally
     * left as the whole budget minus one page, so a caller sees a plausible
     * non-zero figure rather than a device that looks full */
    *free = *total > (4u << 20) ? *total - (4u << 20) : 0;
    return cudaSuccess;
}

cudaError_t cudaMemsetD8(void *devPtr, unsigned char value, size_t count)
{
    fg_simulate_memcpy(count);
    if (devPtr && count) memset(devPtr, value, count);
    return cudaSuccess;
}

/* ---- memcpy --------------------------------------------------------------- */

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind)
{
    (void)kind;
    fg_simulate_memcpy(count);
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind,
                            cudaStream_t stream)
{
    (void)kind; (void)stream;
    fg_simulate_memcpy(count);
    memcpy(dst, src, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyHtoD(void *dstDevice, const void *srcHost, size_t count)
{
    fg_simulate_memcpy(count);
    memcpy(dstDevice, srcHost, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyDtoH(void *dstHost, const void *srcDevice, size_t count)
{
    fg_simulate_memcpy(count);
    memcpy(dstHost, srcDevice, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyDtoD(void *dstDevice, const void *srcDevice, size_t count)
{
    fg_simulate_memcpy(count);
    memcpy(dstDevice, srcDevice, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyHtoDAsync(void *dstDevice, const void *srcHost, size_t count,
                                cudaStream_t stream)
{
    (void)stream;
    fg_simulate_memcpy(count);
    memcpy(dstDevice, srcHost, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyDtoHAsync(void *dstHost, const void *srcDevice, size_t count,
                                cudaStream_t stream)
{
    (void)stream;
    fg_simulate_memcpy(count);
    memcpy(dstHost, srcDevice, count);
    return cudaSuccess;
}

/* ---- streams / events ---------------------------------------------------- */

cudaError_t cudaStreamCreate(cudaStream_t *pStream)
{
    if (!pStream) return cudaErrorInvalidValue;
    fg_handle_t *h = fg_handle_make(1);
    *pStream = h ? fg_handle_ptr(h) : NULL;
    return cudaSuccess;
}

cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
{
    (void)flags;
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags,
                                         int priority)
{
    (void)flags; (void)priority;
    return cudaStreamCreate(pStream);
}

cudaError_t cudaStreamDestroy(cudaStream_t stream)
{
    (void)stream;
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t stream)
{
    (void)stream;
    fg_simulate_sync();
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t stream)
{
    (void)stream;
    return cudaSuccess; /* fakes are always "ready" */
}

cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags)
{
    (void)stream; (void)event; (void)flags;
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t *event)
{
    if (!event) return cudaErrorInvalidValue;
    fg_handle_t *h = fg_handle_make(2);
    *event = h ? fg_handle_ptr(h) : NULL;
    return cudaSuccess;
}

cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags)
{
    (void)flags;
    return cudaEventCreate(event);
}

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    (void)event; (void)stream;
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t event)
{
    (void)event;
    return cudaSuccess;
}

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    (void)event;
    fg_simulate_sync();
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    (void)event;
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    if (!ms) return cudaErrorInvalidValue;
    (void)start; (void)end;
    /* No real timestamps are tracked; report the configured per-kernel
     * latency so shim-compiled timing drivers produce a plausible number
     * instead of failing to link. Real timing comes only from real-GPU
     * cudaEventElapsedTime, never from this path. */
    *ms = (float)fg_cfg()->kernel_latency_ms;
    return cudaSuccess;
}

const char *cudaGetErrorString(cudaError_t error)
{
    switch (error) {
    case cudaSuccess:              return "no error";
    case cudaErrorInvalidValue:     return "invalid argument";
    case cudaErrorInvalidDevice:    return "invalid device ordinal";
    case cudaErrorInitializationError: return "initialization error";
    case cudaErrorInvalidConfiguration: return "invalid configuration argument";
    default:                        return "unknown error";
    }
}

/* The fake's error state (gpu-sim-u2o.6.1.3). A fake normally has none --
 * every call already returned success. Launch validation is the exception:
 * it is the one place the fake can detect a genuinely bad program, and nvcc's
 * generated stub DISCARDS the return value of __cudaLaunchKernel, so without
 * sticky state an invalid launch would vanish silently and the preflight
 * smoke test would happily approve it. Defined here, before the launch paths
 * that record into it; the public accessors live further down. */

static cudaError_t g_last_error = cudaSuccess;

static cudaError_t fg_record(cudaError_t e)
{
    if (e != cudaSuccess) g_last_error = e;
    return e;
}

/* Validate a kernel launch config; returns cudaSuccess or
 * cudaErrorInvalidConfiguration, and records the failure in the fake's error
 * state so it survives nvcc discarding the launch return value. Lets the
 * preflight smoke test catch
 * malformed launches instead of silently succeeding. */
static cudaError_t fg_validate_launch(int gridX, int gridY, int gridZ,
                                      int blockX, int blockY, int blockZ,
                                      unsigned int sharedMem)
{
    if (gridX < 1 || gridY < 1 || gridZ < 1) return cudaErrorInvalidConfiguration;
    if (blockX < 1 || blockX > 1024) return cudaErrorInvalidConfiguration;
    if (blockY < 1 || blockY > 1024) return cudaErrorInvalidConfiguration;
    if (blockZ < 1 || blockZ > 64) return cudaErrorInvalidConfiguration;
    {
        long long threads = (long long)blockX * blockY * blockZ;
        if (threads < 1 || threads > 1024) return cudaErrorInvalidConfiguration;
    }
    if (sharedMem > (unsigned int)228 * 1024) return cudaErrorInvalidConfiguration;
    return cudaSuccess;
}

cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t cb,
                                  void *userData)
{
    (void)stream; (void)userData;
    if (cb) cb(stream, 0, userData); /* invoke immediately */
    return cudaSuccess;
}

/* ---- device / context ---------------------------------------------------- */

cudaError_t cudaGetDeviceCount(int *count)
{
    if (!count) return cudaErrorInvalidValue;
    *count = fg_cfg()->num_devices;
    return cudaSuccess;
}

cudaError_t cudaGetDevice(int *device)
{
    if (!device) return cudaErrorInvalidValue;
    *device = fg_cfg()->current_device;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device)
{
    struct fg_config *c = fg_cfg();
    if (device < 0 || device >= c->num_devices) {
        return (cudaError_t)cudaErrorInvalidDevice;
    }
    c->current_device = device;
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize(void)
{
    fg_simulate_sync();
    /* surface (and consume) a pending launch-validation failure, the way real
     * cudart does: the launch return value may have been discarded by the
     * compiler stub, so the error has to arrive through here. */
    {
        cudaError_t pending = g_last_error;
        g_last_error = cudaSuccess;
        return pending;
    }
}

cudaError_t cudaDeviceReset(void)
{
    return cudaSuccess;
}

/* Fake-GPU resource limits: recorded but never enforced (the simulator does
   not track device memory quotas). SetLimit stores, GetLimit reads back, so a
   set/read-back pair is consistent rather than silently stale. */
static size_t g_fake_limit[64];

cudaError_t cudaDeviceSetLimit(int limit, size_t value)
{
    if (limit >= 0 && limit < 64) g_fake_limit[limit] = value;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t *value, int limit)
{
    if (!value) return cudaErrorInvalidValue;
    if (limit >= 0 && limit < 64) *value = g_fake_limit[limit];
    else *value = 0;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device)
{
    if (!prop) return cudaErrorInvalidValue;
    struct fg_config *c = fg_cfg();
    if (device < 0 || device >= c->num_devices)
        return (cudaError_t)cudaErrorInvalidDevice;
    memset(prop, 0, sizeof(*prop));
    snprintf(prop->name, sizeof(prop->name), "FakeGPU-%d", device);
    prop->totalGlobalMem          = (size_t)c->device_mem_mb * 1024 * 1024;
    prop->sharedMemPerBlock       = FG_SMEM_PER_BLOCK;
    prop->maxThreadsPerBlock      = 1024;  /* unchanged: real 5090 is 1024 */
    /* gpu-sim-u2o.19: 1536, measured on the real 5090 -- see the overlay
     * comment below for provenance. The shim-struct assignment and the
     * real-layout overlay MUST agree, or the same field reports two different
     * values depending on which header the caller was built against. */
    prop->maxThreadsPerMultiProcessor = FG_MAX_THREADS_PER_SM;
    prop->multiProcessorCount     = FG_SM_COUNT;   /* match real RTX 5090 */
    /* Single-sourced, and written BEFORE the overlay on purpose: the shim
     * places this at byte 312, which is inside the overlay's range, so writing
     * it afterwards would clobber the overlay's `pitch`. See the long note at
     * the shim's own block below. */
    prop->totalL2Cache            = FG_L2_BYTES;
    prop->clockRate               = 1590000;   /* kHz, fake */
    /* Real-layout overlay: third-party binaries (e.g. PyTorch) are built
     * against real CUDA headers, whose cudaDeviceProp layout differs from
     * shim/cuda_runtime.h. They read major/minor (and a few basics) at real
     * offsets (driver_types.h, CUDA 13): totalGlobalMem@288,
     * sharedMemPerBlock@296, maxThreadsPerBlock@320, totalConstMem@352,
     * they read zeros -> compute capability 0.0 -> SIGFPE (gpu-sim-u2o.16).
     * 19 writes at 288..636, all < the SHIM sizeof 696 as plain arithmetic
     * (last write ends at 636 + 4 = 640 < 696; the 8-byte writes sit at
     * lower offsets). "Shim" matters:
     * there are TWO cudaDeviceProp structs -- shim/cuda_runtime.h (696) and
     * real CUDA 13.3 (1008) -- and this overlay targets the REAL layout.
     * The live guards are in tests/test_device_prop_overlay.cu (built
     * against real headers): offsetof 596/604/624 asserted from the header,
     * sizeof(l2CacheSize) == 4, sizeof(cudaDeviceProp) == 1008. */
    {
        unsigned char *base = (unsigned char *)prop;
        size_t total_mem = (size_t)c->device_mem_mb * 1024 * 1024;
        size_t shmem = FG_SMEM_PER_BLOCK;
        size_t const_mem = 64 * 1024;
        size_t pitch = 2147483647u;
        size_t shmem_sm = (size_t)FG_SHMEM_PER_SM;   /* per-SM, measured on RTX 5090 (ts
                                             * 20260928t105453-2dc7af7c); was
                                             * 228 KB, which is not this part */
        int max_threads = 1024, major = 12, minor = 0, sm_count = FG_SM_COUNT;
        int regs_block = FG_REGS_PER_BLOCK, warp = FG_WARP, one = 1;
        int max_dim[3] = {1024, 1024, 64};
        int max_grid[3] = {2147483647, 65535, 65535};
        /* gpu-sim-u2o.19. These three VALUES were wrong; the OFFSETS were not.
         *
         * The bead was reopened saying "the hardcoded byte offsets for these
         * three are wrong for this CUDA version". They are not. Verified with
         * offsetof() against the real CUDA 13.3 headers:
         *   l2CacheSize                 @ 596, size 4
         *   maxThreadsPerMultiProcessor @ 604, size 4
         *   sharedMemPerMultiprocessor  @ 624, size 8
         * and sizeof(cudaDeviceProp) = 1008. The three memcpy targets below
         * match those offsets exactly.
         *
         * What WAS wrong is the intent. The code set 2048 / 233472 / 128 MB,
         * which are not this part's numbers. Read off a real RTX 5090
         * (ts 20260928t105453-2dc7af7c, gpu-sim-u2o.19):
         *   maxThreadsPerMultiProcessor = 1536
         *   sharedMemPerMultiprocessor  = 102400   (100 KB)
         *   l2CacheSize                 = 100663296 (96 MB)
         *
         * So a fake reporting 2048 threads/SM and 128 MB of L2 for a GB202
         * would overstate occupancy by 1.33x and cache by 1.33x. That matters
         * more than 'low priority' suggests: PyTorch reads
         * maxThreadsPerMultiProcessor when it computes occupancy, so an
         * occupancy-sensitive kernel launched under the fake would be sized
         * against a part that does not exist.
         */
        int max_thr_sm = FG_MAX_THREADS_PER_SM, regs_sm = FG_REGS_PER_SM,
            l2 = FG_L2_BYTES;   /* l2CacheSize is an int on CUDA 13.3 */
        memcpy(base + 288, &total_mem, sizeof(total_mem));
        memcpy(base + 296, &shmem, sizeof(shmem));
        memcpy(base + 304, &regs_block, sizeof(regs_block));
        memcpy(base + 308, &warp, sizeof(warp));
        memcpy(base + 312, &pitch, sizeof(pitch));
        memcpy(base + 320, &max_threads, sizeof(max_threads));
        memcpy(base + 324, max_dim, sizeof(max_dim));
        memcpy(base + 336, max_grid, sizeof(max_grid));
        memcpy(base + 352, &const_mem, sizeof(const_mem));
        memcpy(base + 360, &major, sizeof(major));
        memcpy(base + 364, &minor, sizeof(minor));
        memcpy(base + 384, &sm_count, sizeof(sm_count));
        memcpy(base + 560, &one, sizeof(one));        /* concurrentKernels */
        memcpy(base + 588, &one, sizeof(one));        /* unifiedAddressing */
        memcpy(base + 596, &l2, sizeof(l2));
        memcpy(base + 604, &max_thr_sm, sizeof(max_thr_sm));
        memcpy(base + 624, &shmem_sm, sizeof(shmem_sm));
        memcpy(base + 632, &regs_sm, sizeof(regs_sm));
        memcpy(base + 636, &one, sizeof(one));        /* managedMemory */
    }
    prop->isMultiGpuBoard         = 0;
    prop->multiGpuBoardGroupID    = 0;
    prop->totalConstMem           = 64 * 1024;
    /* Single-sourced with the overlay's l2CacheSize. These disagreed by 24x
     * (4 MB here, 96 MB there), and a shim-built caller and a real-header
     * caller would each have been right about a different part. Note this field
     * does not exist in CUDA 13.3's cudaDeviceProp at all, so it can never be the
     * authority -- but it must not contradict the real one either. */
    /* gpu-sim-u2o.19: ROOT CAUSE of the two long-standing overlay gaps.
     *
     * The shim's cudaDeviceProp puts totalL2Cache at byte 312 with size 8, so
     * this assignment writes bytes 312-319. The overlay writes `pitch` at 312
     * (8 bytes) and `max_threads` at 320. Because the shim block runs AFTER
     * the overlay, the shim WINS wherever the two overlap -- which is how a
     * field the overlay wrote came back as something else entirely, and why
     * maxThreadsPerBlock and warpSize read 0 while cudaDeviceGetAttribute
     * (a different code path, no struct involved) reported them correctly.
     *
     * The fix is not to change a value, it is to stop two authorities writing
     * the same bytes. The overlay is the authority for real-header callers, so
     * the shim-only fields must not be allowed to overwrite it. Rather than
     * reorder this function late at night, the shim's totalL2Cache is written
     * BEFORE the overlay and the overlay's pitch/max_threads win from then on.
     */
    prop->pciBusID                = device;
    prop->pciDeviceID             = device * 0x10de;
    prop->pciDomainID             = 0;
    prop->major = 12; prop->minor = 0;   /* sm_120: match real RTX 5090 (was 8.0) */
    prop->memPitch              = 2147483647;
    prop->unifiedAddressing     = 1;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device)
{
    return cudaGetDeviceProperties(prop, device);
}

cudaError_t cudaDeviceGetAttribute(int *pi, int attr, int device)
{
    if (!pi) return cudaErrorInvalidValue;
    struct fg_config *c = fg_cfg();
    if (device < 0 || device >= c->num_devices)
        return (cudaError_t)cudaErrorInvalidDevice;
    switch (attr) {
    case cudaDevAttrMaxThreadsPerBlock:          *pi = 1024; break;
    /* gpu-sim-u2o.19: 1536, measured on the real 5090. This is a THIRD
     * place the same number appears; if it drifts from the overlay the fake
     * becomes self-inconsistent, so the test pins all three. */
    case cudaDevAttrMaxThreadsPerMultiProcessor:
        *pi = FG_MAX_THREADS_PER_SM; break;
    case cudaDevAttrMultiProcessorCount:         *pi = FG_SM_COUNT; break;   /* match real RTX 5090 */
    case cudaDevAttrMaxSharedMemoryPerBlock:     *pi = FG_SMEM_PER_BLOCK; break;
    /* gpu-sim-u2o.19 C5: the fake stays self-consistent -- both attribute
     * channels serve the single-sourced value. Recorded hardware quirk: the
     * REAL 5090's cudaDevAttrMaxSharedMemoryPerMultiprocessor reads 0 while
     * the struct reads 102400 (ts 20261010t211802-ff147780). Emulating the 0
     * would reintroduce one-struct-two-answers, so the fake reports what the
     * part has, and the discrepancy lives in metrics.json, not in a case. */
    case cudaDevAttrMaxSharedMemoryPerMultiprocessor:
        *pi = FG_SHMEM_PER_SM; break;
    case cudaDevAttrMaxSharedMemoryPerBlockOptin:
        *pi = FG_OPTIN_PER_BLOCK; break;
    case cudaDevAttrClockRate:                   *pi = 1590000; break;
    /* gpu-sim-u2o.19: cudaDevAttrTotalGlobalMem does not exist in CUDA 13.x --
     * it was removed when the attribute list was renumbered. The case is GONE
     * rather than renumbered, because keeping a case for a non-existent
     * enumerator is precisely the drift this file keeps having. Use
     * cudaDeviceGetAttribute(cudaDevAttrTotalGlobalMemory) or read
     * cudaDeviceProp.totalGlobalMem instead. */
    case cudaDevAttrComputeCapabilityMajor:      *pi = 12;  break;   /* sm_120 */
    case cudaDevAttrComputeCapabilityMinor:      *pi = 0;   break;
    default:                                     *pi = 0;    break;
    }
    return cudaSuccess;
}

/* ---- kernel launch ------------------------------------------------------- */

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func)
{
    (void)func;
    if (!attr) return cudaErrorInvalidValue;
    memset(attr, 0, sizeof(*attr));
    attr->maxThreadsPerBlock = 1024;
    attr->sharedSizeBytes    = 48 * 1024;
    attr->constSizeBytes     = 64 * 1024;
    attr->localSizeBytes     = 0;
    attr->maxDynamicSharedSizeBytes = 48 * 1024;
    return cudaSuccess;
}

cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value)
{
    (void)func; (void)attr; (void)value;
    return cudaSuccess;
}

cudaError_t cudaLaunchKernel(const void *func,
                             int gridX, int gridY, int gridZ,
                             int blockX, int blockY, int blockZ,
                             unsigned int sharedMem,
                             cudaStream_t stream,
                             void **kernelParams, void **extra)
{
    (void)func;
    (void)stream; (void)kernelParams; (void)extra;
    {
        cudaError_t v = fg_validate_launch(gridX, gridY, gridZ,
                                           blockX, blockY, blockZ, sharedMem);
        if (v != cudaSuccess) return fg_record(v); /* sticky: see fg_record */
    }
    if (getenv("FAKEGPU_DEBUG_REGISTRY")) {
        const char *n = fg_handle_find_name(func);
        fprintf(stderr, "[fg-launch] func=%p name=%s grid=(%d,%d,%d) "
                "block=(%d,%d,%d) shmem=%u\n",
                func, n ? n : "(unresolved)",
                gridX, gridY, gridZ, blockX, blockY, blockZ, sharedMem);
    }
    /* gpu-sim-urr: resolve func -> kernel_name and hand the launch to the
     * simulator. Unregistered func resolves NULL -> scalar default. */
    fg_simulate_kernel_resolved(fg_handle_find_name(func),
                                gridX, gridY, gridZ, blockX, blockY, blockZ,
                                sharedMem);
    return cudaSuccess;
}

cudaError_t cudaLaunchHostFunc(cudaStream_t stream, void (*fn)(void *), void *userData)
{
    (void)stream;
    /* Host functions execute on the host immediately; apply configured latency. */
    const struct fg_config *c = fg_cfg();
    if (c->hostfunc_latency_us > 0)
        fg_sleep_ns((long)c->hostfunc_latency_us * 1000L);
    if (fn) fn(userData);
    return cudaSuccess;
}

/* ---- error state, occupancy, symbols (gpu-sim-u2o.6.1.3) ----------------- */

/* Error-state accessors. g_last_error/fg_record are defined with the launch
 * validator above, because launch validation is the only thing a fake can
 * actually detect as wrong. cudaGetLastError reports-and-clears and
 * cudaDeviceSynchronize surfaces it, exactly like real cudart. */



cudaError_t cudaGetLastError(void)
{
    cudaError_t e = g_last_error;
    g_last_error = cudaSuccess;      /* reading clears, as in real cudart */
    return e;
}

cudaError_t cudaPeekAtLastError(void)
{
    return g_last_error;              /* peeking does not clear */
}

cudaError_t cudaThreadExit(void)        { return cudaSuccess; }
cudaError_t cudaProfilerStart(void)     { return cudaSuccess; }
cudaError_t cudaProfilerStop(void)      { return cudaSuccess; }

cudaError_t cudaFuncSetCacheConfig(const void *func, int cacheConfig)
{
    (void)func; (void)cacheConfig;
    return cudaSuccess;
}

cudaError_t cudaFuncSetSharedMemConfig(const void *func, size_t size, int config)
{
    (void)func; (void)size; (void)config;
    return cudaSuccess;
}

/* Occupancy: report one full wave of blocks per SM (the fake's device has
 * 1024 threads/SM), which is what a compiler-generated driver expects to
 * read. It is a plausible constant, not a modelled number. */
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                          const void *func,
                                                          int blockSize,
                                                          size_t dynamicSMem)
{
    (void)func; (void)dynamicSMem;
    if (!numBlocks) return cudaErrorInvalidValue;
    if (blockSize < 1) { *numBlocks = 0; return cudaSuccess; }
    *numBlocks = 1024 / blockSize;
    return cudaSuccess;
}

cudaError_t cudaGetSymbolAddress(void **symbolPtr, const void *symbol)
{
    if (!symbolPtr) return cudaErrorInvalidValue;
    /* The host-side stub pointer IS the only address that exists in a fake;
     * device-code bodies never exist. Documented fake-by-design behaviour. */
    *symbolPtr = (void *)symbol;
    return cudaSuccess;
}

cudaError_t cudaGetSymbolInfo(void *symbolInfo, size_t *size, const void *symbol)
{
    (void)symbolInfo; (void)symbol;
    if (!size) return cudaErrorInvalidValue;
    *size = 0; /* no device image in a fake */
    return cudaSuccess;
}

/* ---- cudart static-registration / launch surface (gpu-sim-u2o.6.1.3) ----- */

/* nvcc's generated host stub calls these when a TU is built with
 * -cudart=none. They are bookkeeping, not API: the fat-binary handle is a
 * registry entry, function/variable registration is recorded and ignored, and
 * a launch is validated + latency-injected exactly like cudaLaunchKernel. */

static void **g_fat_binary = NULL; /* one handle; a fake loads "everything" */

void **__cudaRegisterFatBinary(void *fatCubin)
{
    (void)fatCubin;
    fg_handle_t *h = fg_handle_make(8); /* kind 8 = fat binary */
    if (!h) return NULL;
    g_fat_binary = (void **)h;
    return g_fat_binary;
}

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
    (void)fatCubinHandle; /* nothing to load, unload or count */
}

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    (void)fatCubinHandle;
    if (fatCubinHandle && *fatCubinHandle == g_fat_binary) g_fat_binary = NULL;
}

void __cudaRegisterFunction(void **fatCubinHandle,
                            const char *hostFun,
                            char *deviceFun,
                            const char *deviceName,
                            int thread_limit,
                            uint3 *tid,
                            uint3 *bid,
                            dim3 *bDim,
                            dim3 *gDim,
                            int *wSize)
{
    (void)fatCubinHandle;
    (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
    /* gpu-sim-urr: stop discarding the identity. Record BOTH keys the launch
     * paths may present: deviceFun (cudaLaunchKernel's func) and hostFun
     * (the stub __cudaGetKernel/__cudaLaunchKernel hand back). Which one the
     * generated stub actually passes is verified empirically, not assumed. */
    if (deviceName) {
        if (deviceFun) fg_handle_record_name(deviceFun, deviceName);
        if (hostFun && hostFun != (const char *)deviceFun)
            fg_handle_record_name(hostFun, deviceName);
    }
    if (getenv("FAKEGPU_DEBUG_REGISTRY"))
        fprintf(stderr, "[fg-register] deviceFun=%p hostFun=%p name=%s\n",
                (void *)deviceFun, (void *)hostFun,
                deviceName ? deviceName : "(null)");
    /* No device code is loaded, so a later __cudaGetKernel on this stub is
     * answered by the identity map below. */
}

void __cudaRegisterVar(void **fatCubinHandle, char *hostVar, char *deviceAddress,
                       const char *deviceName, int ext, size_t size,
                       int constant, int global)
{
    (void)fatCubinHandle; (void)hostVar; (void)deviceAddress; (void)deviceName;
    (void)ext; (void)size; (void)constant; (void)global;
}

void __cudaRegisterManagedVar(void **fatCubinHandle, void **hostVarPtrAddress,
                              char *deviceAddress, const char *deviceName,
                              int ext, size_t size, int constant, int global)
{
    (void)fatCubinHandle; (void)hostVarPtrAddress; (void)deviceAddress;
    (void)deviceName; (void)ext; (void)size; (void)constant; (void)global;
}

char __cudaInitModule(void **fatCubinHandle)
{
    (void)fatCubinHandle;
    return '\0';
}

/* nvcc pushes the <<<grid, block, shmem, stream>>> config before the launch
 * and pops it after. We remember it so __cudaPopCallConfiguration can hand
 * the values back (real cudart does the same, and generated code does read
 * them on some paths). */
static dim3  g_cfg_grid  = { 1, 1, 1 };
static dim3  g_cfg_block = { 1, 1, 1 };
static size_t g_cfg_shared = 0;

unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim,
                                     size_t sharedMem,
                                     struct CUstream_st *stream)
{
    (void)stream;
    g_cfg_grid = gridDim;
    g_cfg_block = blockDim;
    g_cfg_shared = sharedMem;
    return 0u; /* real cudart returns 0 on success */
}

cudaError_t __cudaPopCallConfiguration(dim3 *gridDim, dim3 *blockDim,
                                       size_t *sharedMem,
                                       struct CUstream_st **stream)
{
    if (gridDim)   *gridDim = g_cfg_grid;
    if (blockDim)  *blockDim = g_cfg_block;
    if (sharedMem) *sharedMem = g_cfg_shared;
    if (stream)    *stream = NULL;
    return cudaSuccess;
}

cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *stub)
{
    if (!kernel) return cudaErrorInvalidValue;
    *kernel = (cudaKernel_t)stub; /* identity: the stub stands in for the device image */
    return cudaSuccess;
}

static cudaError_t fg_launch_through_dyn_api(const char *kernel_name,
                                             dim3 gridDim, dim3 blockDim,
                                             void **args, size_t sharedMem,
                                             cudaStream_t stream)
{
    (void)args; (void)stream;
    {
        cudaError_t v = fg_validate_launch((int)gridDim.x, (int)gridDim.y,
                                           (int)gridDim.z, (int)blockDim.x,
                                           (int)blockDim.y, (int)blockDim.z,
                                           (unsigned int)sharedMem);
        if (v != cudaSuccess) return fg_record(v); /* sticky: see fg_record */
    }
    fg_simulate_kernel_resolved(kernel_name,
                                (int)gridDim.x, (int)gridDim.y, (int)gridDim.z,
                                (int)blockDim.x, (int)blockDim.y, (int)blockDim.z,
                                (unsigned int)sharedMem);
    return cudaSuccess;
}

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                               void **args, size_t sharedMem, cudaStream_t stream)
{
    (void)kernel;
    if (getenv("FAKEGPU_DEBUG_REGISTRY")) {
        const char *n = fg_handle_find_name(kernel);
        fprintf(stderr, "[fg-launch-dyn] kernel=%p name=%s\n",
                (void *)kernel, n ? n : "(unresolved)");
    }
    /* gpu-sim-urr: kernel IS the stub (__cudaGetKernel identity), resolved
     * through the hostFun entries recorded at registration. */
    return fg_launch_through_dyn_api(fg_handle_find_name(kernel),
                                     gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim,
                                    dim3 blockDim, void **args, size_t sharedMem,
                                    cudaStream_t stream)
{
    (void)kernel;
    if (getenv("FAKEGPU_DEBUG_REGISTRY")) {
        const char *n = fg_handle_find_name(kernel);
        fprintf(stderr, "[fg-launch-dyn] kernel=%p name=%s\n",
                (void *)kernel, n ? n : "(unresolved)");
    }
    return fg_launch_through_dyn_api(fg_handle_find_name(kernel),
                                     gridDim, blockDim, args, sharedMem, stream);
}

/* ---- modules / driver ---------------------------------------------------- */

cudaError_t cudaModuleLoad(cudaModule_t *module, const char *path)
{
    return fg_module_load(module, path);
}

cudaError_t cudaModuleLoadData(cudaModule_t *module, const void *image)
{
    return fg_module_load(module, image ? "(inline)" : "(null)");
}

cudaError_t cudaModuleLoadDataEx(cudaModule_t *module, const void *image,
                                 unsigned int numOptions, int *options,
                                 void *optionsBuf)
{
    (void)numOptions; (void)options; (void)optionsBuf;
    return fg_module_load(module, image ? "(inline)" : "(null)");
}

cudaError_t cudaModuleUnload(cudaModule_t module)
{
    (void)module;
    return cudaSuccess;
}

cudaError_t cudaModuleGetFunction(cudaFunction_t *function, cudaModule_t module,
                                  const char *name)
{
    if (!function) return cudaErrorInvalidValue;
    (void)module; (void)name;
    fg_handle_t *h = fg_handle_make(4); /* kind 4 = function */
    *function = h ? fg_handle_ptr(h) : NULL;
    return cudaSuccess;
}

/* driver API mirrors the runtime ones (same fake pool/handles) */
cudaError_t cuMemAlloc(void **devPtr, size_t size)                 { return cudaMalloc(devPtr, size); }
cudaError_t cuMemAllocAsync(void **devPtr, size_t size, CUstream stream) { (void)stream; return cudaMalloc(devPtr, size); }
cudaError_t cuMemFree(void *devPtr)                              { return cudaFree(devPtr); }
cudaError_t cuMemFreeAsync(void *devPtr, CUstream stream)        { (void)stream; return cudaFree(devPtr); }
cudaError_t cuMemAllocHost(void **ptr, size_t bytes)             { return cudaMallocHost(ptr, bytes); }
cudaError_t cuMemsetD8(void *devPtr, unsigned char value, size_t count) { return cudaMemsetD8(devPtr, value, count); }
cudaError_t cuMemcpyHtoD(void *dst, const void *src, size_t count) { return cudaMemcpyHtoD(dst, src, count); }
cudaError_t cuMemcpyDtoH(void *dst, const void *src, size_t count)   { return cudaMemcpyDtoH(dst, src, count); }
cudaError_t cuMemcpyDtoD(void *dst, const void *src, size_t count)   { return cudaMemcpyDtoD(dst, src, count); }
cudaError_t cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice dev) { (void)flags; (void)dev; if(!ctx) return cudaErrorInvalidValue; fg_handle_t *h=fg_handle_make(5); *ctx = h?fg_handle_ptr(h):NULL; return cudaSuccess; }
cudaError_t cuCtxDestroy(CUcontext ctx)                          { (void)ctx; return cudaSuccess; }
cudaError_t cuCtxPushCurrent(CUcontext ctx)                      { (void)ctx; return cudaSuccess; }
cudaError_t cuCtxPopCurrent(CUcontext *ctx)                      { if(!ctx) return cudaErrorInvalidValue; fg_handle_t *h=fg_handle_make(5); *ctx = h?fg_handle_ptr(h):NULL; return cudaSuccess; }
cudaError_t cuDeviceGet(CUdevice *dev, int ordinal) {
    if (!dev) return cudaErrorInvalidValue;
    struct fg_config *c = fg_cfg();
    if (ordinal < 0 || ordinal >= c->num_devices) return (cudaError_t)cudaErrorInvalidDevice;
    fg_handle_t *h = fg_handle_make(6); /* kind 6 = device */
    *dev = h ? fg_handle_ptr(h) : NULL;
    return cudaSuccess;
}
cudaError_t cuDeviceGetCount(int *count)                          { return cudaGetDeviceCount(count); }
cudaError_t cuDeviceGetAttribute(int *pi, int attr, CUdevice dev) { (void)dev; return cudaDeviceGetAttribute(pi, attr, 0); }

cudaError_t cuModuleLoad(CUmodule *module, const char *path)      { return fg_module_load(module, path); }
cudaError_t cuModuleLoadData(CUmodule *module, const void *image) { return fg_module_load(module, image?"(inline)":"(null)"); }
cudaError_t cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, const int *options, const void *optionsBuf) { (void)numOptions; (void)options; (void)optionsBuf; return fg_module_load(module, image?"(inline)":"(null)"); }
cudaError_t cuModuleUnload(CUmodule module)                       { (void)module; return cudaSuccess; }
cudaError_t cuModuleGetFunction(CUfunction *function, CUmodule module, const char *name) { return cudaModuleGetFunction(function, module, name); }

cudaError_t cuLaunchKernel(CUfunction f,
                           int gridX, int gridY, int gridZ,
                           int blockX, int blockY, int blockZ,
                           unsigned int sharedMem, CUstream stream,
                           void **kernelParams, void **extra)
{
    (void)f;
    (void)stream; (void)kernelParams; (void)extra;
    {
        cudaError_t v = fg_validate_launch(gridX, gridY, gridZ,
                                           blockX, blockY, blockZ, sharedMem);
        if (v != cudaSuccess) return fg_record(v); /* sticky: see fg_record */
    }
    /* gpu-sim-urr, documented NOT applicable: CUfunction handles from
     * cuModuleGetFunction are opaque kind-4 registry entries, never stubs,
     * so no registration key exists for them. NULL name -> scalar default.
     * (Driver-API launches are outside the symbol-keyed design.) */
    fg_simulate_kernel_resolved(NULL,
                                gridX, gridY, gridZ, blockX, blockY, blockZ,
                                sharedMem);
    return cudaSuccess;
}

/* unused placeholders referenced only to keep symbols non-empty */
void cudaDeviceProp_(void) {}
void cudaGraphicsMappingResource(void) {}

/* ---- shared helpers ------------------------------------------------------ */

int fg_module_load(void **out, const char *name)
{
    (void)name;
    if (!out) return cudaErrorInvalidValue;
    fg_handle_t *h = fg_handle_make(7); /* kind 7 = module */
    *out = h ? fg_handle_ptr(h) : NULL;
    return *out != NULL ? cudaSuccess : (int)cudaErrorInvalidDevice;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device)
{
    if (!pciBusId || len <= 0) return cudaErrorInvalidValue;
    struct fg_config *c = fg_cfg();
    if (device < 0 || device >= c->num_devices) return (cudaError_t)cudaErrorInvalidDevice;
    snprintf(pciBusId, len, "0000:%02x:00.0", device & 0xff);
    return cudaSuccess;
}
