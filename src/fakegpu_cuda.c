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
#include <string.h>

/* Write `count` random bytes to `dst` at an API boundary. src is not touched. */
static void fg_write_random(void *dst, size_t count)
{
    if (!dst || count == 0) return;
    fg_fill(dst, count);
}

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
    (void)value;
    if (devPtr && count) fg_fill(devPtr, count);
    return cudaSuccess;
}

cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream)
{
    (void)stream;
    return cudaMemset(devPtr, value, count);
}

cudaError_t cudaMemsetD8(void *devPtr, unsigned char value, size_t count)
{
    (void)value;
    if (devPtr && count) fg_fill(devPtr, count);
    return cudaSuccess;
}

/* ---- memcpy --------------------------------------------------------------- */

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind)
{
    (void)src; (void)kind;
    fg_simulate_memcpy(count);
    fg_write_random(dst, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind,
                            cudaStream_t stream)
{
    (void)src; (void)kind; (void)stream;
    fg_simulate_memcpy(count);
    fg_write_random(dst, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyHtoD(void *dstDevice, const void *srcHost, size_t count)
{
    (void)srcHost;
    fg_simulate_memcpy(count);
    fg_write_random(dstDevice, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyDtoH(void *dstHost, const void *srcDevice, size_t count)
{
    (void)srcDevice;
    fg_simulate_memcpy(count);
    fg_write_random(dstHost, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyDtoD(void *dstDevice, const void *srcDevice, size_t count)
{
    (void)srcDevice;
    fg_simulate_memcpy(count);
    fg_write_random(dstDevice, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyHtoDAsync(void *dstDevice, const void *srcHost, size_t count,
                                cudaStream_t stream)
{
    (void)srcHost; (void)stream;
    fg_simulate_memcpy(count);
    fg_write_random(dstDevice, count);
    return cudaSuccess;
}

cudaError_t cudaMemcpyDtoHAsync(void *dstHost, const void *srcDevice, size_t count,
                                cudaStream_t stream)
{
    (void)srcDevice; (void)stream;
    fg_simulate_memcpy(count);
    fg_write_random(dstHost, count);
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

cudaError_t cudaDeviceSetLimit(int limit, size_t value)
{
    (void)limit; (void)value;
    return cudaSuccess;
}

cudaError_t cudaDeviceGetLimit(size_t *value, int limit)
{
    (void)limit;
    if (!value) return cudaErrorInvalidValue;
    *value = 0;
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
    prop->sharedMemPerBlock       = 48 * 1024;
    prop->maxThreadsPerBlock      = 1024;
    prop->maxThreadsPerMultiProcessor = 2048;
    prop->multiProcessorCount     = 170;   /* match real RTX 5090 */
    prop->clockRate               = 1590000;   /* kHz, fake */
    /* Real-layout overlay: third-party binaries (e.g. PyTorch) are built
     * against real CUDA headers, whose cudaDeviceProp layout differs from
     * shim/cuda_runtime.h. They read major/minor (and a few basics) at real
     * offsets (driver_types.h, CUDA 13): totalGlobalMem@288,
     * sharedMemPerBlock@296, maxThreadsPerBlock@320, totalConstMem@352,
     * major@360, minor@364, multiProcessorCount@384. Without this overlay
     * they read zeros -> compute capability 0.0 -> SIGFPE (gpu-sim-u2o.16).
     * All offsets are < shim sizeof (verified 696 by sizeof_check). */
    {
        unsigned char *base = (unsigned char *)prop;
        size_t total_mem = (size_t)c->device_mem_mb * 1024 * 1024;
        size_t shmem = 48 * 1024;
        size_t const_mem = 64 * 1024;
        size_t pitch = 2147483647u;
        size_t shmem_sm = (size_t)228 * 1024;   /* per-SM, matches RTX 5090 */
        int max_threads = 1024, major = 12, minor = 0, sm_count = 170;
        int regs_block = 65536, warp = 32, one = 1;
        int max_dim[3] = {1024, 1024, 64};
        int max_grid[3] = {2147483647, 65535, 65535};
        int max_thr_sm = 2048, regs_sm = 65536, l2 = 128 * 1024 * 1024;
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
    prop->totalL2Cache            = 4 * 1024 * 1024;
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
    case cudaDevAttrMaxThreadsPerMultiProcessor: *pi = 2048; break;
    case cudaDevAttrMultiProcessorCount:         *pi = 170;  break;   /* match real RTX 5090 */
    case cudaDevAttrMaxSharedMemoryPerBlock:     *pi = 48 * 1024; break;
    case cudaDevAttrClockRate:                   *pi = 1590000; break;
    case cudaDevAttrTotalGlobalMem:              *pi = (int)(c->device_mem_mb * 1024); break;
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
    fg_simulate_kernel();
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
    (void)fatCubinHandle; (void)hostFun; (void)deviceFun; (void)deviceName;
    (void)thread_limit; (void)tid; (void)bid; (void)bDim; (void)gDim; (void)wSize;
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

static cudaError_t fg_launch_through_dyn_api(dim3 gridDim, dim3 blockDim,
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
    fg_simulate_kernel();
    return cudaSuccess;
}

cudaError_t __cudaLaunchKernel(cudaKernel_t kernel, dim3 gridDim, dim3 blockDim,
                               void **args, size_t sharedMem, cudaStream_t stream)
{
    (void)kernel;
    return fg_launch_through_dyn_api(gridDim, blockDim, args, sharedMem, stream);
}

cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel, dim3 gridDim,
                                    dim3 blockDim, void **args, size_t sharedMem,
                                    cudaStream_t stream)
{
    (void)kernel;
    return fg_launch_through_dyn_api(gridDim, blockDim, args, sharedMem, stream);
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
    fg_simulate_kernel();
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
