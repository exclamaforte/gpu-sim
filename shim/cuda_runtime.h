#ifndef FAKEGPU_CUDA_RUNTIME_H
#define FAKEGPU_CUDA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * fakegpu's CUDA shim. Declares the subset of the CUDA runtime + driver API
 * that the interposer (libfakegpu.so) provides. Workloads compiled for CPU-only
 * CI include this header instead of <cuda_runtime.h>; on a GPU box you would
 * instead compile against the real CUDA headers and run the workload WITHOUT
 * LD_PRELOAD against real hardware.
 */

/* Opaque CUDA stream/context struct tags, declared at file scope (not inside
 * a parameter list): a struct first named inside a parameter list gets
 * prototype scope, so the next declaration naming it would be a different
 * type and the compiler would reject the pair as conflicting. */
struct CUstream_st;
struct CUcontext_st;

/* ---- types needed by the __cuda* registration surface (u2o.6.1.3) -------- */


/* uint3: only used as an out-parameter by __cudaRegisterFunction. */
typedef struct uint3 {
    unsigned int x, y, z;
} uint3;

/* opaque kernel handle used by __cudaGetKernel / __cudaLaunchKernel */
typedef const void *cudaKernel_t;

/* NOTE (gpu-sim-u2o.6.1.1): an earlier revision tried to make this header
 * shadow-safe inside nvcc compilation (keyword macros, uint3, dim3 ctor,
 * C++ defaults) so preflight could compile drivers with -Ishim. That path
 * is abandoned: nvcc force-includes cuda_runtime.h AND pulls its own
 * headers through wrapper chains, making a shadowing header unworkable.
 * Preflight smoke binaries compile against REAL headers and link
 * libfakegpu, running GPU-free against the direct link instead (gpu-sim-
 * u2o.6.5.1: no LD_PRELOAD -- against nvcc's default -cudart=static it
 * would be a no-op, so smoke binaries compile -cudart=none and link no
 * cudart at all). This header
 * stays gcc-only by design.

 * The fake path that DOES work, and is now the mandated preflight route
 * (gpu-sim-u2o.6.1.3): compile a generated driver with
 *   nvcc -cudart=none <src> -o <exe> -L<repo> -lfakegpu
 * The binary then links NEITHER libcudart NOR libcuda, so it is
 * structurally incapable of touching a real GPU, and every cudart symbol
 * it references must resolve inside libfakegpu -- there is no per-symbol
 * fallthrough to a real CUDA library. gpu-sim-u2o.6.1.1 proved the
 * previous "LD_PRELOAD interposition" claim false: nvcc defaults to
 * -cudart=static, so cudart's cudaMalloc is defined INSIDE the executable
 * and LD_PRELOAD cannot interpose a symbol the binary already defines.
 *
 * The __cuda* registration/launch surface below exists for exactly that
 * link mode: nvcc's generated host stub references those symbols
 * directly, so libfakegpu must define them itself.
 */

/* ---- types --------------------------------------------------------------- */

typedef int cudaError_t;

#define cudaSuccess                  ((cudaError_t)0)
#define cudaErrorInvalidValue        ((cudaError_t)1)
#define cudaErrorInvalidDevice       ((cudaError_t)2)  /* also used for mem alloc */
#define cudaErrorInitializationError ((cudaError_t)3)
#define cudaErrorInvalidConfiguration ((cudaError_t)9) /* matches real CUDA */
#define cudaErrorUnknownError        ((cudaError_t)999)

/* dim3 (matches real CUDA layout: x, y, z). Plain C struct: nvcc-side
 * code uses the real headers' dim3, never this one. */
typedef struct dim3 {
    unsigned int x, y, z;
} dim3;

/* cudaMemcpyKind (matches real CUDA numeric values) */
enum cudaMemcpyKind {
    cudaMemcpyHostToHost     = 0,
    cudaMemcpyHostToDevice   = 1,
    cudaMemcpyDeviceToHost   = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault        = 4,
};

/* opaque handles — generated small integer handles cast to pointer */
typedef void *cudaStream_t;
typedef void *cudaEvent_t;
typedef void *cudaModule_t;
typedef void *cudaFunction_t;
typedef void *cudaIpcMemHandle_t;
typedef void *cudaGraphicsResource_t;

/* driver API opaque handles */
typedef void *CUdevice;
typedef void *CUcontext;
typedef void *CUstream;
typedef void *CUevent;
typedef void *CUfunction;
typedef void *CUmodule;
typedef void *CUresult;

#define CU_RESULT_OK ((CUresult)(0))

/* gpu-sim-u2o.19. THESE VALUES WERE WRONG, and the symptom was that
 * cudaDeviceGetAttribute returned 0 to every caller built against real CUDA
 * headers while the struct channel returned the right numbers -- so the fake
 * described two different parts depending on which channel you asked.
 *
 * The shim numbered these 1,2,3,4,5,6,7,10,11,75,76. Real CUDA 13.3 numbers
 * them 1,39,16,8,12,13,38,75,76. Only MaxThreadsPerBlock (1),
 * ComputeCapabilityMajor (75) and ComputeCapabilityMinor (76) coincided, which
 * is exactly the pattern observed: three of four probed attributes returned 0
 * and the one that worked was the one whose number matched.
 *
 * This is the FOURTH channel problem in this file, and the same lesson as the
 * three places maxThreadsPerMultiProcessor was written: a value (or a key)
 * duplicated across channels drifts, and the drift is undetectable by a caller
 * who has no way to know the other channel exists. 6.6.2's "one byte
 * convention in the file" again.
 *
 * Values read from CUDA 13.3's cuda_runtime.h. */
enum cudaDeviceAttr {
    cudaDevAttrMaxThreadsPerBlock           = 1,
    cudaDevAttrMaxRegistersPerBlock         = 12,
    cudaDevAttrClockRate                    = 13,
    cudaDevAttrMultiProcessorCount          = 16,
    cudaDevAttrMaxSharedMemoryPerBlock      = 8,
    cudaDevAttrL2CacheSize                  = 38,
    cudaDevAttrMaxThreadsPerMultiProcessor  = 39,
    /* gpu-sim-u2o.19 C5: values read from CUDA 13.3's driver_types.h (the
     * runtime header does not carry the numbers). 81/97, not guessed. */
    cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
    cudaDevAttrMaxSharedMemoryPerBlockOptin  = 97,
    cudaDevAttrComputeCapabilityMajor       = 75,
    cudaDevAttrComputeCapabilityMinor       = 76,
    cudaDevAttrTotalAmountOfConstantMemory  = 10,
};

enum cudaLimit { cudaLimitMallocHeapSize = 0 };

/* cudaDeviceProp — a stable, generous layout the shim & stub agree on.
   Real CUDA workloads in CI are built against THIS header, so layout is
   consistent. We expose the few fields commonly read by programs. */
struct cudaDeviceProp {
    char   name[256];
    size_t totalGlobalMem;
    size_t sharedMemPerBlock;
    int    maxThreadsPerBlock;
    int    maxThreadsPerMultiProcessor;
    int    multiProcessorCount;
    int    clockRate;
    int    isMultiGpuBoard;
    int    multiGpuBoardGroupID;
    int    maxThreadsPerBlockPad;   /* padding for ABI stability */
    size_t totalConstMem;
    size_t totalL2Cache;
    int    pciBusID;
    int    pciDeviceID;
    int    pciDomainID;
    int    major;
    int    minor;
    size_t memPitch;
    int    unifiedAddressing;
    int    reserved[84];            /* keep struct large enough */
};

struct cudaFuncAttributes {
    int        maxThreadsPerBlock;
    int        sharedSizeBytes;
    int        constSizeBytes;
    int        localSizeBytes;
    int        maxDynamicSharedSizeBytes;
    int        reserved;
};

/* ---- runtime memory ------------------------------------------------------ */

cudaError_t cudaMalloc(void **devPtr, size_t size);
cudaError_t cudaFree(void *devPtr);
cudaError_t cudaMallocAsync(void **devPtr, size_t size, cudaStream_t stream);
cudaError_t cudaFreeAsync(void *devPtr, cudaStream_t stream);
cudaError_t cudaMallocManaged(void **devPtr, size_t size, unsigned int flags);
cudaError_t cudaMallocHost(void **ptr, size_t size);
cudaError_t cudaFreeHost(void *ptr);
cudaError_t cudaHostAlloc(void **pHost, size_t bytes, unsigned int flags);
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void *devPtr, int value, size_t count, cudaStream_t stream);
cudaError_t cudaMemsetD8(void *devPtr, unsigned char value, size_t count);

/* ---- runtime memcpy ------------------------------------------------------ */

cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, int kind);
cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t count, int kind,
                            cudaStream_t stream);
cudaError_t cudaMemcpyHtoD(void *dstDevice, const void *srcHost, size_t count);
cudaError_t cudaMemcpyDtoH(void *dstHost, const void *srcDevice, size_t count);
cudaError_t cudaMemcpyDtoD(void *dstDevice, const void *srcDevice, size_t count);
cudaError_t cudaMemcpyHtoDAsync(void *dstDevice, const void *srcHost, size_t count,
                                cudaStream_t stream);
cudaError_t cudaMemcpyDtoHAsync(void *dstHost, const void *srcDevice, size_t count,
                                cudaStream_t stream);

/* ---- streams / events ---------------------------------------------------- */

cudaError_t cudaStreamCreate(cudaStream_t *pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *pStream, unsigned int flags,
                                         int priority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags);

cudaError_t cudaEventCreate(cudaEvent_t *event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *event, unsigned int flags);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);

/* human-readable error strings (subset of real CUDA messages) */
const char *cudaGetErrorString(cudaError_t error);

/* ---- device / context ---------------------------------------------------- */

cudaError_t cudaGetDeviceCount(int *count);
cudaError_t cudaGetDevice(int *device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaDeviceSetLimit(int limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t *value, int limit);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device);
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp *prop, int device);
cudaError_t cudaDeviceGetAttribute(int *pi, int attr, int device);

/* ---- functions / kernels ------------------------------------------------- */

cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes *attr, const void *func);
cudaError_t cudaFuncSetAttribute(const void *func, int attr, int value);
cudaError_t cudaLaunchKernel(const void *func,
                             int gridX, int gridY, int gridZ,
                             int blockX, int blockY, int blockZ,
                             unsigned int sharedMem,
                             cudaStream_t stream,
                             void **kernelParams, void **extra);
cudaError_t cudaLaunchHostFunc(cudaStream_t stream,
                               void (*fn)(void *), void *userData);

/* ---- modules / driver ---------------------------------------------------- */

cudaError_t cudaModuleLoad(cudaModule_t *module, const char *path);
cudaError_t cudaModuleLoadData(cudaModule_t *module, const void *image);
cudaError_t cudaModuleLoadDataEx(cudaModule_t *module, const void *image,
                                 unsigned int numOptions, int *options,
                                 void *optionsBuf);
cudaError_t cudaModuleUnload(cudaModule_t module);
cudaError_t cudaModuleGetFunction(cudaFunction_t *function, cudaModule_t module,
                                  const char *name);

cudaError_t cuMemAlloc(void **devPtr, size_t size);
cudaError_t cuMemAllocAsync(void **devPtr, size_t size, CUstream stream);
cudaError_t cuMemFree(void *devPtr);
cudaError_t cuMemFreeAsync(void *devPtr, CUstream stream);
cudaError_t cuMemAllocHost(void **ptr, size_t bytes);
cudaError_t cuMemsetD8(void *devPtr, unsigned char value, size_t count);
cudaError_t cuMemcpyHtoD(void *dst, const void *src, size_t count);
cudaError_t cuMemcpyDtoH(void *dst, const void *src, size_t count);
cudaError_t cuMemcpyDtoD(void *dst, const void *src, size_t count);
cudaError_t cuCtxCreate(CUcontext *ctx, unsigned int flags, CUdevice dev);
cudaError_t cuCtxDestroy(CUcontext ctx);
cudaError_t cuCtxPushCurrent(CUcontext ctx);
cudaError_t cuCtxPopCurrent(CUcontext *ctx);
cudaError_t cuDeviceGet(CUdevice *dev, int ordinal);
cudaError_t cuDeviceGetCount(int *count);
cudaError_t cuDeviceGetAttribute(int *pi, int attr, CUdevice dev);
cudaError_t cuModuleLoad(CUmodule *module, const char *path);
cudaError_t cuModuleLoadData(CUmodule *module, const void *image);
cudaError_t cuModuleLoadDataEx(CUmodule *module, const void *image,
                               unsigned int numOptions, const int *options,
                               const void *optionsBuf);
cudaError_t cuModuleUnload(CUmodule module);
cudaError_t cuModuleGetFunction(CUfunction *function, CUmodule module,
                                const char *name);
cudaError_t cuLaunchKernel(CUfunction f,
                           int gridX, int gridY, int gridZ,
                           int blockX, int blockY, int blockZ,
                           unsigned int sharedMem, CUstream stream,
                           void **kernelParams, void **extra);

/* ---- error state --------------------------------------------------------- */

/* The error/state family generated drivers call after every API call.
 * A fake has no sticky error state: it always reports success, and the
 * peek variants never clear anything. */

cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
cudaError_t cudaThreadExit(void);

/* ---- occupancy / symbols ------------------------------------------------- */

cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks,
                                                          const void *func,
                                                          int blockSize,
                                                          size_t dynamicSMem);
cudaError_t cudaGetSymbolAddress(void **symbolPtr, const void *symbol);
cudaError_t cudaGetSymbolInfo(void *symbolInfo, size_t *size, const void *symbol);
cudaError_t cudaFuncSetCacheConfig(const void *func, int cacheConfig);
cudaError_t cudaFuncSetSharedMemConfig(const void *func, size_t size, int config);
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);

/* ---- cudart static-registration / launch surface (gpu-sim-u2o.6.1.3) ----- */

/* These are what nvcc's generated host stub references when a translation
 * unit is compiled with -cudart=none. They are registration bookkeeping, not
 * API: no-ops returning success/0 are the correct fake behaviour. The
 * signatures are copied from the CUDA toolkit's crt/host_runtime.h and
 * crt/device_functions.h so the ABI matches the real caller. */

void  **__cudaRegisterFatBinary(void *fatCubin);
void     __cudaRegisterFatBinaryEnd(void **fatCubinHandle);
void     __cudaUnregisterFatBinary(void **fatCubinHandle);
void     __cudaRegisterFunction(void **fatCubinHandle,
                                const char *hostFun,
                                char *deviceFun,
                                const char *deviceName,
                                int thread_limit,
                                uint3 *tid,
                                uint3 *bid,
                                dim3 *bDim,
                                dim3 *gDim,
                                int *wSize);
void     __cudaRegisterVar(void **fatCubinHandle,
                           char *hostVar,
                           char *deviceAddress,
                           const char *deviceName,
                           int ext,
                           size_t size,
                           int constant,
                           int global);
void     __cudaRegisterManagedVar(void **fatCubinHandle,
                                   void **hostVarPtrAddress,
                                   char *deviceAddress,
                                   const char *deviceName,
                                   int ext,
                                   size_t size,
                                   int constant,
                                   int global);
char     __cudaInitModule(void **fatCubinHandle);

unsigned    __cudaPushCallConfiguration(dim3 gridDim,
                                         dim3 blockDim,
                                         size_t sharedMem,
                                         struct CUstream_st *stream);
cudaError_t __cudaPopCallConfiguration(dim3 *gridDim,
                                       dim3 *blockDim,
                                       size_t *sharedMem,
                                       struct CUstream_st **stream);
cudaError_t __cudaGetKernel(cudaKernel_t *kernel, const void *stub);
cudaError_t __cudaLaunchKernel(cudaKernel_t kernel,
                               dim3 gridDim,
                               dim3 blockDim,
                               void **args,
                               size_t sharedMem,
                               cudaStream_t stream);
cudaError_t __cudaLaunchKernel_ptsz(cudaKernel_t kernel,
                                    dim3 gridDim,
                                    dim3 blockDim,
                                    void **args,
                                    size_t sharedMem,
                                    cudaStream_t stream);

/* ---- helpers used by workloads & interop --------------------------------- */

cudaError_t cudaStreamAddCallback(cudaStream_t stream,
                                  void (*callback)(cudaStream_t, int, void *),
                                  void *userData);
typedef void (*cudaStreamCallback_t)(cudaStream_t, int, void *);

void cudaDeviceProp_(void); /* placeholder, unused */

#ifdef __cplusplus
}
#endif

#endif /* FAKEGPU_CUDA_RUNTIME_H */
