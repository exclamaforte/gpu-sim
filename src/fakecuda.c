#define _GNU_SOURCE
#include "fakegpu_topology.h"

/* minimal declarations to avoid cuda_runtime.h conflict */
struct fg_config { int num_devices; int current_device; };
struct fg_config *fg_cfg(void);
void *fg_pool_alloc(size_t size);
void fg_fill(void *dst, size_t n);
void fg_sleep_ns(long ns);
#include "fakegpu_topology.h"
#include "cuda.h"

#include <dlfcn.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* ---- synthetic cudart export table (diagnostic) --------------------------
 * Static cudart resolves driver entry points through the table returned by
 * cuGetExportTable, not through dlsym. We cannot see the table's layout, so
 * each slot gets a tiny trampoline that stashes its SLOT INDEX in r10 and
 * jumps to fc_tbl_dispatch, which logs (slot, args) and returns a chosen
 * value. That turns the opaque ABI into something we can map slot -> name
 * empirically, without ever loading the real driver at run time.
 */
#include <sys/mman.h>

static long fc_tbl_calls[8192];

long fc_tbl_dispatch(long a1, long a2, long a3, long a4, long a5) {
    long idx;
    __asm__ volatile("mov %%r10, %0" : "=r"(idx));
    if (idx >= 0 && idx < 8192) fc_tbl_calls[idx]++;
    fprintf(stderr, "fakecuda: table slot %ld args=(%ld,%ld,%ld,%ld,%ld)\n",
            idx, a1, a2, a3, a4, a5);
    return 0;
}

static void *fc_export_table(void) {
    static void *tbl[8192];
    static int init = 0;
    if (!init) {
        size_t codelen = 8192 * 11;
        unsigned char *code = mmap(NULL, (codelen + 4095) & ~(size_t)4095,
                                   PROT_READ | PROT_WRITE | PROT_EXEC,
                                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (code == MAP_FAILED) {
            for (int i = 0; i < 8192; i++) tbl[i] = (void *)0; /* will crash: no exec mem */
        } else {
            for (int i = 0; i < 8192; i++) {
                unsigned char *p = code + (size_t)i * 11;
                p[0] = 0x41; p[1] = 0xba;                 /* mov $imm32, %r10d */
                p[2] = (unsigned char)(i & 0xff);
                p[3] = (unsigned char)((i >> 8) & 0xff);
                p[4] = (unsigned char)((i >> 16) & 0xff);
                p[5] = (unsigned char)((i >> 24) & 0xff);
                p[6] = 0xe9;                              /* jmp rel32 */
                long rel = (long)((unsigned char *)&fc_tbl_dispatch - (p + 11));
                p[7] = (unsigned char)(rel & 0xff);
                p[8] = (unsigned char)((rel >> 8) & 0xff);
                p[9] = (unsigned char)((rel >> 16) & 0xff);
                p[10] = (unsigned char)((rel >> 24) & 0xff);
                tbl[i] = p;
            }
        }
        init = 1;
    }
    return tbl;
}

CUresult cuGetExportTable(const void **ppExportTable, const CUuuid *pExportTableId) {
    if (pExportTableId) {
        const unsigned char *b = (const unsigned char *)pExportTableId;
        fprintf(stderr, "fakecuda: cuGetExportTable uuid=%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x\n",
                b[0],b[1],b[2],b[3],b[4],b[5],b[6],b[7],b[8],b[9],b[10],b[11],b[12],b[13],b[14],b[15]);
    }
    /* Diagnostic: GPUSIM_EXPORT_TABLE=1 returns a synthetic table whose slots
       all point at a stub returning SUCCESS. That avoids the NULL-fn crash and
       shows how far cudart gets, but the table layout is opaque, so the value
       read back for any entry is wrong. Default is the truthful NOT_SUPPORTED. */
    if (getenv("GPUSIM_EXPORT_TABLE")) {
        if (ppExportTable) *ppExportTable = fc_export_table();
        fprintf(stderr, "fakecuda: cuGetExportTable -> 0 (SUCCESS, synthetic stub table)\n");
        return CUDA_SUCCESS;
    }
    if (ppExportTable) *ppExportTable = NULL;
    fprintf(stderr, "fakecuda: cuGetExportTable -> 801 (NOT_SUPPORTED)\n");
    return CUDA_ERROR_NOT_SUPPORTED;
}

/* ---- device count / topology -------------------------------------------- */

static fg_topo_t *fc_topo = NULL;
static int fc_loaded = 0;

static void fc_load_topo(void) {
    if (fc_loaded) return;
    fc_loaded = 1;
    const char *p = getenv("GPUSIM_TOPOLOGY");
    if (p && *p) {
        char err[256];
        fc_topo = fg_topo_load(p, err, sizeof(err));
        if (!fc_topo) {
            fprintf(stderr, "fakecuda: corrupted topology: %s\n", err);
            exit(1);
        }
    }
}

static int fc_device_count(void) {
    fc_load_topo();
    if (fc_topo) return fc_topo->n_dev > 0 ? fc_topo->n_dev : 1;
    const char *e = getenv("GPUSIM_NUM_DEVICES");
    if (!e) e = getenv("FAKEGPU_NUM_DEVICES");
    if (e && *e) {
        int n = atoi(e);
        if (n > 0) return n;
    }
    return 1;
}

static int fc_sm_count(int dev) {
    fc_load_topo();
    if (fc_topo && dev < fc_topo->n_dev) return (int)fc_topo->devs[dev].sm_count;
    return 80;
}
static int fc_clock(int dev) {
    fc_load_topo();
    if (fc_topo && dev < fc_topo->n_dev) return (int)fc_topo->devs[dev].clock_mhz * 1000;
    return 1000000;
}
static unsigned long long fc_mem(int dev) {
    fc_load_topo();
    if (fc_topo && dev < fc_topo->n_dev) return fc_topo->devs[dev].mem_bytes;
    return (unsigned long long)16ULL * 1024 * 1024 * 1024;
}

/* ---- global fake context ------------------------------------------------- */

static CUcontext fc_ctx = (CUcontext)0x1;
static CUcontext fc_current = NULL;

/* ---- basic init ---------------------------------------------------------- */

CUresult cuInit(unsigned int Flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuInit");
    (void)Flags;
    fprintf(stderr,"fakecuda: cuInit\n");
    return CUDA_SUCCESS;
}

CUresult cuDriverGetVersion(int *driverVersion) {
    fprintf(stderr,"fakecuda: %s\n", "cuDriverGetVersion");
    if (!driverVersion) return CUDA_ERROR_INVALID_VALUE;
    *driverVersion = 13040;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetCount(int *count) {
    fprintf(stderr,"fakecuda: %s\n", "cuDeviceGetCount");
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    *count = fc_device_count();
    fprintf(stderr,"fakecuda: cuDeviceGetCount -> %d\n", *count);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGet(CUdevice *device, int ordinal) {
    fprintf(stderr,"fakecuda: %s\n", "cuDeviceGet");
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    if (ordinal < 0 || ordinal >= fc_device_count()) return CUDA_ERROR_INVALID_VALUE;
    *device = (CUdevice)ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDeviceGetAttribute");
    if (!pi) return CUDA_ERROR_INVALID_VALUE;
    switch (attrib) {
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK: *pi = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X: *pi = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y: *pi = 1024; break;
    case CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z: *pi = 64; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X: *pi = 2147483647; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y: *pi = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z: *pi = 65535; break;
    case CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK: *pi = 49152; break;
    case CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_WARP_SIZE: *pi = 32; break;
    case CU_DEVICE_ATTRIBUTE_MAX_PITCH: *pi = 2147483647; break;
    case CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_CLOCK_RATE: *pi = fc_clock(dev); break;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT: *pi = 512; break;
    case CU_DEVICE_ATTRIBUTE_GPU_OVERLAP: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *pi = fc_sm_count(dev); break;
    case CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT: *pi = 0; break;
    case CU_DEVICE_ATTRIBUTE_INTEGRATED: *pi = 0; break;
    case CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_MODE: *pi = 0; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH: *pi = 4096; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT: *pi = 4096; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH: *pi = 4096; break;
    case CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT: *pi = 512; break;
    case CU_DEVICE_ATTRIBUTE_CONCURRENT_KERNELS: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_ECC_ENABLED: *pi = 0; break;
    case CU_DEVICE_ATTRIBUTE_PCI_BUS_ID: *pi = dev; break; /* not real but nonzero */
    case CU_DEVICE_ATTRIBUTE_PCI_DEVICE_ID: *pi = dev; break;
    case CU_DEVICE_ATTRIBUTE_PCI_DOMAIN_ID: *pi = 0; break;
    case CU_DEVICE_ATTRIBUTE_TCC_DRIVER: *pi = 0; break;
    case CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE: *pi = 1215000; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH: *pi = 384; break;
    case CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE: *pi = 4194304; break;
    case CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR: *pi = 2048; break;
    case CU_DEVICE_ATTRIBUTE_STREAM_PRIORITIES_SUPPORTED: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_GLOBAL_L1_CACHE_SUPPORTED: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_LOCAL_L1_CACHE_SUPPORTED: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR: *pi = 9; break;
    case CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR: *pi = 0; break;
    case CU_DEVICE_ATTRIBUTE_CAN_TEX2D_GATHER: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH: *pi = 16384; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS: *pi = 2048; break;
    case CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING: *pi = 1; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_WIDTH: *pi = 16384; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_GATHER_HEIGHT: *pi = 16384; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH_ALTERNATE: *pi = 8192; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT_ALTERNATE: *pi = 8192; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH_ALTERNATE: *pi = 8192; break;
    case CU_DEVICE_ATTRIBUTE_TEXTURE_PITCH_ALIGNMENT: *pi = 32; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_WIDTH: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_WIDTH: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURECUBEMAP_LAYERED_LAYERS: *pi = 2046; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_WIDTH: *pi = 16384; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_WIDTH: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_HEIGHT: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_WIDTH: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_HEIGHT: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE3D_DEPTH: *pi = 16384; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_WIDTH: *pi = 16384; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE1D_LAYERED_LAYERS: *pi = 2048; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_WIDTH: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_HEIGHT: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACE2D_LAYERED_LAYERS: *pi = 2048; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_WIDTH: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_WIDTH: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_SURFACECUBEMAP_LAYERED_LAYERS: *pi = 2046; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LINEAR_WIDTH: *pi = 134217728; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_WIDTH: *pi = 65536; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_HEIGHT: *pi = 32768; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_LINEAR_PITCH: *pi = 1048544; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_WIDTH: *pi = 16384; break;
    case CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_MIPMAPPED_HEIGHT: *pi = 16384; break;
    default:
        fprintf(stderr, "fakecuda: cuDeviceGetAttribute unknown attrib=%d (returning 0)\n", (int)attrib);
        *pi = 0;
        break;
    }
    if (getenv("GPUSIM_DEBUG_DRV")) fprintf(stderr, "fakecuda: cuDeviceGetAttribute(%d) -> %d value=%d\n", (int)attrib, 0, *pi);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDeviceGetName");
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    snprintf(name, len, "FakeGPU %d", dev);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *pciBusId, int len, CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDeviceGetPCIBusId");
    if (!pciBusId || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    snprintf(pciBusId, len, "0000:%02x:00.0", dev & 0xff);
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDeviceGetUuid");
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    memset(uuid, 0, sizeof(*uuid));
    uuid->bytes[0] = (unsigned char)(dev + 1);
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDeviceTotalMem");
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    *bytes = (size_t)fc_mem(dev);
    return CUDA_SUCCESS;
}

/* ---- context ------------------------------------------------------------- */

CUresult cuCtxCreate(CUcontext *pctx, CUctxCreateParams *ctxCreateParams, unsigned int flags, CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxCreate");
    (void)ctxCreateParams; (void)flags; (void)dev;
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = fc_ctx;
    fc_current = fc_ctx;
    return CUDA_SUCCESS;
}
CUresult cuCtxDestroy(CUcontext ctx) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxDestroy"); (void)ctx; return CUDA_SUCCESS; }
CUresult cuCtxSetCurrent(CUcontext ctx) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxSetCurrent"); fc_current = ctx; return CUDA_SUCCESS; }
CUresult cuCtxGetCurrent(CUcontext *pctx) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxGetCurrent");
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = fc_current ? fc_current : fc_ctx;
    if (!fc_current) fc_current = fc_ctx;
    return CUDA_SUCCESS;
}
CUresult cuCtxPushCurrent(CUcontext ctx) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxPushCurrent"); fc_current = ctx; return CUDA_SUCCESS; }
CUresult cuCtxPopCurrent(CUcontext *pctx) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxPopCurrent");
    if (pctx) *pctx = fc_current;
    fc_current = NULL;
    return CUDA_SUCCESS;
}
CUresult cuCtxSynchronize(void) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxSynchronize"); return CUDA_SUCCESS; }
CUresult cuCtxGetDevice(CUdevice *device) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxGetDevice");
    if (!device) return CUDA_ERROR_INVALID_VALUE;
    *device = 0;
    return CUDA_SUCCESS;
}
CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version) {
    fprintf(stderr,"fakecuda: %s\n", "cuCtxGetApiVersion");
    (void)ctx;
    if (version) *version = 12000;
    return CUDA_SUCCESS;
}

/* primary context (NCCL often uses these) */
CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDevicePrimaryCtxRetain");
    (void)dev;
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    *pctx = fc_ctx;
    return CUDA_SUCCESS;
}
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDevicePrimaryCtxRelease"); (void)dev; return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxSetFlags(CUdevice dev, unsigned int flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuDevicePrimaryCtxSetFlags"); (void)dev; (void)flags; return CUDA_SUCCESS; }
CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned int *flags, int *active) {
    fprintf(stderr,"fakecuda: %s\n", "cuDevicePrimaryCtxGetState");
    (void)dev;
    if (flags) *flags = 0;
    if (active) *active = 1;
    return CUDA_SUCCESS;
}
CUresult cuDevicePrimaryCtxReset(CUdevice dev) {
    fprintf(stderr,"fakecuda: %s\n", "cuDevicePrimaryCtxReset"); (void)dev; return CUDA_SUCCESS; }

/* ---- memory -------------------------------------------------------------- */

CUresult cuMemAlloc(CUdeviceptr *dptr, size_t bytesize) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemAlloc");
    if (!dptr) return CUDA_ERROR_INVALID_VALUE;
    void *p = fg_pool_alloc(bytesize ? bytesize : 1);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    *dptr = (CUdeviceptr)p;
    return CUDA_SUCCESS;
}
CUresult cuMemFree(CUdeviceptr dptr) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemFree"); (void)dptr; return CUDA_SUCCESS; }
CUresult cuMemAllocHost(void **pp, size_t bytesize) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemAllocHost");
    if (!pp) return CUDA_ERROR_INVALID_VALUE;
    void *p = fg_pool_alloc(bytesize ? bytesize : 1);
    if (!p) return CUDA_ERROR_OUT_OF_MEMORY;
    *pp = p;
    return CUDA_SUCCESS;
}
CUresult cuMemFreeHost(void *p) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemFreeHost"); (void)p; return CUDA_SUCCESS; }
CUresult cuMemHostAlloc(void **pp, size_t bytesize, unsigned int Flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemHostAlloc");
    (void)Flags;
    return cuMemAllocHost(pp, bytesize);
}
CUresult cuMemHostRegister(void *p, size_t bytesize, unsigned int Flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemHostRegister");
    (void)p; (void)bytesize; (void)Flags;
    return CUDA_SUCCESS;
}
CUresult cuMemHostUnregister(void *p) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemHostUnregister"); (void)p; return CUDA_SUCCESS; }
CUresult cuMemcpyHtoD(CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemcpyHtoD");
    if (dstDevice && srcHost && ByteCount) memcpy((void *)dstDevice, srcHost, ByteCount);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyHtoDAsync(CUdeviceptr dst, const void *src, size_t n, CUstream s) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemcpyHtoDAsync"); (void)s; return cuMemcpyHtoD(dst, src, n); }
CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t ByteCount) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemcpyDtoH");
    if (dstHost && srcDevice && ByteCount) memcpy(dstHost, (void *)srcDevice, ByteCount);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyDtoHAsync(void *dst, CUdeviceptr src, size_t n, CUstream s) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemcpyDtoHAsync"); (void)s; return cuMemcpyDtoH(dst, src, n); }
CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t n) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemcpyDtoD");
    if (dst && src && n) memcpy((void *)dst, (void *)src, n);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyPeer(CUdeviceptr dst, CUcontext dstCtx, CUdeviceptr src, CUcontext srcCtx, size_t n) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemcpyPeer");
    (void)dstCtx; (void)srcCtx;
    if (dst && src && n) memcpy((void *)dst, (void *)src, n);
    return CUDA_SUCCESS;
}
CUresult cuMemcpyPeerAsync(CUdeviceptr dst, CUcontext dstCtx, CUdeviceptr src, CUcontext srcCtx, size_t n, CUstream s) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemcpyPeerAsync");
    (void)s;
    return cuMemcpyPeer(dst, dstCtx, src, srcCtx, n);
}
CUresult cuMemsetD8(CUdeviceptr dst, unsigned char uc, size_t n) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemsetD8");
    if (dst && n) memset((void *)dst, uc, n);
    return CUDA_SUCCESS;
}
CUresult cuMemsetD32(CUdeviceptr dst, unsigned int val, size_t n) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemsetD32"); (void)dst; (void)val; (void)n; return CUDA_SUCCESS; }

CUresult cuMemGetHandleForAddressRange(void *handle, CUdeviceptr dptr, size_t size, CUmemRangeHandleType handleType, unsigned long long flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuMemGetHandleForAddressRange");
    (void)handle; (void)dptr; (void)size; (void)handleType; (void)flags;
    return CUDA_SUCCESS;
}

/* ---- streams / events ---------------------------------------------------- */

CUresult cuStreamCreate(CUstream *phStream, unsigned int Flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuStreamCreate");
    if (!phStream) return CUDA_ERROR_INVALID_VALUE;
    *phStream = (CUstream)0x2;
    return CUDA_SUCCESS;
}
CUresult cuStreamCreateWithPriority(CUstream *phStream, unsigned int flags, int priority) {
    fprintf(stderr,"fakecuda: %s\n", "cuStreamCreateWithPriority");
    (void)priority;
    return cuStreamCreate(phStream, flags);
}
CUresult cuStreamDestroy(CUstream hStream) {
    fprintf(stderr,"fakecuda: %s\n", "cuStreamDestroy"); (void)hStream; return CUDA_SUCCESS; }
CUresult cuStreamSynchronize(CUstream hStream) {
    fprintf(stderr,"fakecuda: %s\n", "cuStreamSynchronize"); (void)hStream; return CUDA_SUCCESS; }
CUresult cuStreamQuery(CUstream hStream) {
    fprintf(stderr,"fakecuda: %s\n", "cuStreamQuery"); (void)hStream; return CUDA_SUCCESS; }
CUresult cuStreamWaitEvent(CUstream hStream, CUevent hEvent, unsigned int Flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuStreamWaitEvent"); (void)hStream; (void)hEvent; (void)Flags; return CUDA_SUCCESS; }

CUresult cuEventCreate(CUevent *phEvent, unsigned int Flags) {
    fprintf(stderr,"fakecuda: %s\n", "cuEventCreate");
    if (!phEvent) return CUDA_ERROR_INVALID_VALUE;
    *phEvent = (CUevent)0x3;
    return CUDA_SUCCESS;
}
CUresult cuEventDestroy(CUevent hEvent) {
    fprintf(stderr,"fakecuda: %s\n", "cuEventDestroy"); (void)hEvent; return CUDA_SUCCESS; }
CUresult cuEventRecord(CUevent hEvent, CUstream hStream) {
    fprintf(stderr,"fakecuda: %s\n", "cuEventRecord"); (void)hEvent; (void)hStream; return CUDA_SUCCESS; }
CUresult cuEventSynchronize(CUevent hEvent) {
    fprintf(stderr,"fakecuda: %s\n", "cuEventSynchronize"); (void)hEvent; return CUDA_SUCCESS; }
CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd) {
    fprintf(stderr,"fakecuda: %s\n", "cuEventElapsedTime");
    (void)hStart; (void)hEnd;
    if (pMilliseconds) *pMilliseconds = 0.0f;
    return CUDA_SUCCESS;
}

/* ---- modules / launch ---------------------------------------------------- */

CUresult cuModuleLoad(CUmodule *module, const char *fname) {
    fprintf(stderr,"fakecuda: %s\n", "cuModuleLoad");
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    *module = (CUmodule)0x4;
    return CUDA_SUCCESS;
}
CUresult cuModuleLoadData(CUmodule *module, const void *image) {
    fprintf(stderr,"fakecuda: %s\n", "cuModuleLoadData");
    if (!module) return CUDA_ERROR_INVALID_VALUE;
    *module = (CUmodule)0x4;
    return CUDA_SUCCESS;
}
CUresult cuModuleLoadDataEx(CUmodule *module, const void *image, unsigned int numOptions, CUjit_option *options, void **optionValues) {
    fprintf(stderr,"fakecuda: %s\n", "cuModuleLoadDataEx");
    (void)numOptions; (void)options; (void)optionValues;
    return cuModuleLoadData(module, image);
}
CUresult cuModuleUnload(CUmodule hmod) {
    fprintf(stderr,"fakecuda: %s\n", "cuModuleUnload"); (void)hmod; return CUDA_SUCCESS; }
CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule hmod, const char *name) {
    fprintf(stderr,"fakecuda: %s\n", "cuModuleGetFunction");
    (void)hmod; (void)name;
    if (!hfunc) return CUDA_ERROR_INVALID_VALUE;
    *hfunc = (CUfunction)0x5;
    return CUDA_SUCCESS;
}
CUresult cuLaunchKernel(CUfunction f, unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ, unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ, unsigned int sharedMemBytes, CUstream hStream, void **kernelParams, void **extra) {
    fprintf(stderr,"fakecuda: %s\n", "cuLaunchKernel");
    (void)f; (void)gridDimX; (void)gridDimY; (void)gridDimZ; (void)blockDimX; (void)blockDimY; (void)blockDimZ; (void)sharedMemBytes; (void)hStream; (void)kernelParams; (void)extra;
    return CUDA_SUCCESS;
}

/* ---- pointer attributes -------------------------------------------------- */

CUresult cuPointerGetAttribute(void *data, CUpointer_attribute attribute, CUdeviceptr ptr) {
    fprintf(stderr,"fakecuda: %s\n", "cuPointerGetAttribute");
    (void)attribute; (void)ptr;
    if (data) memset(data, 0, 8);
    return CUDA_SUCCESS;
}

/* ---- occupancy (optional) ------------------------------------------------ */

CUresult cuOccupancyMaxActiveBlocksPerMultiprocessor(int *numBlocks, CUfunction func, int blockSize, size_t dynamicSMemSize) {
    fprintf(stderr,"fakecuda: %s\n", "cuOccupancyMaxActiveBlocksPerMultiprocessor");
    (void)func; (void)blockSize; (void)dynamicSMemSize;
    if (numBlocks) *numBlocks = 1;
    return CUDA_SUCCESS;
}

/* ---- cuGetProcAddress ----------------------------------------------------- */

static void *fc_lookup(const char *symbol) {
    if (!symbol) return NULL;
    /* init */
    if (!strcmp(symbol, "cuInit")) return (void *)cuInit;
    if (!strcmp(symbol, "cudaGetDriverEntryPoint")) return (void *)0;
    if (!strcmp(symbol, "cudaGetDriverEntryPointByVersion")) return (void *)0;
    if (!strcmp(symbol, "cuGetExportTable")) return (void *)cuGetExportTable;
    if (!strcmp(symbol, "cuGetProcAddress")) return (void *)cuGetProcAddress;
    if (!strcmp(symbol, "cuDriverGetVersion")) return (void *)cuDriverGetVersion;
    if (!strcmp(symbol, "cuDeviceGetCount")) return (void *)cuDeviceGetCount;
    if (!strcmp(symbol, "cuDeviceGet")) return (void *)cuDeviceGet;
    if (!strcmp(symbol, "cuDeviceGetAttribute")) return (void *)cuDeviceGetAttribute;
    if (!strcmp(symbol, "cuDeviceGetName")) return (void *)cuDeviceGetName;
    if (!strcmp(symbol, "cuDeviceGetPCIBusId")) return (void *)cuDeviceGetPCIBusId;
    if (!strcmp(symbol, "cuDeviceGetUuid")) return (void *)cuDeviceGetUuid;
    if (!strcmp(symbol, "cuDeviceTotalMem") || !strcmp(symbol, "cuDeviceTotalMem_v2")) return (void *)cuDeviceTotalMem;
    /* context */
    if (!strcmp(symbol, "cuCtxCreate") || !strcmp(symbol, "cuCtxCreate_v2")) return (void *)cuCtxCreate;
    if (!strcmp(symbol, "cuCtxDestroy") || !strcmp(symbol, "cuCtxDestroy_v2")) return (void *)cuCtxDestroy;
    if (!strcmp(symbol, "cuCtxSetCurrent")) return (void *)cuCtxSetCurrent;
    if (!strcmp(symbol, "cuCtxGetCurrent")) return (void *)cuCtxGetCurrent;
    if (!strcmp(symbol, "cuCtxPushCurrent") || !strcmp(symbol, "cuCtxPushCurrent_v2")) return (void *)cuCtxPushCurrent;
    if (!strcmp(symbol, "cuCtxPopCurrent") || !strcmp(symbol, "cuCtxPopCurrent_v2")) return (void *)cuCtxPopCurrent;
    if (!strcmp(symbol, "cuCtxSynchronize")) return (void *)cuCtxSynchronize;
    if (!strcmp(symbol, "cuCtxGetDevice")) return (void *)cuCtxGetDevice;
    if (!strcmp(symbol, "cuCtxGetApiVersion")) return (void *)cuCtxGetApiVersion;
    if (!strcmp(symbol, "cuDevicePrimaryCtxRetain")) return (void *)cuDevicePrimaryCtxRetain;
    if (!strcmp(symbol, "cuDevicePrimaryCtxRelease")) return (void *)cuDevicePrimaryCtxRelease;
    if (!strcmp(symbol, "cuDevicePrimaryCtxSetFlags")) return (void *)cuDevicePrimaryCtxSetFlags;
    if (!strcmp(symbol, "cuDevicePrimaryCtxGetState")) return (void *)cuDevicePrimaryCtxGetState;
    if (!strcmp(symbol, "cuDevicePrimaryCtxReset")) return (void *)cuDevicePrimaryCtxReset;
    /* memory */
    if (!strcmp(symbol, "cuMemAlloc") || !strcmp(symbol, "cuMemAlloc_v2")) return (void *)cuMemAlloc;
    if (!strcmp(symbol, "cuMemFree") || !strcmp(symbol, "cuMemFree_v2")) return (void *)cuMemFree;
    if (!strcmp(symbol, "cuMemAllocHost") || !strcmp(symbol, "cuMemAllocHost_v2")) return (void *)cuMemAllocHost;
    if (!strcmp(symbol, "cuMemFreeHost")) return (void *)cuMemFreeHost;
    if (!strcmp(symbol, "cuMemHostAlloc")) return (void *)cuMemHostAlloc;
    if (!strcmp(symbol, "cuMemHostRegister") || !strcmp(symbol, "cuMemHostRegister_v2")) return (void *)cuMemHostRegister;
    if (!strcmp(symbol, "cuMemHostUnregister")) return (void *)cuMemHostUnregister;
    if (!strcmp(symbol, "cuMemcpyHtoD") || !strcmp(symbol, "cuMemcpyHtoD_v2")) return (void *)cuMemcpyHtoD;
    if (!strcmp(symbol, "cuMemcpyHtoDAsync")) return (void *)cuMemcpyHtoDAsync;
    if (!strcmp(symbol, "cuMemcpyDtoH") || !strcmp(symbol, "cuMemcpyDtoH_v2")) return (void *)cuMemcpyDtoH;
    if (!strcmp(symbol, "cuMemcpyDtoHAsync")) return (void *)cuMemcpyDtoHAsync;
    if (!strcmp(symbol, "cuMemcpyDtoD") || !strcmp(symbol, "cuMemcpyDtoD_v2")) return (void *)cuMemcpyDtoD;
    if (!strcmp(symbol, "cuMemcpyPeer")) return (void *)cuMemcpyPeer;
    if (!strcmp(symbol, "cuMemcpyPeerAsync")) return (void *)cuMemcpyPeerAsync;
    if (!strcmp(symbol, "cuMemsetD8") || !strcmp(symbol, "cuMemsetD8_v2")) return (void *)cuMemsetD8;
    if (!strcmp(symbol, "cuMemsetD32") || !strcmp(symbol, "cuMemsetD32_v2")) return (void *)cuMemsetD32;
    if (!strcmp(symbol, "cuMemGetHandleForAddressRange")) return (void *)cuMemGetHandleForAddressRange;
    /* streams */
    if (!strcmp(symbol, "cuStreamCreate")) return (void *)cuStreamCreate;
    if (!strcmp(symbol, "cuStreamCreateWithPriority")) return (void *)cuStreamCreateWithPriority;
    if (!strcmp(symbol, "cuStreamDestroy") || !strcmp(symbol, "cuStreamDestroy_v2")) return (void *)cuStreamDestroy;
    if (!strcmp(symbol, "cuStreamSynchronize")) return (void *)cuStreamSynchronize;
    if (!strcmp(symbol, "cuStreamQuery")) return (void *)cuStreamQuery;
    if (!strcmp(symbol, "cuStreamWaitEvent")) return (void *)cuStreamWaitEvent;
    /* events */
    if (!strcmp(symbol, "cuEventCreate")) return (void *)cuEventCreate;
    if (!strcmp(symbol, "cuEventDestroy") || !strcmp(symbol, "cuEventDestroy_v2")) return (void *)cuEventDestroy;
    if (!strcmp(symbol, "cuEventRecord")) return (void *)cuEventRecord;
    if (!strcmp(symbol, "cuEventSynchronize")) return (void *)cuEventSynchronize;
    if (!strcmp(symbol, "cuEventElapsedTime")) return (void *)cuEventElapsedTime;
    /* modules */
    if (!strcmp(symbol, "cuModuleLoad")) return (void *)cuModuleLoad;
    if (!strcmp(symbol, "cuModuleLoadData")) return (void *)cuModuleLoadData;
    if (!strcmp(symbol, "cuModuleLoadDataEx")) return (void *)cuModuleLoadDataEx;
    if (!strcmp(symbol, "cuModuleUnload")) return (void *)cuModuleUnload;
    if (!strcmp(symbol, "cuModuleGetFunction")) return (void *)cuModuleGetFunction;
    if (!strcmp(symbol, "cuLaunchKernel")) return (void *)cuLaunchKernel;
    /* pointer */
    if (!strcmp(symbol, "cuPointerGetAttribute")) return (void *)cuPointerGetAttribute;
    if (!strcmp(symbol, "cuOccupancyMaxActiveBlocksPerMultiprocessor")) return (void *)cuOccupancyMaxActiveBlocksPerMultiprocessor;
    return NULL;
}

CUresult cuGetProcAddress(const char *symbol, void **pfn, int cudaVersion, cuuint64_t flags, CUdriverProcAddressQueryResult *result) {
    fprintf(stderr,"fakecuda: %s\n", "cuGetProcAddress");
    (void)cudaVersion; (void)flags;
    if (!pfn) return CUDA_ERROR_INVALID_VALUE;
    void *f = fc_lookup(symbol);
    fprintf(stderr,"fakecuda: getProcAddress %s -> %p\n", symbol, f);
    if (!f) fprintf(stderr,"fakecuda: missing symbol %s\n", symbol);
    *pfn = f;
    if (result) *result = f ? CU_GET_PROC_ADDRESS_SUCCESS : CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND;
    return f ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}





CUresult cudaGetDriverEntryPoint(const char *symbol, void **funcPtr, unsigned long long flags, int *supported) {
    (void)flags;
    if (funcPtr) *funcPtr = fc_lookup(symbol);
    if (supported) *supported = (fc_lookup(symbol) != NULL) ? 1 : 0;
    fprintf(stderr, "fakecuda: cudaGetDriverEntryPoint %s -> %p supported=%d\n", symbol ? symbol : "(null)", funcPtr ? *funcPtr : NULL, supported ? *supported : -1);
    return CUDA_SUCCESS;
}
CUresult cudaGetDriverEntryPointByVersion(const char *symbol, void **funcPtr, unsigned int cudaVersion, unsigned long long flags, int *supported) {
    (void)cudaVersion;
    return cudaGetDriverEntryPoint(symbol, funcPtr, flags, supported);
}

/* ---- interpose dlopen to keep real libcuda out of process --------------- */

static void *real_dlopen(const char *f, int flags) {
    static void *(*p)(const char *, int) = NULL;
    if (!p) p = dlvsym(RTLD_NEXT, "dlopen", "GLIBC_2.2.5");
    return p(f, flags);
}
static void *real_dlsym(void *h, const char *s) {
    static void *(*p)(void *, const char *) = NULL;
    if (!p) p = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    return p(h, s);
}
static int real_dlclose(void *h) {
    static int (*p)(void *) = NULL;
    if (!p) p = dlvsym(RTLD_NEXT, "dlclose", "GLIBC_2.2.5");
    return p(h);
}

void *dlopen(const char *filename, int flags) {
    if (filename && strstr(filename, "libcuda.so")) {
        return (void *)0xdeadbeef;
    }
    return real_dlopen(filename, flags);
}

void *dlsym(void *handle, const char *symbol) {
    if (handle == (void *)0xdeadbeef) {
        void *f = fc_lookup(symbol);
        fprintf(stderr, "fakecuda: dlsym(fakehandle) %s -> %p\n", symbol ? symbol : "(null)", f);
        return f;
    }
    return real_dlsym(handle, symbol);
}

int dlclose(void *handle) {
    if (handle == (void *)0xdeadbeef) return 0;
    return real_dlclose(handle);
}
