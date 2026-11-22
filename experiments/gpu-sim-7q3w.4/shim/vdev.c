/* vdev.c -- libcuda.so.1 that presents N virtual B300 devices ALIASING the one real GPU.
 *
 * PI ruling (gpu-sim-7q3w.4, 2026-11-22): identity and topology are modelled, arithmetic
 * is real. So this shim answers the ENUMERATION, IDENTITY, MEMORY-ACCOUNTING and
 * PEER-ACCESS questions with B300 answers, and lets every context, stream, allocation
 * and kernel launch through to the single real device behind it.
 *
 * What it deliberately does NOT model:
 *   - compute capability: forwarded. A virtual sm_100 would make torch build kernels the
 *     real GPU cannot run; "arithmetic is real" is the whole point of aliasing.
 *   - free memory: forwarded. Identity says 288 GiB (cuDeviceTotalMem), but the allocator
 *     sees the real free bytes -- reporting 288 GiB free would over-commit and die with a
 *     real OOM, which is a worse lie than an honest size.
 *
 * Everything not overridden here is a trampoline into the real driver (forwarders.S,
 * generated from the real driver's export list by gen_forwarders.py).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <link.h>
#include <cuda.h>

/* cuda.h maps the legacy names onto their _v2 spellings; the real driver exports BOTH,
 * and so must this shim (a caller may dlsym either), so the macros come back off.
 */
#undef cuDeviceTotalMem
#undef cuMemGetInfo
#undef cuCtxCreate
#undef cuDevicePrimaryCtxRelease
#undef cuDevicePrimaryCtxSetFlags
#undef cuGetProcAddress
#undef cuDeviceGetUuid

#include "b300_spec.h"    /* D1 is the only source of these values */

static unsigned vdev_node(void) {
    const char *e = getenv("VDEV_NODE");
    int n = e ? atoi(e) : 0;
    if (n < 0 || n >= B300_NODES) n = 0;
    return (unsigned)n;
}
#define VDEV_BUS_BASE (B300_PCI_BUS_BASE + vdev_node() * B300_PCI_BUS_NODE_STRIDE)

extern void *fwd_real_handle;
static CUresult (*real_cuInit)(unsigned);
static CUresult (*real_cuDriverGetVersion)(int *);
static CUresult (*real_cuDeviceGetCount)(int *);
static CUresult (*real_cuDeviceGet)(CUdevice *, int);
static CUresult (*real_cuDeviceGetName)(char *, int, CUdevice);
static CUresult (*real_cuDeviceGetAttribute)(int *, CUdevice_attribute, CUdevice);
static CUresult (*real_cuMemGetInfo_v2)(size_t *, size_t *);
static CUresult (*real_cuDevicePrimaryCtxRetain)(CUcontext *, CUdevice);
static CUresult (*real_cuDevicePrimaryCtxRelease_v2)(CUdevice);
static CUresult (*real_cuDevicePrimaryCtxGetState)(CUdevice, unsigned *, int *);
static CUresult (*real_cuDevicePrimaryCtxSetFlags_v2)(CUdevice, unsigned);
static CUresult (*real_cuCtxCreate_v2)(CUcontext *, unsigned, CUdevice);
static CUresult (*real_cuGetProcAddress)(const char *, void **, int, cuuint64_t,
                                         CUdriverProcAddressQueryResult *);
static CUresult (*real_cuGetProcAddress_v2)(const char *, void **, int, cuuint64_t,
                                            CUdriverProcAddressQueryResult *);

/* Own RTLD_LOCAL handle: with RTLD_GLOBAL the real driver's own internal PLT calls
 * resolve to OUR symbols (they are earlier in the global scope), and cuInit calling
 * itself through that loop is a stack overflow, measured once with gdb. */
static void *need(const char *n) {
    static void *h;
    if (!h) {
        const char *p = getenv("VDEV_REAL_CUDA");
        h = dlopen(p ? p : "/usr/lib/libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "vdev: cannot load the real driver: %s\n", dlerror()); abort(); }
    }
    void *p = dlsym(h, n);
    if (!p) fprintf(stderr, "vdev: real driver lacks %s\n", n);
    return p;
}

__attribute__((constructor)) static void vdev_init(void) {
    real_cuInit = need("cuInit");
    real_cuDriverGetVersion = need("cuDriverGetVersion");
    real_cuDeviceGetCount = need("cuDeviceGetCount");
    real_cuDeviceGet = need("cuDeviceGet");
    real_cuDeviceGetName = need("cuDeviceGetName");
    real_cuDeviceGetAttribute = need("cuDeviceGetAttribute");
    real_cuMemGetInfo_v2 = need("cuMemGetInfo_v2");
    real_cuDevicePrimaryCtxRetain = need("cuDevicePrimaryCtxRetain");
    real_cuDevicePrimaryCtxRelease_v2 = need("cuDevicePrimaryCtxRelease_v2");
    real_cuDevicePrimaryCtxGetState = need("cuDevicePrimaryCtxGetState");
    real_cuDevicePrimaryCtxSetFlags_v2 = need("cuDevicePrimaryCtxSetFlags_v2");
    real_cuCtxCreate_v2 = need("cuCtxCreate_v2");
    real_cuGetProcAddress = need("cuGetProcAddress");
    real_cuGetProcAddress_v2 = need("cuGetProcAddress_v2");
}

/* ---------- enumeration and identity ---------- */
CUresult cuInit(unsigned flags) { return real_cuInit ? real_cuInit(flags) : CUDA_ERROR_UNKNOWN; }

CUresult cuDriverGetVersion(int *v) { return real_cuDriverGetVersion(v); }

CUresult cuDeviceGetCount(int *n) { if (!n) return CUDA_ERROR_INVALID_VALUE; *n = B300_NVDEV; return CUDA_SUCCESS; }

CUresult cuDeviceGet(CUdevice *d, int ordinal) {
    if (!d) return CUDA_ERROR_INVALID_VALUE;
    if (ordinal < 0 || ordinal >= B300_NVDEV) return CUDA_ERROR_INVALID_DEVICE;
    *d = ordinal;
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetName(char *name, int len, CUdevice dev) {
    (void)dev;
    if (!name || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    snprintf(name, (size_t)len, "%s", B300_NAME);
    return CUDA_SUCCESS;
}

/* Distinct UUID and PCI bus id per virtual device: without this, NCCL and torch dedupe
 * the "devices" into one, and the illusion is that of a single GPU. */
CUresult cuDeviceGetUuid(CUuuid *uuid, CUdevice dev) {
    /* also exported as _v2 by the real driver; the forwarder covers that spelling */
    if (!uuid) return CUDA_ERROR_INVALID_VALUE;
    memset(uuid, 0, sizeof *uuid);
    memcpy(uuid->bytes, "VDEV-", 5);
    uuid->bytes[5] = (unsigned char)('0' + vdev_node());
    uuid->bytes[6] = (unsigned char)('0' + (dev % 10));
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetPCIBusId(char *bus, int len, CUdevice dev) {
    if (!bus || len <= 0) return CUDA_ERROR_INVALID_VALUE;
    snprintf(bus, (size_t)len, "0000:%02x:00.0",
             (unsigned)(VDEV_BUS_BASE + (unsigned)dev));
    return CUDA_SUCCESS;
}

CUresult cuDeviceTotalMem_v2(size_t *bytes, CUdevice dev) {
    (void)dev;
    if (!bytes) return CUDA_ERROR_INVALID_VALUE;
    *bytes = (size_t)B300_TOTAL_MEM_BYTES;
    return CUDA_SUCCESS;
}
CUresult cuDeviceTotalMem(size_t *bytes, CUdevice dev) { return cuDeviceTotalMem_v2(bytes, dev); }

CUresult cuDeviceGetAttribute(int *value, CUdevice_attribute attr, CUdevice dev) {
    if (!value) return CUDA_ERROR_INVALID_VALUE;
    switch (attr) {
        case CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT: *value = B300_SM_COUNT; return CUDA_SUCCESS;
        default: break;   /* compute capability, memory clocks, everything measured: real */
    }
    CUdevice real_dev = 0;   /* aliasing: every virtual device IS device 0 */
    if (real_cuDeviceGetAttribute) return real_cuDeviceGetAttribute(value, attr, real_dev);
    return CUDA_ERROR_NOT_SUPPORTED;
}

CUresult cuMemGetInfo_v2(size_t *free_b, size_t *total_b) {
    if (!free_b || !total_b) return CUDA_ERROR_INVALID_VALUE;
    size_t f = 0, t = 0;
    CUresult rc = real_cuMemGetInfo_v2(&f, &t);
    if (rc != CUDA_SUCCESS) return rc;
    *free_b = f;            /* real free: the allocator must not over-commit */
    *total_b = (size_t)B300_TOTAL_MEM_BYTES;   /* identity */
    return CUDA_SUCCESS;
}
CUresult cuMemGetInfo(size_t *f, size_t *t) { return cuMemGetInfo_v2(f, t); }

/* ---------- contexts: created on the REAL device, per virtual device ---------- */
#define MAX_CTX B300_NVDEV
static CUcontext ctx_of[MAX_CTX];

CUresult cuDevicePrimaryCtxRetain(CUcontext *pctx, CUdevice dev) {
    if (!pctx) return CUDA_ERROR_INVALID_VALUE;
    if (dev < 0 || dev >= B300_NVDEV) return CUDA_ERROR_INVALID_DEVICE;
    CUresult rc = real_cuDevicePrimaryCtxRetain(pctx, 0);
    if (rc == CUDA_SUCCESS) ctx_of[dev] = *pctx;
    return rc;
}

CUresult cuDevicePrimaryCtxRelease_v2(CUdevice dev) {
    (void)dev;
    return real_cuDevicePrimaryCtxRelease_v2(0);
}
CUresult cuDevicePrimaryCtxRelease(CUdevice dev) { return cuDevicePrimaryCtxRelease_v2(dev); }

CUresult cuDevicePrimaryCtxGetState(CUdevice dev, unsigned *flags, int *active) {
    (void)dev;
    return real_cuDevicePrimaryCtxGetState(0, flags, active);
}

CUresult cuDevicePrimaryCtxSetFlags_v2(CUdevice dev, unsigned flags) {
    (void)dev;
    return real_cuDevicePrimaryCtxSetFlags_v2(0, flags);
}

CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned flags, CUdevice dev) {
    if (dev < 0 || dev >= B300_NVDEV) return CUDA_ERROR_INVALID_DEVICE;
    return real_cuCtxCreate_v2(pctx, flags, 0);   /* alias to the real device */
}
CUresult cuCtxCreate(CUcontext *pctx, unsigned flags, CUdevice dev) { return cuCtxCreate_v2(pctx, flags, dev); }

/* ---------- the resolver torch actually uses ---------- */
static void *override_for(const char *name) {
    struct { const char *n; void *p; } table[] = {
        {"cuInit", (void *)cuInit}, {"cuDriverGetVersion", (void *)cuDriverGetVersion},
        {"cuDeviceGetCount", (void *)cuDeviceGetCount}, {"cuDeviceGet", (void *)cuDeviceGet},
        {"cuDeviceGetName", (void *)cuDeviceGetName}, {"cuDeviceGetUuid", (void *)cuDeviceGetUuid},
        {"cuDeviceGetPCIBusId", (void *)cuDeviceGetPCIBusId},
        {"cuDeviceGetAttribute", (void *)cuDeviceGetAttribute},
        {"cuDeviceTotalMem_v2", (void *)cuDeviceTotalMem_v2},
        {"cuDeviceTotalMem", (void *)cuDeviceTotalMem},
        {"cuMemGetInfo_v2", (void *)cuMemGetInfo_v2}, {"cuMemGetInfo", (void *)cuMemGetInfo},
        {"cuDevicePrimaryCtxRetain", (void *)cuDevicePrimaryCtxRetain},
        {"cuDevicePrimaryCtxRelease_v2", (void *)cuDevicePrimaryCtxRelease_v2},
        {"cuDevicePrimaryCtxRelease", (void *)cuDevicePrimaryCtxRelease},
        {"cuDevicePrimaryCtxGetState", (void *)cuDevicePrimaryCtxGetState},
        {"cuDevicePrimaryCtxSetFlags_v2", (void *)cuDevicePrimaryCtxSetFlags_v2},
        {"cuCtxCreate_v2", (void *)cuCtxCreate_v2}, {"cuCtxCreate", (void *)cuCtxCreate},
    };
    for (size_t i = 0; i < sizeof table / sizeof table[0]; i++)
        if (strcmp(table[i].n, name) == 0) return table[i].p;
    return NULL;
}

CUresult cuGetProcAddress_v2(const char *name, void **fn, int ver, cuuint64_t flags,
                            CUdriverProcAddressQueryResult *status) {
    void *mine = override_for(name);
    if (mine) {
        if (fn) *fn = mine;
        if (status) *status = CU_GET_PROC_ADDRESS_SUCCESS;
        return CUDA_SUCCESS;
    }
    if (real_cuGetProcAddress_v2) return real_cuGetProcAddress_v2(name, fn, ver, flags, status);
    return real_cuGetProcAddress(name, fn, ver, flags, status);
}

/* the header's own spelling: cuGetProcAddress IS the v2 entry point */
CUresult cuGetProcAddress(const char *name, void **fn, int ver, cuuint64_t flags,
                          CUdriverProcAddressQueryResult *status) {
    return cuGetProcAddress_v2(name, fn, ver, flags, status);
}
