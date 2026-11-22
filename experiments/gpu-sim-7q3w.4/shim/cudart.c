/* cudart.c -- libcudart interposer for the 7q3w.4 virtual B300 node.
 *
 * WHY THIS EXISTS (PI measurement, 2026-11-22 17:43Z): torch is NOT statically
 * linked against cudart.  libtorch_cuda.so imports cudaGetDeviceCount and has
 * NEEDED libcudart.so.13; a three-line interposer on cudaGetDeviceCount binds.
 * With the NVML shim returning 8, device_count() is 8, but torch.cuda.init()
 * then fails 'invalid device ordinal' because cudart still sees one real device.
 * This shim makes cudart's count/properties/set/get agree with the NVML view.
 *
 * WHAT IS MODELLED: count, name, HBM capacity, PCI bus id, SM count.
 * WHAT IS FORWARDED: compute capability (the real GPU is 12.0; a virtual 10.3
 * would build kernels this GPU cannot run), free memory (the allocator must see
 * real free bytes), and every function not overridden (the real libcudart is
 * still the NEEDED library; this shim only preempts the named entry points).
 *
 * Install with LD_PRELOAD (not as libcudart.so.13): the real runtime must still
 * load, and the overrides call its functions directly.
 */
#define _GNU_SOURCE
#include "b300_spec.h"
#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void *real_cudart(void) {
    static void *h = NULL;
    if (!h) {
        h = dlopen("libcudart.so.13", RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
        if (!h) h = dlopen("libcudart.so.13", RTLD_NOW | RTLD_LOCAL);
    }
    return h;
}
#define REAL(name) dlsym(real_cudart(), name)

static __thread int vdev_current = 0;

static unsigned vdev_node(void) {
    const char *e = getenv("VDEV_NODE");
    int n = e ? atoi(e) : 0;
    if (n < 0 || n >= B300_NODES) n = 0;
    return (unsigned)n;
}
#define VDEV_BUS_BASE (B300_PCI_BUS_BASE + vdev_node() * B300_PCI_BUS_NODE_STRIDE)

cudaError_t cudaGetDeviceCount(int *count) {
    if (!count) return cudaErrorInvalidValue;
    *count = B300_NVDEV;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int device) {
    if (device < 0 || device >= B300_NVDEV) return cudaErrorInvalidDevice;
    typedef cudaError_t (*F)(int);
    F f = (F)REAL("cudaSetDevice");
    if (!f) return cudaErrorUnknown;
    cudaError_t rc = f(0);   /* every virtual device aliases the one real GPU */
    if (rc == cudaSuccess) vdev_current = device;
    return rc;
}

cudaError_t cudaGetDevice(int *device) {
    if (!device) return cudaErrorInvalidValue;
    *device = vdev_current;
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp *prop, int device) {
    if (!prop) return cudaErrorInvalidValue;
    if (device < 0 || device >= B300_NVDEV) return cudaErrorInvalidDevice;
    typedef cudaError_t (*F)(struct cudaDeviceProp *, int);
    F f = (F)REAL("cudaGetDeviceProperties");
    if (!f) return cudaErrorUnknown;
    cudaError_t rc = f(prop, 0);   /* real properties first: capability, limits */
    if (rc != cudaSuccess) return rc;
    snprintf(prop->name, sizeof prop->name, "%s", B300_NAME);
    prop->totalGlobalMem = (size_t)B300_TOTAL_MEM_BYTES;
    prop->multiProcessorCount = B300_SM_COUNT;
    prop->pciDomainID = 0;
    prop->pciBusID = (int)(VDEV_BUS_BASE + (unsigned)device);
    prop->pciDeviceID = 0;
    /* distinct UUID per virtual device, as the driver shim does, so consumers
     * that dedupe by UUID do not collapse the eight devices into one. */
    memset(&prop->uuid, 0, sizeof prop->uuid);
    memcpy(prop->uuid.bytes, "VDEV-", 5);
    prop->uuid.bytes[5] = (unsigned char)('0' + vdev_node());
    prop->uuid.bytes[6] = (unsigned char)('0' + (device % 10));
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, enum cudaDeviceAttr attr, int device) {
    if (!value) return cudaErrorInvalidValue;
    if (device < 0 || device >= B300_NVDEV) return cudaErrorInvalidDevice;
    switch (attr) {
        case cudaDevAttrMultiProcessorCount: *value = B300_SM_COUNT; return cudaSuccess;
        case cudaDevAttrPciDomainId: *value = 0; return cudaSuccess;
        case cudaDevAttrPciBusId: *value = (int)(VDEV_BUS_BASE + (unsigned)device); return cudaSuccess;
        case cudaDevAttrPciDeviceId: *value = 0; return cudaSuccess;
        default: break;
    }
    typedef cudaError_t (*F)(int *, enum cudaDeviceAttr, int);
    F f = (F)REAL("cudaDeviceGetAttribute");
    return f ? f(value, attr, 0) : cudaErrorUnknown;
}

cudaError_t cudaDeviceGetPCIBusId(char *pciBusId, int len, int device) {
    if (!pciBusId || len <= 0) return cudaErrorInvalidValue;
    if (device < 0 || device >= B300_NVDEV) return cudaErrorInvalidDevice;
    snprintf(pciBusId, (size_t)len, "%04X:%02X:%02X.%X",
             0u, (unsigned)(VDEV_BUS_BASE + (unsigned)device), 0u, 0u);
    return cudaSuccess;
}

cudaError_t cudaMemGetInfo(size_t *freeMem, size_t *totalMem) {
    if (!freeMem || !totalMem) return cudaErrorInvalidValue;
    typedef cudaError_t (*F)(size_t *, size_t *);
    F f = (F)REAL("cudaMemGetInfo");
    if (!f) return cudaErrorUnknown;
    size_t f_real = 0, t_real = 0;
    cudaError_t rc = f(&f_real, &t_real);
    if (rc != cudaSuccess) return rc;
    *freeMem = f_real;                     /* real free: do not over-commit */
    *totalMem = (size_t)B300_TOTAL_MEM_BYTES;   /* B300 identity */
    return cudaSuccess;
}
