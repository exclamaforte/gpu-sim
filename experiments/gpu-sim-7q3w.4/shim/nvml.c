/* nvml.c -- libnvidia-ml.so.1 presenting B300 identity for the 7q3w.4 virtual node.
 *
 * WHY THIS EXISTS (PI measurement, 2026-11-22 17:43Z): torch's pre-init
 * device_count() is decided by NVML, not by libcuda and not by cudart:
 *     torch.cuda._device_count_nvml()  -> 1     <- the decider
 *     torch._C._cuda_getDeviceCount()  -> 8     <- dynamic interposition binds
 *     torch.cuda.device_count()        -> 1     <- NVML wins
 * So the count/identity layer has to answer NVML.  This shim is deliberately
 * shared-shape with the 7q3w.3 NVML work: one implementation, not two.
 *
 * WHAT IS MODELLED: count, name, UUID, PCI bus id, HBM capacity, PCIe identity.
 * WHAT IS FORWARDED: free memory (the allocator must see real free bytes, not a
 * 288 GiB lie), compute capability (a virtual sm_103 would build kernels the real
 * GPU cannot run), driver version, and every unmodified nvml* entry point via the
 * generated trampolines.
 *
 * The real library is loaded by nvml_forwarders.c's constructor with an absolute
 * path and RTLD_LOCAL; this file only dlsyms a handful of names from that handle.
 */
#define _GNU_SOURCE
#define NVML_NO_UNVERSIONED_FUNC_DEFS 1
#include "b300_spec.h"
#include <nvml.h>
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

extern void *nvml_fwd_real_handle;   /* set by generated nvml_forwarders.c */

static void *real_sym(const char *name) {
    return nvml_fwd_real_handle ? dlsym(nvml_fwd_real_handle, name) : NULL;
}

/* A fake handle is a small non-NULL integer; the real library is only ever
 * asked about real device 0 (the one GPU behind all virtual devices). */
static nvmlDevice_t fake_handle(unsigned idx) {
    return (nvmlDevice_t)(uintptr_t)(idx + 1u);
}
static int fake_index(nvmlDevice_t dev) {
    uintptr_t v = (uintptr_t)dev;
    if (v == 0 || v > B300_NVDEV) return -1;
    return (int)(v - 1u);
}
static nvmlDevice_t real_dev0(void) {
    static nvmlDevice_t h = NULL;
    if (!h) {
        typedef nvmlReturn_t (*F)(unsigned, nvmlDevice_t *);
        F f = (F)real_sym("nvmlDeviceGetHandleByIndex_v2");
        if (f) (void)f(0, &h);
    }
    return h;
}

/* ---------- library lifecycle ---------- */
nvmlReturn_t nvmlInit_v2(void) {
    typedef nvmlReturn_t (*F)(void);
    F f = (F)real_sym("nvmlInit_v2");
    if (f) (void)f();
    return NVML_SUCCESS;   /* the illusion exists even if the real NVML is unhappy */
}
nvmlReturn_t nvmlInit(void) { return nvmlInit_v2(); }

nvmlReturn_t nvmlShutdown(void) {
    typedef nvmlReturn_t (*F)(void);
    F f = (F)real_sym("nvmlShutdown");
    if (f) (void)f();
    return NVML_SUCCESS;
}

const char *nvmlErrorString(nvmlReturn_t result) {
    typedef const char *(*F)(nvmlReturn_t);
    F f = (F)real_sym("nvmlErrorString");
    return f ? f(result) : "Unknown Error";
}

nvmlReturn_t nvmlSystemGetDriverVersion(char *version, unsigned int length) {
    typedef nvmlReturn_t (*F)(char *, unsigned int);
    F f = (F)real_sym("nvmlSystemGetDriverVersion");
    return f ? f(version, length) : NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlSystemGetConfComputeSettings(nvmlSystemConfComputeSettings_t *settings) {
    typedef nvmlReturn_t (*F)(nvmlSystemConfComputeSettings_t *);
    F f = (F)real_sym("nvmlSystemGetConfComputeSettings");
    return f ? f(settings) : NVML_ERROR_NOT_SUPPORTED;
}

nvmlReturn_t nvmlSystemGetConfComputeState(nvmlConfComputeSystemState_t *state) {
    typedef nvmlReturn_t (*F)(nvmlConfComputeSystemState_t *);
    F f = (F)real_sym("nvmlSystemGetConfComputeState");
    return f ? f(state) : NVML_ERROR_NOT_SUPPORTED;
}

/* ---------- count and handles: B300_NVDEV virtual devices ---------- */
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned int *count) {
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    *count = B300_NVDEV;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetCount(unsigned int *count) { return nvmlDeviceGetCount_v2(count); }

nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned int index, nvmlDevice_t *device) {
    if (!device || index >= B300_NVDEV) return NVML_ERROR_INVALID_ARGUMENT;
    *device = fake_handle(index);
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned int index, nvmlDevice_t *device) {
    return nvmlDeviceGetHandleByIndex_v2(index, device);
}

nvmlReturn_t nvmlDeviceGetHandleByPciBusId_v2(const char *pciBusId, nvmlDevice_t *device) {
    unsigned domain = 0, bus = 0, dev = 0, fn = 0;
    if (!pciBusId || !device) return NVML_ERROR_INVALID_ARGUMENT;
    if (sscanf(pciBusId, "%x:%x:%x.%x", &domain, &bus, &dev, &fn) != 4)
        return NVML_ERROR_INVALID_ARGUMENT;
    (void)domain; (void)dev; (void)fn;
    if (bus < B300_PCI_BUS_BASE || bus >= B300_PCI_BUS_BASE + B300_NVDEV)
        return NVML_ERROR_NOT_FOUND;
    *device = fake_handle(bus - B300_PCI_BUS_BASE);
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetHandleByPciBusId(const char *pciBusId, nvmlDevice_t *device) {
    return nvmlDeviceGetHandleByPciBusId_v2(pciBusId, device);
}

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t device, unsigned int *index) {
    int idx = fake_index(device);
    if (idx < 0 || !index) return NVML_ERROR_INVALID_ARGUMENT;
    *index = (unsigned int)idx;
    return NVML_SUCCESS;
}

/* ---------- identity ---------- */
nvmlReturn_t nvmlDeviceGetName(nvmlDevice_t device, char *name, unsigned int length) {
    if (!name || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    if (fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(name, length, "%s", B300_NAME);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetUUID(nvmlDevice_t device, char *uuid, unsigned int length) {
    int idx = fake_index(device);
    if (idx < 0 || !uuid || length == 0) return NVML_ERROR_INVALID_ARGUMENT;
    snprintf(uuid, length, "GPU-VDEV-B300-%04d", idx);
    return NVML_SUCCESS;
}

static void fill_pci(nvmlDevice_t device, unsigned int *domain, unsigned int *bus,
                     unsigned int *dev, char *bus_id, size_t bus_id_len,
                     char *legacy, size_t legacy_len) {
    int idx = fake_index(device);
    unsigned b = B300_PCI_BUS_BASE + (idx < 0 ? 0 : (unsigned)idx);
    *domain = 0;
    *bus = b;
    *dev = 0;
    if (bus_id) snprintf(bus_id, bus_id_len, "%08X:%02X:%02X.0", 0u, b, 0u);
    if (legacy) snprintf(legacy, legacy_len, "%04X:%02X:%02X.0", 0u, b, 0u);
}

nvmlReturn_t nvmlDeviceGetPciInfo_v3(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    if (!pci || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    memset(pci, 0, sizeof *pci);
    fill_pci(device, &pci->domain, &pci->bus, &pci->device,
             pci->busId, sizeof pci->busId,
             pci->busIdLegacy, sizeof pci->busIdLegacy);
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetPciInfo_v2(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    return nvmlDeviceGetPciInfo_v3(device, pci);
}
nvmlReturn_t nvmlDeviceGetPciInfo(nvmlDevice_t device, nvmlPciInfo_t *pci) {
    return nvmlDeviceGetPciInfo_v3(device, pci);
}

nvmlReturn_t nvmlDeviceGetPciInfoExt(nvmlDevice_t device, nvmlPciInfoExt_t *pci) {
    if (!pci || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    memset(pci, 0, sizeof *pci);
    fill_pci(device, &pci->domain, &pci->bus, &pci->device,
             pci->busId, sizeof pci->busId, NULL, 0);
    pci->baseClass = 0x03;   /* display controller */
    pci->subClass = 0x00;
    return NVML_SUCCESS;
}

/* ---------- memory: B300 total identity, REAL free bytes ---------- */
nvmlReturn_t nvmlDeviceGetMemoryInfo(nvmlDevice_t device, nvmlMemory_t *memory) {
    if (!memory || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    typedef nvmlReturn_t (*F)(nvmlDevice_t, nvmlMemory_t *);
    F f = (F)real_sym("nvmlDeviceGetMemoryInfo");
    nvmlDevice_t real = real_dev0();
    if (f && real && f(real, memory) == NVML_SUCCESS) {
        memory->total = (unsigned long long)B300_TOTAL_MEM_BYTES;
        return NVML_SUCCESS;
    }
    memory->total = (unsigned long long)B300_TOTAL_MEM_BYTES;
    memory->free = (unsigned long long)B300_TOTAL_MEM_BYTES;
    memory->used = 0;
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceGetMemoryInfo_v2(nvmlDevice_t device, nvmlMemory_v2_t *memory) {
    if (!memory || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    typedef nvmlReturn_t (*F)(nvmlDevice_t, nvmlMemory_v2_t *);
    F f = (F)real_sym("nvmlDeviceGetMemoryInfo_v2");
    nvmlDevice_t real = real_dev0();
    if (f && real && f(real, memory) == NVML_SUCCESS) {
        memory->total = (unsigned long long)B300_TOTAL_MEM_BYTES;
        return NVML_SUCCESS;
    }
    if (memory->version == 0) memory->version = nvmlMemory_v2;
    memory->total = (unsigned long long)B300_TOTAL_MEM_BYTES;
    memory->reserved = 0;
    memory->free = (unsigned long long)B300_TOTAL_MEM_BYTES;
    memory->used = 0;
    return NVML_SUCCESS;
}

/* Capability is FORWARDED (A1): the real GPU is 12.0 even though D1 says the
 * B300 identity is 10.3.  A virtual 10.3 would make torch build kernels this GPU
 * cannot execute; the modelled identity is name/capacity/topology, not the ISA. */
nvmlReturn_t nvmlDeviceGetCudaComputeCapability(nvmlDevice_t device, int *major, int *minor) {
    if (!major || !minor || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    typedef nvmlReturn_t (*F)(nvmlDevice_t, int *, int *);
    F f = (F)real_sym("nvmlDeviceGetCudaComputeCapability");
    nvmlDevice_t real = real_dev0();
    if (!f || !real) return NVML_ERROR_NOT_SUPPORTED;
    return f(real, major, minor);
}

/* ---------- PCIe identity from D1 ---------- */
nvmlReturn_t nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice_t device, unsigned int *gen) {
    if (!gen || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    *gen = B300_PCIE_GEN;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice_t device, unsigned int *width) {
    if (!width || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    *width = B300_PCIE_WIDTH;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetPcieLinkMaxSpeed(nvmlDevice_t device, unsigned int *maxSpeed) {
    if (!maxSpeed || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    *maxSpeed = NVML_PCIE_LINK_MAX_SPEED_64000MBPS;
    return NVML_SUCCESS;
}

/* ---------- surfaces owned by later 7q3w.4/7q3w.3 increments ---------- */
nvmlReturn_t nvmlDeviceGetPlatformInfo(nvmlDevice_t device, nvmlPlatformInfo_t *info) {
    (void)device; (void)info;
    return NVML_ERROR_NOT_SUPPORTED;   /* D2 node topology */
}
nvmlReturn_t nvmlDeviceGetFieldValues(nvmlDevice_t device, int count, nvmlFieldValue_t *values) {
    (void)device; (void)count; (void)values;
    return NVML_ERROR_NOT_SUPPORTED;
}
nvmlReturn_t nvmlDeviceGetMemoryErrorCounter(nvmlDevice_t device,
                                             nvmlMemoryErrorType_t errorType,
                                             nvmlEccCounterType_t counterType,
                                             nvmlMemoryLocation_t locationType,
                                             unsigned long long *count) {
    (void)device; (void)errorType; (void)counterType; (void)locationType;
    if (!count || fake_index(device) < 0) return NVML_ERROR_INVALID_ARGUMENT;
    *count = 0;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetGpuFabricInfoV(nvmlDevice_t device,
                                         nvmlGpuFabricInfoV_t *info) {
    (void)device; (void)info;
    return NVML_ERROR_NOT_SUPPORTED;   /* no modelled fabric in this increment */
}
nvmlReturn_t nvmlDeviceGetNvLinkState(nvmlDevice_t device, unsigned int link,
                                      nvmlEnableState_t *isActive) {
    (void)device; (void)link; (void)isActive;
    return NVML_ERROR_NOT_SUPPORTED;   /* D2 owns the NVLink view */
}
nvmlReturn_t nvmlDeviceGetNvLinkCapability(nvmlDevice_t device, unsigned int link,
                                           nvmlNvLinkCapability_t capability,
                                           unsigned int *capResult) {
    (void)device; (void)link; (void)capability; (void)capResult;
    return NVML_ERROR_NOT_SUPPORTED;
}
nvmlReturn_t nvmlDeviceGetNvLinkRemoteDeviceType(nvmlDevice_t device, unsigned int link,
                                                 nvmlIntNvLinkDeviceType_t *type) {
    (void)device; (void)link; (void)type;
    return NVML_ERROR_NOT_SUPPORTED;
}
nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo_v2(nvmlDevice_t device, unsigned int link,
                                                 nvmlPciInfo_t *pci) {
    (void)device; (void)link; (void)pci;
    return NVML_ERROR_NOT_SUPPORTED;
}
nvmlReturn_t nvmlDeviceGetNvLinkRemotePciInfo(nvmlDevice_t device, unsigned int link,
                                              nvmlPciInfo_t *pci) {
    return nvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci);
}
