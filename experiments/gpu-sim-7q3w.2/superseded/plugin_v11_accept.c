/* gpu-sim-7q3w.2: ncclNet_v11_t built from the PUBLIC nccl_net ABI headers
 * vendored at v2.32.6-1 (nccl_headers/, Apache-2.0, see provenance.md).
 *
 * Step 1 (7q3w.1) refused in getProperties so NCCL could not read a struct it
 * never filled. This step fills it from the real header instead of guessing.
 *
 * Every entry point NCCL reaches before data movement is implemented far enough
 * to be traversed: init -> devices -> getProperties (accept), then
 * listen/connect/accept return a token so the connection setup completes.
 * isend/irecv are out of scope for this step (7q3w.2 ends at accept); they are
 * traced stubs so the log shows exactly where NCCL got to.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nccl_headers/types.h"
#include "nccl_headers/err.h"
#include "nccl_headers/net_device.h"
#include "nccl_headers/common.h"
#include "nccl_headers/net.h"

/* The header is the source of the layout: prove it is the ABI we compile
 * against, and that its vtable size is the one the loader sees (176). */
_Static_assert(sizeof(ncclNet_v11_t) == 176, "ncclNet_v11_t layout drifted");
_Static_assert(sizeof(ncclNetProperties_v11_t) == 112, "props layout drifted");

#define TRACE(fmt, ...) do { fprintf(stderr, "PLUGIN " fmt "\n", ##__VA_ARGS__); fflush(stderr); } while (0)

/* Interposer-defined property values: the test asserts these, so a plugin that
 * accepts and then fills garbage cannot pass. */
#define U2O_SPEED_MBPS   42000
#define U2O_PORT         7
#define U2O_MAX_COMMS    16
#define U2O_MAX_RECVS    4
#define U2O_MAX_MULTI    1
#define U2O_P2P_BYTES    (1ULL << 26)
#define U2O_COLL_BYTES   (1ULL << 22)
#define U2O_DEV_NAME     "u2o-minimal-net"

static int _devices_called;
static int _props_called;
static int _accept_called;

static ncclResult_t p_init(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config,
                           ncclDebugLogger_t logFn, ncclProfilerCallback_t profFn) {
    (void)config; (void)logFn; (void)profFn;
    TRACE("init commId=0x%llx", (unsigned long long)commId);
    *ctx = (void*)(uintptr_t)0x1;
    return ncclSuccess;
}

static ncclResult_t p_devices(int* ndev) {
    TRACE("devices()");
    _devices_called++;
    *ndev = 1;
    return ncclSuccess;
}

static ncclResult_t p_getProperties(int dev, ncclNetProperties_v11_t* props) {
    TRACE("getProperties dev=%d", dev);
    _props_called++;
    if (dev != 0) return ncclInvalidArgument;
    memset(props, 0, sizeof(*props));
    props->name = (char*)U2O_DEV_NAME;
    props->pciPath = (char*)"/sys/devices/pci0000:00/0000:00:01.0";
    props->guid = 0x0b5e2c1d4f6a7081ULL;
    props->ptrSupport = NCCL_PTR_HOST;
    props->regIsGlobal = 0;
    props->forceFlush = 0;
    props->speed = U2O_SPEED_MBPS;
    props->port = U2O_PORT;
    props->latency = 1.0f;
    props->maxComms = U2O_MAX_COMMS;
    props->maxRecvs = U2O_MAX_RECVS;
    props->netDeviceType = NCCL_NET_DEVICE_HOST;
    props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
    props->vProps.ndevs = 0;
    memset(props->vProps.devs, 0, sizeof(props->vProps.devs));
    props->maxP2pBytes = U2O_P2P_BYTES;
    props->maxCollBytes = U2O_COLL_BYTES;
    props->maxMultiRequestSize = U2O_MAX_MULTI;
    return ncclSuccess;
}

static ncclResult_t p_listen(void* ctx, int dev, void* handle, void** listenComm) {
    (void)ctx; (void)dev;
    TRACE("listen dev=%d", dev);
    memset(handle, 0, 64);
    memcpy(handle, "U2OHANDLE", 9);
    *listenComm = (void*)(uintptr_t)0x2;
    return ncclSuccess;
}

static ncclResult_t p_connect(void* ctx, int dev, void* handle, void** sendComm,
                              ncclNetDeviceHandle_v11_t** sendDevComm) {
    (void)ctx; (void)handle;
    TRACE("connect dev=%d", dev);
    *sendComm = (void*)(uintptr_t)0x3;
    if (sendDevComm) *sendDevComm = NULL;
    return ncclSuccess;
}

static ncclResult_t p_accept(void* listenComm, void** recvComm,
                             ncclNetDeviceHandle_v11_t** recvDevComm) {
    (void)listenComm;
    TRACE("accept()");
    _accept_called++;
    *recvComm = (void*)(uintptr_t)0x4;
    if (recvDevComm) *recvDevComm = NULL;
    return ncclSuccess;
}

static ncclResult_t p_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
    TRACE("regMr size=%zu type=%d", size, type);
    *mhandle = (void*)(uintptr_t)0x5;
    return ncclSuccess;
}

static ncclResult_t p_regMrDmaBuf(void* comm, void* data, size_t size, int type,
                                  uint64_t offset, int fd, void** mhandle) {
    TRACE("regMrDmaBuf size=%zu type=%d", size, type);
    *mhandle = (void*)(uintptr_t)0x6;
    return ncclSuccess;
}

static ncclResult_t p_deregMr(void* comm, void* mhandle) {
    TRACE("deregMr");
    return ncclSuccess;
}

static ncclResult_t p_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle,
                            void* phandle, void** request) {
    TRACE("isend size=%zu tag=%d", size, tag);
    *request = (void*)(uintptr_t)0x7;
    return ncclSuccess;
}

static ncclResult_t p_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags,
                            void** mhandles, void** phandles, void** request) {
    TRACE("irecv n=%d", n);
    *request = (void*)(uintptr_t)0x8;
    return ncclSuccess;
}

static ncclResult_t p_iflush(void* recvComm, int n, void** data, int* sizes, void** mhandles,
                             void** request) {
    (void)recvComm; (void)n; (void)data; (void)sizes; (void)mhandles; (void)request;
    return ncclSuccess;
}

static ncclResult_t p_test(void* request, int* done, int* sizes) {
    (void)request; (void)sizes;
    *done = 1;
    return ncclSuccess;
}

static ncclResult_t p_closeSend(void* sendComm) { (void)sendComm; return ncclSuccess; }
static ncclResult_t p_closeRecv(void* recvComm) { (void)recvComm; return ncclSuccess; }
static ncclResult_t p_closeListen(void* listenComm) { (void)listenComm; return ncclSuccess; }

static ncclResult_t p_getDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) {
    (void)comm; (void)mhandle;
    if (dptr_mhandle) *dptr_mhandle = NULL;
    return ncclSuccess;
}

static ncclResult_t p_irecvConsumed(void* recvComm, int n, void* request) {
    (void)recvComm; (void)n; (void)request;
    return ncclSuccess;
}

static ncclResult_t p_makeVDevice(int* d, ncclNetVDeviceProps_v11_t* props) {
    (void)props;
    *d = -1;
    return ncclInvalidUsage;
}

static ncclResult_t p_finalize(void* ctx) {
    TRACE("finalize devices=%d props=%d accept=%d", _devices_called, _props_called, _accept_called);
    return ncclSuccess;
}

static ncclResult_t p_setNetAttr(void* ctx, ncclNetAttr_v11_t* netAttr) {
    (void)ctx; (void)netAttr;
    return ncclSuccess;
}

#define MAKE_PLUGIN(VER) \
__attribute__((visibility("default"))) \
ncclNet_v11_t ncclNetPlugin##VER = { \
    .name = U2O_DEV_NAME, \
    .init = p_init, .devices = p_devices, .getProperties = p_getProperties, \
    .listen = p_listen, .connect = p_connect, .accept = p_accept, \
    .regMr = p_regMr, .regMrDmaBuf = p_regMrDmaBuf, .deregMr = p_deregMr, \
    .isend = p_isend, .irecv = p_irecv, .iflush = p_iflush, \
    .test = p_test, .closeSend = p_closeSend, .closeRecv = p_closeRecv, \
    .closeListen = p_closeListen, .getDeviceMr = p_getDeviceMr, \
    .irecvConsumed = p_irecvConsumed, .makeVDevice = p_makeVDevice, \
    .finalize = p_finalize, .setNetAttr = p_setNetAttr, \
};

MAKE_PLUGIN(_v6) MAKE_PLUGIN(_v7) MAKE_PLUGIN(_v8)
MAKE_PLUGIN(_v9) MAKE_PLUGIN(_v10) MAKE_PLUGIN(_v11)
