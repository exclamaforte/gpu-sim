/* gpu-sim-m3d.4: minimal ncclNet plugin ("sim") exporting the data symbols
 * NCCL's loader actually dlsyms: ncclNetPlugin_v11 and ncclNetPlugin_v12.
 *
 * Built against vendored headers (experiments/gpu-sim-m3d.4/vendor) taken
 * verbatim from NVIDIA/nccl tag v2.32.3-1. Everything returns ncclSuccess and
 * logs its member name to $GPUSIM_TRACE.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "nccl_net.h"

#ifndef WRONG_SYMBOL
#define WRONG_SYMBOL 0
#endif

static void trace(const char *member, const char *extra) {
    const char *p = getenv("GPUSIM_TRACE");
    if (!p || !*p) return;
    FILE *f = fopen(p, "a");
    if (!f) return;
    /* line-buffered append */
    fprintf(f, "%s%s%s\n", member, (extra && *extra) ? " " : "", extra ? extra : "");
    fflush(f);
    fclose(f);
}

static int net_ndevs(void) {
    const char *e = getenv("GPUSIM_NET_DEVICES");
    int n = e ? atoi(e) : 1;
    if (n < 1) n = 1;
    if (n > 8) n = 8;
    return n;
}
static int net_speed(void) { const char *e=getenv("GPUSIM_NET_SPEED"); return e?atoi(e):100000; }
static float net_latency(void){ const char *e=getenv("GPUSIM_NET_LATENCY"); return e?atof(e):1.0f; }

/* ---- v12 implementation ---- */
static void* v12_ctx = NULL;

static ncclResult_t v12_init(void** ctx, uint64_t commId, ncclNetCommConfig_v12_t* config,
                            ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
    (void)config;(void)logFunction;(void)profFunction;
    if (ctx) *ctx = &v12_ctx;
    char buf[128]; snprintf(buf,sizeof(buf),"init commId=%llu", (unsigned long long)commId);
    trace("v12.init", buf);
    return ncclSuccess;
}
static ncclResult_t v12_devices(int* ndev) {
    if (ndev) *ndev = net_ndevs();
    trace("v12.devices", "");
    return ncclSuccess;
}
static ncclResult_t v12_getProperties(int dev, ncclNetProperties_v12_t* props) {
    if (props) {
        memset(props, 0, sizeof(*props));
        props->name = (char*)"sim";
        props->pciPath = (char*)"";
        props->guid = 0;
        props->ptrSupport = NCCL_PTR_HOST;
        props->regIsGlobal = 1;
        props->forceFlush = 0;
        props->speed = net_speed();
        props->port = 0;
        props->latency = net_latency();
        props->maxComms = 1<<20;
        props->maxRecvs = 1;
        props->netDeviceType = NCCL_NET_DEVICE_HOST;
        props->netDeviceVersion = 0;
        props->vProps.ndevs = 1;
        props->vProps.devs[0] = dev;
        props->maxP2pBytes = (size_t)1<<30;
        props->maxCollBytes = (size_t)1<<30;
        props->maxMultiRequestSize = 1;
        props->railId = -1;
        props->planeId = -1;
    }
    char buf[64]; snprintf(buf,sizeof(buf),"getProperties dev=%d", dev);
    trace("v12.getProperties", buf);
    return ncclSuccess;
}
/* generic stubs */
static ncclResult_t v12_listen(void* ctx,int dev,void* handle,void** listenComm){(void)ctx;(void)dev;(void)handle;(void)listenComm; trace("v12.listen",""); return ncclSuccess;}
static ncclResult_t v12_connect(void* ctx,int dev,void* handle,void** sendComm,ncclNetDeviceHandle_v12_t** sendDevComm){(void)ctx;(void)dev;(void)handle;(void)sendComm;(void)sendDevComm; trace("v12.connect",""); return ncclSuccess;}
static ncclResult_t v12_accept(void* listenComm,void** recvComm,ncclNetDeviceHandle_v12_t** recvDevComm){(void)listenComm;(void)recvComm;(void)recvDevComm; trace("v12.accept",""); return ncclSuccess;}
static ncclResult_t v12_regMr(void* comm,void* data,size_t size,int type,void** mhandle){(void)comm;(void)data;(void)size;(void)type; if(mhandle)*mhandle=NULL; trace("v12.regMr",""); return ncclSuccess;}
static ncclResult_t v12_regMrDmaBuf(void* comm,void* data,size_t size,int type,uint64_t offset,int fd,void** mhandle){(void)comm;(void)data;(void)size;(void)type;(void)offset;(void)fd; if(mhandle)*mhandle=NULL; trace("v12.regMrDmaBuf",""); return ncclSuccess;}
static ncclResult_t v12_deregMr(void* comm,void* mhandle){(void)comm;(void)mhandle; trace("v12.deregMr",""); return ncclSuccess;}
static ncclResult_t v12_isend(void* sendComm,void* data,size_t size,int tag,void* mhandle,void* phandle,void** request){(void)sendComm;(void)data;(void)tag;(void)mhandle;(void)phandle; if(request)*request=NULL; char buf[64]; snprintf(buf,sizeof(buf),"isend size=%zu",size); trace("v12.isend",buf); return ncclSuccess;}
static ncclResult_t v12_irecv(void* recvComm,int n,void** data,size_t* sizes,int* tags,void** mhandles,void** phandles,void** request){(void)recvComm;(void)n;(void)data;(void)sizes;(void)tags;(void)mhandles;(void)phandles; if(request)*request=NULL; trace("v12.irecv",""); return ncclSuccess;}
static ncclResult_t v12_iflush(void* recvComm,int n,void** data,int* sizes,void** mhandles,void** request){(void)recvComm;(void)n;(void)data;(void)sizes;(void)mhandles; if(request)*request=NULL; trace("v12.iflush",""); return ncclSuccess;}
static ncclResult_t v12_test(void* request,int* done,int* sizes){(void)request; if(done)*done=1; if(sizes)*sizes=0; trace("v12.test",""); return ncclSuccess;}
static ncclResult_t v12_closeSend(void* sendComm){(void)sendComm; trace("v12.closeSend",""); return ncclSuccess;}
static ncclResult_t v12_closeRecv(void* recvComm){(void)recvComm; trace("v12.closeRecv",""); return ncclSuccess;}
static ncclResult_t v12_closeListen(void* listenComm){(void)listenComm; trace("v12.closeListen",""); return ncclSuccess;}
static ncclResult_t v12_getDeviceMr(void* comm,void* mhandle,void** dptr){(void)comm;(void)mhandle; if(dptr)*dptr=NULL; trace("v12.getDeviceMr",""); return ncclSuccess;}
static ncclResult_t v12_irecvConsumed(void* recvComm,int n,void* request){(void)recvComm;(void)n;(void)request; trace("v12.irecvConsumed",""); return ncclSuccess;}
static ncclResult_t v12_makeVDevice(int* d,ncclNetVDeviceProps_v12_t* props){(void)d;(void)props; trace("v12.makeVDevice",""); return ncclSuccess;}
static ncclResult_t v12_finalize(void* ctx){(void)ctx; trace("v12.finalize",""); return ncclSuccess;}
static ncclResult_t v12_setNetAttr(void* ctx,ncclNetAttr_v12_t* netAttr){(void)ctx;(void)netAttr; trace("v12.setNetAttr",""); return ncclSuccess;}

static ncclNet_v12_t v12_plugin = {
    .name = "sim",
    .init = v12_init,
    .devices = v12_devices,
    .getProperties = v12_getProperties,
    .listen = v12_listen,
    .connect = v12_connect,
    .accept = v12_accept,
    .regMr = v12_regMr,
    .regMrDmaBuf = v12_regMrDmaBuf,
    .deregMr = v12_deregMr,
    .isend = v12_isend,
    .irecv = v12_irecv,
    .iflush = v12_iflush,
    .test = v12_test,
    .closeSend = v12_closeSend,
    .closeRecv = v12_closeRecv,
    .closeListen = v12_closeListen,
    .getDeviceMr = v12_getDeviceMr,
    .irecvConsumed = v12_irecvConsumed,
    .makeVDevice = v12_makeVDevice,
    .finalize = v12_finalize,
    .setNetAttr = v12_setNetAttr,
};

/* ---- v11 implementation ---- */
static void* v11_ctx = NULL;
static ncclResult_t v11_init(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config,
                            ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
    (void)config;(void)logFunction;(void)profFunction;
    if (ctx) *ctx = &v11_ctx;
    char buf[128]; snprintf(buf,sizeof(buf),"init commId=%llu", (unsigned long long)commId);
    trace("v11.init", buf);
    return ncclSuccess;
}
static ncclResult_t v11_devices(int* ndev) {
    if (ndev) *ndev = net_ndevs();
    trace("v11.devices", "");
    return ncclSuccess;
}
static ncclResult_t v11_getProperties(int dev, ncclNetProperties_v11_t* props) {
    if (props) {
        memset(props, 0, sizeof(*props));
        props->name = (char*)"sim";
        props->pciPath = (char*)"";
        props->guid = 0;
        props->ptrSupport = NCCL_PTR_HOST;
        props->regIsGlobal = 1;
        props->forceFlush = 0;
        props->speed = net_speed();
        props->port = 0;
        props->latency = net_latency();
        props->maxComms = 1<<20;
        props->maxRecvs = 1;
        props->netDeviceType = NCCL_NET_DEVICE_HOST;
        props->netDeviceVersion = 0;
        props->vProps.ndevs = 1;
        props->vProps.devs[0] = dev;
        props->maxP2pBytes = (size_t)1<<30;
        props->maxCollBytes = (size_t)1<<30;
        props->maxMultiRequestSize = 1;
    }
    char buf[64]; snprintf(buf,sizeof(buf),"getProperties dev=%d", dev);
    trace("v11.getProperties", buf);
    return ncclSuccess;
}
static ncclResult_t v11_listen(void* ctx,int dev,void* handle,void** listenComm){(void)ctx;(void)dev;(void)handle;(void)listenComm; trace("v11.listen",""); return ncclSuccess;}
static ncclResult_t v11_connect(void* ctx,int dev,void* handle,void** sendComm,ncclNetDeviceHandle_v11_t** sendDevComm){(void)ctx;(void)dev;(void)handle;(void)sendComm;(void)sendDevComm; trace("v11.connect",""); return ncclSuccess;}
static ncclResult_t v11_accept(void* listenComm,void** recvComm,ncclNetDeviceHandle_v11_t** recvDevComm){(void)listenComm;(void)recvComm;(void)recvDevComm; trace("v11.accept",""); return ncclSuccess;}
static ncclResult_t v11_regMr(void* comm,void* data,size_t size,int type,void** mhandle){(void)comm;(void)data;(void)size;(void)type; if(mhandle)*mhandle=NULL; trace("v11.regMr",""); return ncclSuccess;}
static ncclResult_t v11_regMrDmaBuf(void* comm,void* data,size_t size,int type,uint64_t offset,int fd,void** mhandle){(void)comm;(void)data;(void)size;(void)type;(void)offset;(void)fd; if(mhandle)*mhandle=NULL; trace("v11.regMrDmaBuf",""); return ncclSuccess;}
static ncclResult_t v11_deregMr(void* comm,void* mhandle){(void)comm;(void)mhandle; trace("v11.deregMr",""); return ncclSuccess;}
static ncclResult_t v11_isend(void* sendComm,void* data,size_t size,int tag,void* mhandle,void* phandle,void** request){(void)sendComm;(void)data;(void)tag;(void)mhandle;(void)phandle; if(request)*request=NULL; char buf[64]; snprintf(buf,sizeof(buf),"isend size=%zu",size); trace("v11.isend",buf); return ncclSuccess;}
static ncclResult_t v11_irecv(void* recvComm,int n,void** data,size_t* sizes,int* tags,void** mhandles,void** phandles,void** request){(void)recvComm;(void)n;(void)data;(void)sizes;(void)tags;(void)mhandles;(void)phandles; if(request)*request=NULL; trace("v11.irecv",""); return ncclSuccess;}
static ncclResult_t v11_iflush(void* recvComm,int n,void** data,int* sizes,void** mhandles,void** request){(void)recvComm;(void)n;(void)data;(void)sizes;(void)mhandles; if(request)*request=NULL; trace("v11.iflush",""); return ncclSuccess;}
static ncclResult_t v11_test(void* request,int* done,int* sizes){(void)request; if(done)*done=1; if(sizes)*sizes=0; trace("v11.test",""); return ncclSuccess;}
static ncclResult_t v11_closeSend(void* sendComm){(void)sendComm; trace("v11.closeSend",""); return ncclSuccess;}
static ncclResult_t v11_closeRecv(void* recvComm){(void)recvComm; trace("v11.closeRecv",""); return ncclSuccess;}
static ncclResult_t v11_closeListen(void* listenComm){(void)listenComm; trace("v11.closeListen",""); return ncclSuccess;}
static ncclResult_t v11_getDeviceMr(void* comm,void* mhandle,void** dptr){(void)comm;(void)mhandle; if(dptr)*dptr=NULL; trace("v11.getDeviceMr",""); return ncclSuccess;}
static ncclResult_t v11_irecvConsumed(void* recvComm,int n,void* request){(void)recvComm;(void)n;(void)request; trace("v11.irecvConsumed",""); return ncclSuccess;}
static ncclResult_t v11_makeVDevice(int* d,ncclNetVDeviceProps_v11_t* props){(void)d;(void)props; trace("v11.makeVDevice",""); return ncclSuccess;}
static ncclResult_t v11_finalize(void* ctx){(void)ctx; trace("v11.finalize",""); return ncclSuccess;}
static ncclResult_t v11_setNetAttr(void* ctx,ncclNetAttr_v11_t* netAttr){(void)ctx;(void)netAttr; trace("v11.setNetAttr",""); return ncclSuccess;}

static ncclNet_v11_t v11_plugin = {
    .name = "sim",
    .init = v11_init,
    .devices = v11_devices,
    .getProperties = v11_getProperties,
    .listen = v11_listen,
    .connect = v11_connect,
    .accept = v11_accept,
    .regMr = v11_regMr,
    .regMrDmaBuf = v11_regMrDmaBuf,
    .deregMr = v11_deregMr,
    .isend = v11_isend,
    .irecv = v11_irecv,
    .iflush = v11_iflush,
    .test = v11_test,
    .closeSend = v11_closeSend,
    .closeRecv = v11_closeRecv,
    .closeListen = v11_closeListen,
    .getDeviceMr = v11_getDeviceMr,
    .irecvConsumed = v11_irecvConsumed,
    .makeVDevice = v11_makeVDevice,
    .finalize = v11_finalize,
    .setNetAttr = v11_setNetAttr,
};

/* ---- exported symbols ---- */
#ifdef __cplusplus
extern "C" {
#endif
#if WRONG_SYMBOL
/* old wrong name (8g6 signature): ncclNet_vN -- loader dlsym for ncclNetPlugin_vN returns NULL */
__attribute__((visibility("default"))) ncclNet_v12_t ncclNet_v12 = { .name = "sim-wrong" };
#else
__attribute__((visibility("default"))) ncclNet_v12_t ncclNetPlugin_v12 = v12_plugin;
__attribute__((visibility("default"))) ncclNet_v11_t ncclNetPlugin_v11 = v11_plugin;
#endif
#ifdef __cplusplus
}
#endif
