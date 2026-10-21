#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "nccl_net.h"
extern ncclNet_v12_t ncclNetPlugin_v12;

#define CAP (64*1024*1024)

int main(void) {
    /* init plugin */
    void *ctx = NULL;
    ncclResult_t r = ncclNetPlugin_v12.init(&ctx, 1, NULL, NULL, NULL);
    if (r != ncclSuccess) { printf("init failed %d\n", r); return 1; }
    int ndev = 0;
    ncclNetPlugin_v12.devices(&ndev);
    ncclNetProperties_v12_t props;
    ncclNetPlugin_v12.getProperties(0, &props);
    void *handle = calloc(1, 64);
    void *listenComm = NULL;
    ncclNetPlugin_v12.listen(ctx, 0, handle, &listenComm);
    void *sendComm = NULL;
    ncclNetDeviceHandle_v12_t *sendDev = NULL;
    ncclNetPlugin_v12.connect(ctx, 0, handle, &sendComm, &sendDev);
    void *mhandle = NULL;
    ncclNetPlugin_v12.regMr(sendComm, malloc(1024), 1024, 0, &mhandle);
    void *req = NULL;
    char *buf = (char*)malloc(CAP+1);
    r = ncclNetPlugin_v12.isend(sendComm, buf, CAP+1, 0, mhandle, NULL, &req);
    printf("oversized isend returned %d (expect %d)\n", r, ncclInvalidArgument);
    /* cleanup */
    ncclNetPlugin_v12.closeSend(sendComm);
    ncclNetPlugin_v12.closeListen(listenComm);
    ncclNetPlugin_v12.finalize(ctx);
    return (r == ncclInvalidArgument) ? 0 : 1;
}