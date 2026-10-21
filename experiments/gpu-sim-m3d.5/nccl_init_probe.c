#include <stdio.h>
#include <stdlib.h>
#include <nccl.h>
int main(void) {
    int ver = 0;
    ncclGetVersion(&ver);
    printf("nccl version=%d\n", ver);
    ncclUniqueId id;
    ncclGetUniqueId(&id);
    ncclComm_t comm;
    ncclResult_t res = ncclCommInitRank(&comm, 1, id, 0);
    printf("ncclCommInitRank=%d (%s)\n", res, ncclGetErrorString(res));
    if (res == ncclSuccess) { ncclCommDestroy(comm); printf("SUCCESS\n"); }
    else printf("FAIL\n");
    return (res == ncclSuccess) ? 0 : 1;
}
