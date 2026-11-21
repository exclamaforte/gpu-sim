#include <stdint.h>
#include <stddef.h>
#include <stdio.h>

__attribute__((constructor)) static void _dbg(void){fprintf(stderr,"PLUGIN_LOADED\n");fflush(stderr);}

typedef struct ncclNet_v11_t ncclNet_v11_t;

struct ncclNet_v11_t {
    const char       *name;
    int             (*init)(int);
    int             (*devices)(void);
    int             (*getProperties)(void*, void*);
    int             (*listen)(void*, void*);
    int             (*connect)(void*, void*, void*);
    int             (*accept)(void*, void*, void*);
    int             (*regMr)(void*);
    int             (*regMrDmaBuf)(void*);
    int             (*deregMr)(void*);
    int             (*isend)(int, void*, int, void*);
    int             (*irecv)(int, void*, int, void*);
    int             (*iflush)(int, void*, int, void*);
    int             (*test)(void*);
    int             (*closeSend)(void*);
    int             (*closeRecv)(void*);
    int             (*closeListen)(void*);
    int             (*getDeviceMr)(void*);
    int             (*irecvConsumed)(void*);
    int             (*makeVDevice)(void*);
    int             (*finalize)(void);
    int             (*setNetAttr)(void*, void*);
};
_Static_assert(sizeof(ncclNet_v11_t) == 176, "layout must match DWARF byte_size");

static int stub(void) { return 0; }
#define STUB ((void (*)(void))stub)

/* Route (b): getProperties refuses, so NCCL aborts init deterministically instead of
 * reading an uninitialized ncclNetProperties struct (which SIGSEGV'd). */
static int getProperties_fail(void *dev, void *props) { (void)dev; (void)props; return 1; }

#define MAKE_PLUGIN(VER) \
__attribute__((visibility("default"))) \
ncclNet_v11_t ncclNetPlugin##VER = { \
    .name = "u2o-minimal-net", \
    .init = STUB, .devices = STUB, .getProperties = (int (*)(void*, void*))getProperties_fail, \
    .listen = STUB, .connect = STUB, .accept = STUB, \
    .regMr = STUB, .regMrDmaBuf = STUB, .deregMr = STUB, \
    .isend = STUB, .irecv = STUB, .iflush = STUB, \
    .test = STUB, .closeSend = STUB, .closeRecv = STUB, \
    .closeListen = STUB, .getDeviceMr = STUB, .irecvConsumed = STUB, \
    .makeVDevice = STUB, .finalize = STUB, .setNetAttr = STUB, \
};

MAKE_PLUGIN(_v6) MAKE_PLUGIN(_v7) MAKE_PLUGIN(_v8)
MAKE_PLUGIN(_v9) MAKE_PLUGIN(_v10) MAKE_PLUGIN(_v11)
