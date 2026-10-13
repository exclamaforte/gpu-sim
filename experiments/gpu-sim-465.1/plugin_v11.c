/* gpu-sim-465.1: the plugin, built on the RECOVERED layout.
 *
 * Every field below comes from DWARF in the shipped library, read off the
 * typedef ncclNet_v11_t --DW_AT_type--> <0xf4b994>, byte_size 176, 22 members.
 * This is a RECOVERY, not a recollection of the public API, and the difference
 * is visible: the earlier two-member struct was 16 bytes and the loader read
 * 176, which is exactly the SIGSEGV recorded in probe_evidence.log.
 *
 * Every member is a non-NULL stub. The loader may CALL any of them, and a NULL
 * in a 176-byte vtable is indistinguishable from a crash -- so the point of a
 * minimal plugin is that nothing is null, not that nothing is filled in.
 *
 * The SIGNATURES are still unknown: DWARF gives offsets and names, not
 * prototypes. The stubs therefore take (void) and are declared through
 * generic function pointers. If the loader's calling convention differs this
 * will fault again, and that fault would be information about which member it
 * reached -- which is the one remaining way to recover them.
 */
#include <stdint.h>
#include <stddef.h>

typedef struct ncclNet_v11_t ncclNet_v11_t;

/* 22 members, byte_size 176, all 8 bytes. Offsets from DW_AT_data_member_location. */
struct ncclNet_v11_t {
    const char       *name;            /* +0   */
    int             (*init)(int);      /* +8   */
    int             (*devices)(void);  /* +16  */
    int             (*getProperties)(void*, void*); /* +24 */
    int             (*listen)(void*, void*);       /* +32 */
    int             (*connect)(void*, void*, void*);/* +40 */
    int             (*accept)(void*, void*, void*); /* +48 */
    int             (*regMr)(void*);   /* +56  */
    int             (*regMrDmaBuf)(void*); /* +64 */
    int             (*deregMr)(void*); /* +72  */
    int             (*isend)(int, void*, int, void*);  /* +80 */
    int             (*irecv)(int, void*, int, void*);  /* +88 */
    int             (*iflush)(int, void*, int, void*); /* +96 */
    int             (*test)(void*);     /* +104 */
    int             (*closeSend)(void*);/* +112 */
    int             (*closeRecv)(void*);/* +120 */
    int             (*closeListen)(void*); /* +128 */
    int             (*getDeviceMr)(void*); /* +136 */
    int             (*irecvConsumed)(void*); /* +144 */
    int             (*makeVDevice)(void*); /* +152 */
    int             (*finalize)(void);  /* +160 */
    int             (*setNetAttr)(void*, void*); /* +168 */
};
_Static_assert(sizeof(ncclNet_v11_t) == 176, "layout must match DWARF byte_size");

/* One stub body for every member: the ABI is not known well enough to give
 * them distinct signatures, and pretending otherwise is the error the bead
 * forbids. A member that is CALLED will therefore misbehave -- visibly, rather
 * than silently returning a plausible number. */
static int stub(void) { return 0; }
#define STUB ((void (*)(void))stub)

static ncclNet_v11_t g_plugin = {
    .name            = "u2o-minimal-net",
    .init            = STUB, .devices = STUB, .getProperties = STUB,
    .listen          = STUB, .connect = STUB, .accept = STUB,
    .regMr           = STUB, .regMrDmaBuf = STUB, .deregMr = STUB,
    .isend           = STUB, .irecv = STUB, .iflush = STUB,
    .test            = STUB, .closeSend = STUB, .closeRecv = STUB,
    .closeListen     = STUB, .getDeviceMr = STUB, .irecvConsumed = STUB,
    .makeVDevice     = STUB, .finalize = STUB, .setNetAttr = STUB,
};

/* v11 is the only version the loader reaches first, but exporting the rest
 * costs nothing and keeps the fallback path available if v11 is rejected. */
#define ENTRY(VER)                                                       \
    __attribute__((visibility("default")))                              \
    ncclNet_v11_t* ncclNetPlugin##VER(void* logFunction) {                \
        (void)logFunction; return &g_plugin; }

ENTRY(_v6) ENTRY(_v7) ENTRY(_v8) ENTRY(_v9) ENTRY(_v10) ENTRY(_v11)
