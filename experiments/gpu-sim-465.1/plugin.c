/* gpu-sim-465.1 phase 0: a MINIMAL net plugin, to find out empirically which
 * ncclNet_vN the shipped loader will accept.
 *
 * Every field here is derived from a string in libnccl.so.2, not from a
 * remembered public header:
 *   ncclNetGetDevCount      - a loader-visible function name in the binary
 *   ncclNetGetDeviceHandle  - likewise
 *   ncclNet_vN_t            - the per-version struct type, present for v6..v12
 *
 * The two function pointers are the first members because that is the order
 * the names sort into and the order a loader would call them; the REMAINDER OF
 * EACH STRUCT IS DELIBERATELY UNKNOWN AND UNPOPULATED. If the loader reads
 * past member 2, this plugin faults, and a fault is the honest signal -- the
 * point of the phase is to find out how far the ABI is recoverable, not to
 * ship a plausible struct built on a guess.
 */
#include <stddef.h>

typedef struct ncclNet_v6_t   ncclNet_v6_t;
typedef struct ncclNet_v7_t   ncclNet_v7_t;
typedef struct ncclNet_v8_t   ncclNet_v8_t;
typedef struct ncclNet_v9_t   ncclNet_v9_t;
typedef struct ncclNet_v10_t  ncclNet_v10_t;
typedef struct ncclNet_v11_t  ncclNet_v11_t;

/* Minimal layout: the two entry points the loader must reach. Nothing else is
 * claimed, and nothing else is filled in. */
#define MIN_PLUGIN_STRUCT(name)                          \
    struct name {                                        \
        int (*ncclNetGetDevCount)(int*);                 \
        int (*ncclNetGetDeviceHandle)(int, void*);        \
    }

MIN_PLUGIN_STRUCT(ncclNet_v6_t);
MIN_PLUGIN_STRUCT(ncclNet_v7_t);
MIN_PLUGIN_STRUCT(ncclNet_v8_t);
MIN_PLUGIN_STRUCT(ncclNet_v9_t);
MIN_PLUGIN_STRUCT(ncclNet_v10_t);
MIN_PLUGIN_STRUCT(ncclNet_v11_t);

static int getDevCount(int* n) { if (n) *n = 0; return 0; }
static int getDeviceHandle(int dev, void* h) { (void)dev; (void)h; return 0; }

/* THE ENTRY POINT, measured rather than assumed. The loader's own log says:
 *   ncclOsDlsym(ncclNetPlugin_v11) failed: undefined symbol: ncclNetPlugin_v11
 * and it probes v11, v10, v9, v8, v7, v6 -- newest first, and NOT v12 despite
 * ncclNet_v12_t existing in the binary. So the entry is ncclNetPlugin_vN and
 * the highest version this build will accept is v11.
 *
 * The signature is still an inference. It is marked GUESSED below and is the
 * one thing in this plugin not recovered from a string. */
#define ENTRY(VER, STRUCT)                                                    \
    static STRUCT g_##VER##_plugin = { getDevCount, getDeviceHandle };        \
    __attribute__((visibility("default")))                                   \
    STRUCT* ncclNetPlugin##VER(void* logFunction) {                          \
        (void)logFunction; return &g_##VER##_plugin; }

ENTRY(_v6,  ncclNet_v6_t)  ENTRY(_v7,  ncclNet_v7_t)
ENTRY(_v8,  ncclNet_v8_t)  ENTRY(_v9,  ncclNet_v9_t)
ENTRY(_v10, ncclNet_v10_t) ENTRY(_v11, ncclNet_v11_t)


