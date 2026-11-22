/* gpu-sim-7q3w.2: measure the ABI from TWO headers and let run.sh diff them.
 *
 * The plugin compiles against the PUBLIC plugin header
 * (plugins/net/example/nccl/net.h in the NCCL tree, Apache-2.0, tag recorded in
 * vendor/nccl/nccl.tag).  NCCL itself was compiled against the LOADER header
 * (src/include/plugin/net/net_v11.h) -- and the loader converts our v11 struct
 * into its internal ncclNetProperties_t FIELD BY FIELD (src/plugin/net/
 * net_v11.cc:14-40).  So a wrong offset does not crash: it silently delivers
 * someone else's value.  That is why this probe prints every offset, and why
 * run.sh refuses to build the plugin unless the two outputs are identical.
 *
 * Compiled twice:
 *   gcc -DABI_HEADER_LOADER  -I vendor/nccl      abi_measure.c
 *   gcc -DABI_HEADER_PLUGIN  -I vendor/example   abi_measure.c
 * Both arms are then run and their output compared byte for byte.
 */
#include <stdio.h>
#include <stddef.h>
#include <stdint.h>

#if defined(ABI_HEADER_LOADER)
#include "nccl_net.h"
#define SIDE "loader(src/include/plugin/net/net_v11.h)"
#elif defined(ABI_HEADER_PLUGIN)
#include "nccl/err.h"
#include "nccl/net_device.h"
#include "nccl/common.h"
#include "nccl/net_v11.h"
#define SIDE "plugin(plugins/net/example/nccl/net.h)"
#else
#error "define ABI_HEADER_LOADER or ABI_HEADER_PLUGIN"
#endif

#define P(f) printf("  %-22s offset=%3zu size=%2zu\n", #f, \
                    offsetof(ncclNetProperties_v11_t, f), \
                    sizeof(((ncclNetProperties_v11_t *)0)->f))

int main(void) {
  printf("side=%s\n", SIDE);
  printf("sizeof(ncclNetProperties_v11_t)=%zu\n", sizeof(ncclNetProperties_v11_t));
  printf("sizeof(ncclNet_v11_t)=%zu\n", sizeof(ncclNet_v11_t));
  printf("sizeof(ncclNetVDeviceProps_v11_t)=%zu\n", sizeof(ncclNetVDeviceProps_v11_t));
  printf("sizeof(ncclNetCommConfig_v11_t)=%zu\n", sizeof(ncclNetCommConfig_v11_t));
  P(name);
  P(pciPath);
  P(guid);
  P(ptrSupport);
  P(regIsGlobal);
  P(forceFlush);
  P(speed);
  P(port);
  P(latency);
  P(maxComms);
  P(maxRecvs);
  P(netDeviceType);
  P(netDeviceVersion);
  P(vProps);
  P(maxP2pBytes);
  P(maxCollBytes);
  P(maxMultiRequestSize);
  printf("NCCL_NET_MAX_DEVS_PER_NIC_V11=%d\n", NCCL_NET_MAX_DEVS_PER_NIC_V11);
  /* NCCL_NET_HANDLE_MAXSIZE and NCCL_PTR_* live in the plugin-side net.h and
     are asserted in plugin_accept.c instead; printing them here would make the
     two arms differ for a reason that is not struct layout. */
  return 0;
}
