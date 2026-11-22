/* gpu-sim-7q3w.2: a v11 net plugin NCCL ACCEPTS, built from the public header.
 *
 * Step 1 (gpu-sim-7q3w.1) made NCCL load a plugin and refuse it, deterministically,
 * because the vtable was a byte-guess and the uninitialised props struct SIGSEGVed
 * 1 run in 5.  This step is acceptance: every field this plugin fills is named in
 * the vendored public header (plugins/net/example/nccl/net.h at the tag recorded
 * in vendor/nccl/nccl.tag), and abi_measure.c proves that header and the header
 * NCCL itself was compiled against agree on every offset.
 *
 * SCOPE, stated so nothing here can be mistaken for more: the connection object is
 * REAL (a TCP listening socket on loopback, so listen/connect/accept are exercised
 * exactly as NCCL drives them), but the DATA path is a NULL transport -- isend and
 * irecv record what they were asked to move, report completion, and move nothing.
 * NCCL's own docs put byte movement in epic steps 3-5; the counter is named
 * bytes_claimed, not bytes_moved, and every run prints NULL DATA PATH once so a
 * reader cannot mistake a completed collective for a transported one.
 *
 * Instrumentation is the point of the acceptance test: the counters are what prove
 * NCCL entered accept()/isend() with our plugin, rather than a log line that could
 * be satisfied by a plugin that was loaded and never used.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>

#include "nccl/err.h"
#include "nccl/net_device.h"
#include "nccl/common.h"
#include "nccl/net.h"

/* ---- ABI pins: these are the numbers abi_measure.c prints from BOTH headers ---- */
_Static_assert(sizeof(ncclNetProperties_v11_t) == 112, "props must be 112 bytes");
_Static_assert(sizeof(ncclNet_v11_t) == 176, "v11 vtable must be 22 pointers");
_Static_assert(sizeof(ncclNetVDeviceProps_v11_t) == 20, "vProps must be 20 bytes");
_Static_assert(offsetof(ncclNetProperties_v11_t, vProps) == 64, "vProps at 64");
_Static_assert(offsetof(ncclNetProperties_v11_t, maxP2pBytes) == 88, "maxP2p at 88");
_Static_assert(offsetof(ncclNetProperties_v11_t, maxMultiRequestSize) == 104,
               "maxMultiRequestSize at 104");
_Static_assert(NCCL_NET_HANDLE_MAXSIZE == 128, "handle blob is 128 bytes");
_Static_assert(NCCL_PTR_HOST == 1 && NCCL_PTR_CUDA == 2, "ptrSupport bits");

#define PLUGIN_NAME "fakeib-net"
#define MAGIC 0x46494231u /* "FIB1" */

typedef struct {
  uint64_t getProperties_calls, devices_calls, init_calls;
  uint64_t listen_calls, connect_calls, accept_calls;
  uint64_t regMr_calls, deregMr_calls;
  uint64_t isend_calls, irecv_calls, test_calls, close_calls;
  uint64_t bytes_claimed, devs_reported;
  uint64_t props_speed, props_guid;
} counters_t;

static counters_t g_c;
static ncclDebugLogger_t g_log;
static int g_nulllogged;

/* Exported so the probe can read the same numbers the plugin counted, even when
   the probe loaded NCCL (and NCCL dlopened us) rather than linking us directly. */
__attribute__((visibility("default"))) counters_t* nccl_plugin_counters(void) {
  return &g_c;
}

__attribute__((visibility("default"))) const char* nccl_plugin_abi(void) {
  return "props=112 vtable=176 tag=v2.32.6-1";
}

static int g_dumped;   /* set by the explicit dump, which suppresses the atexit one */

static void write_counters_file(const char* path, const counters_t* c) {
  if (!path) return;
  char tmp[4096];
  snprintf(tmp, sizeof tmp,
           "{\"getProperties_calls\":%llu,\"devices_calls\":%llu,"
           "\"init_calls\":%llu,\"listen_calls\":%llu,\"connect_calls\":%llu,"
           "\"accept_calls\":%llu,\"regMr_calls\":%llu,\"deregMr_calls\":%llu,"
           "\"isend_calls\":%llu,\"irecv_calls\":%llu,\"test_calls\":%llu,"
           "\"close_calls\":%llu,\"bytes_claimed\":%llu,\"devs_reported\":%llu,"
           "\"props_speed\":%llu,\"props_guid\":%llu}",
           (unsigned long long)c->getProperties_calls,
           (unsigned long long)c->devices_calls,
           (unsigned long long)c->init_calls,
           (unsigned long long)c->listen_calls,
           (unsigned long long)c->connect_calls,
           (unsigned long long)c->accept_calls,
           (unsigned long long)c->regMr_calls,
           (unsigned long long)c->deregMr_calls,
           (unsigned long long)c->isend_calls,
           (unsigned long long)c->irecv_calls,
           (unsigned long long)c->test_calls,
           (unsigned long long)c->close_calls,
           (unsigned long long)c->bytes_claimed,
           (unsigned long long)c->devs_reported,
           (unsigned long long)c->props_speed,
           (unsigned long long)c->props_guid);
  FILE* f = fopen(path, "w");
  if (!f) return;
  fputs(tmp, f);
  fclose(f);
}

/* gpu-sim-7q3w.2 (reviewer finding): the arm used to print the counters BEFORE
   ncclCommDestroy, which is when NCCL actually drives the data path -- so the
   printed line said isend=0 while the atexit-dumped file said 16. Two numbers
   for one fact is exactly the defect the bead is about. This entry point takes
   ONE snapshot, writes it, and hands the same snapshot back, so the printed line
   and the file cannot disagree; it also suppresses the atexit write so nothing
   overwrites the snapshot after the arm printed it. */
__attribute__((visibility("default")))
int nccl_plugin_dump(const char* path, counters_t* out) {
  counters_t snap = g_c;
  if (out) *out = snap;
  write_counters_file(path, &snap);
  g_dumped = 1;
  return 0;
}

static void dump_counters(void) {
  if (g_dumped) return;
  write_counters_file(getenv("FAKEIB_PLUGIN_COUNTERS"), &g_c);
}

__attribute__((constructor)) static void on_load(void) {
  /* The two-run control in run.sh depends on this line existing: the WARN arm
     must NOT show it, so the positive arm cannot be satisfied by our own print. */
  fprintf(stderr, "PLUGIN_LOADED %s\n", PLUGIN_NAME);
  fflush(stderr);
  atexit(dump_counters);
}

static int env_int(const char* name, int dflt) {
  const char* v = getenv(name);
  if (!v || !*v) return dflt;
  return atoi(v);
}

static unsigned long long env_ull(const char* name, unsigned long long dflt) {
  const char* v = getenv(name);
  if (!v || !*v) return dflt;
  return strtoull(v, NULL, 0);
}

/* ---------------- the v11 calls ---------------- */

static ncclResult_t p_init(void** ctx, uint64_t commId, ncclNetCommConfig_v11_t* config,
                           ncclDebugLogger_t logFunction, ncclProfilerCallback_t profFunction) {
  (void)commId; (void)config; (void)profFunction;
  g_c.init_calls++;
  g_log = logFunction;
  if (ctx) *ctx = (void*)(uintptr_t)MAGIC;
  return ncclSuccess;
}

static ncclResult_t p_devices(int* ndev) {
  g_c.devices_calls++;
  int n = env_int("FAKEIB_PLUGIN_DEVS", 1);
  if (n < 1) n = 1;
  *ndev = n;
  g_c.devs_reported = (uint64_t)n;
  return ncclSuccess;
}

static ncclResult_t p_getProperties(int dev, ncclNetProperties_v11_t* props) {
  g_c.getProperties_calls++;
  if (dev < 0 || dev >= env_int("FAKEIB_PLUGIN_DEVS", 1)) return ncclInvalidArgument;
  memset(props, 0, sizeof(*props));
  /* Every value below is defined by our interposer, not by the hardware, so the
     probe can assert it: a plugin that "accepted" but filled garbage cannot pass. */
  static char name[64];
  snprintf(name, sizeof name, "%s", PLUGIN_NAME);
  static char pci[128];
  snprintf(pci, sizeof pci, "0000:%s", getenv("FAKEIB_PLUGIN_PCI") ?: "01:00.0");
  props->name = name;
  props->pciPath = pci;
  props->guid = env_ull("FAKEIB_PLUGIN_GUID", 0xFACE1B0000000001ULL);
  props->ptrSupport = NCCL_PTR_HOST;
  props->regIsGlobal = 0;
  props->forceFlush = 0;
  props->speed = env_int("FAKEIB_PLUGIN_SPEED_MBPS", 400000); /* 400 Gb/s */
  props->port = 1;
  props->latency = (float)atof(getenv("FAKEIB_PLUGIN_LATENCY_US") ?: "1.2");
  props->maxComms = env_int("FAKEIB_PLUGIN_MAX_COMMS", 1 << 20);
  props->maxRecvs = env_int("FAKEIB_PLUGIN_MAX_RECVS", 8);
  props->netDeviceType = NCCL_NET_DEVICE_HOST;
  props->netDeviceVersion = NCCL_NET_DEVICE_INVALID_VERSION;
  props->vProps.ndevs = 1;
  props->vProps.devs[0] = dev;
  props->maxP2pBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxCollBytes = NCCL_MAX_NET_SIZE_BYTES;
  props->maxMultiRequestSize = env_int("FAKEIB_PLUGIN_MAX_MULTI", 8);
  g_c.props_speed = (uint64_t)props->speed;
  g_c.props_guid = props->guid;
  return ncclSuccess;
}

typedef struct { uint32_t magic; uint32_t ip; uint16_t port; } endpoint_t;

typedef struct { int fd; } comm_t;

static ncclResult_t p_listen(void* ctx, int dev, void* handle, void** listenComm) {
  (void)ctx; (void)dev;
  g_c.listen_calls++;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return ncclSystemError;
  int one = 1;
  setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
  struct sockaddr_in a;
  memset(&a, 0, sizeof a);
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  a.sin_port = 0;
  if (bind(fd, (struct sockaddr*)&a, sizeof a) != 0 || listen(fd, 8) != 0) {
    close(fd);
    return ncclSystemError;
  }
  socklen_t alen = sizeof a;
  if (getsockname(fd, (struct sockaddr*)&a, &alen) != 0) { close(fd); return ncclSystemError; }
  /* The handle blob is the address NCCL ships to the peer over bootstrap. */
  endpoint_t ep = { MAGIC, ntohl(a.sin_addr.s_addr), ntohs(a.sin_port) };
  memset(handle, 0, NCCL_NET_HANDLE_MAXSIZE);
  memcpy(handle, &ep, sizeof ep);
  comm_t* lc = calloc(1, sizeof *lc);
  if (!lc) { close(fd); return ncclSystemError; }
  lc->fd = fd;
  *listenComm = lc;
  return ncclSuccess;
}

static ncclResult_t p_connect(void* ctx, int dev, void* handle, void** sendComm,
                              ncclNetDeviceHandle_v11_t** sendDevComm) {
  (void)ctx; (void)dev;
  g_c.connect_calls++;
  endpoint_t ep;
  memcpy(&ep, handle, sizeof ep);
  if (ep.magic != MAGIC) return ncclInvalidArgument;
  int fd = socket(AF_INET, SOCK_STREAM, 0);
  if (fd < 0) return ncclSystemError;
  struct sockaddr_in a;
  memset(&a, 0, sizeof a);
  a.sin_family = AF_INET;
  a.sin_addr.s_addr = htonl(ep.ip);
  a.sin_port = htons(ep.port);
  /* Loopback + a listening socket created moments ago: a short retry covers the
     window where the peer has been told the handle but has not yet accepted. */
  int rc = -1;
  for (int i = 0; i < 200 && rc != 0; i++) {
    rc = connect(fd, (struct sockaddr*)&a, sizeof a);
    if (rc != 0 && errno != ECONNREFUSED) break;
    if (rc != 0) usleep(10000);
  }
  if (rc != 0) { int e = errno; close(fd); errno = e; return ncclSystemError; }
  comm_t* sc = calloc(1, sizeof *sc);
  if (!sc) { close(fd); return ncclSystemError; }
  sc->fd = fd;
  *sendComm = sc;
  if (sendDevComm) *sendDevComm = NULL;
  return ncclSuccess;
}

static ncclResult_t p_accept(void* listenComm, void** recvComm,
                             ncclNetDeviceHandle_v11_t** recvDevComm) {
  g_c.accept_calls++;
  comm_t* lc = listenComm;
  int fd = accept(lc->fd, NULL, NULL);
  if (fd < 0) return ncclSystemError;
  comm_t* rc = calloc(1, sizeof *rc);
  if (!rc) { close(fd); return ncclSystemError; }
  rc->fd = fd;
  *recvComm = rc;
  if (recvDevComm) *recvDevComm = NULL;
  return ncclSuccess;
}

static ncclResult_t p_regMr(void* comm, void* data, size_t size, int type, void** mhandle) {
  (void)comm; (void)data; (void)size;
  g_c.regMr_calls++;
  if (type != NCCL_PTR_HOST) return ncclInvalidArgument; /* we only claim HOST */
  *mhandle = data;
  return ncclSuccess;
}

static ncclResult_t p_regMrDmaBuf(void* comm, void* data, size_t size, int type,
                                  uint64_t offset, int fd, void** mhandle) {
  (void)comm; (void)data; (void)size; (void)type; (void)offset;
  if (fd < 0) return ncclInvalidArgument; /* a capability probe, not a registration */
  return ncclInternalError;               /* honest: no dmabuf support claimed */
}

static ncclResult_t p_deregMr(void* comm, void* mhandle) {
  (void)comm; (void)mhandle;
  g_c.deregMr_calls++;
  return ncclSuccess;
}

typedef struct { int done; int64_t bytes; } req_t;

static ncclResult_t p_isend(void* sendComm, void* data, size_t size, int tag, void* mhandle,
                            void* phandle, void** request) {
  (void)sendComm; (void)data; (void)tag; (void)mhandle; (void)phandle;
  g_c.isend_calls++;
  if (!g_nulllogged && g_log) {
    g_nulllogged = 1;
    g_log(NCCL_LOG_INFO, NCCL_INIT | NCCL_NET, __FILE__, __LINE__,
          "NULL DATA PATH: %s claims completion for transfers without moving "
          "bytes (byte movement is epic steps 3-5)", PLUGIN_NAME);
  }
  g_c.bytes_claimed += size;
  req_t* r = calloc(1, sizeof *r);
  if (!r) return ncclSystemError;
  r->done = 1;
  r->bytes = (int64_t)size;
  *request = r;
  return ncclSuccess;
}

static ncclResult_t p_irecv(void* recvComm, int n, void** data, size_t* sizes, int* tags,
                            void** mhandles, void** phandles, void** request) {
  (void)recvComm; (void)data; (void)tags; (void)mhandles; (void)phandles;
  g_c.irecv_calls++;
  size_t total = 0;
  for (int i = 0; i < n; i++) total += sizes ? sizes[i] : 0;
  g_c.bytes_claimed += total;
  req_t* r = calloc(1, sizeof *r);
  if (!r) return ncclSystemError;
  r->done = 1;
  r->bytes = (int64_t)total;
  *request = r;
  return ncclSuccess;
}

static ncclResult_t p_iflush(void* recvComm, int n, void** data, int* sizes,
                             void** mhandles, void** request) {
  (void)recvComm; (void)data; (void)sizes; (void)mhandles;
  req_t* r = calloc(1, sizeof *r);
  if (!r) return ncclSystemError;
  r->done = 1;
  *request = r;
  return ncclSuccess;
}

static ncclResult_t p_test(void* request, int* done, int* sizes) {
  g_c.test_calls++;
  req_t* r = request;
  if (!r) { *done = 1; return ncclSuccess; }
  *done = r->done;
  if (sizes && r->bytes > 0) sizes[0] = (int)r->bytes;
  return ncclSuccess;
}

static ncclResult_t p_closeSend(void* sendComm) {
  g_c.close_calls++;
  comm_t* c = sendComm;
  if (c) { if (c->fd >= 0) close(c->fd); free(c); }
  return ncclSuccess;
}

static ncclResult_t p_closeRecv(void* recvComm) {
  g_c.close_calls++;
  comm_t* c = recvComm;
  if (c) { if (c->fd >= 0) close(c->fd); free(c); }
  return ncclSuccess;
}

static ncclResult_t p_closeListen(void* listenComm) {
  g_c.close_calls++;
  comm_t* c = listenComm;
  if (c) { if (c->fd >= 0) close(c->fd); free(c); }
  return ncclSuccess;
}

static ncclResult_t p_getDeviceMr(void* comm, void* mhandle, void** dptr_mhandle) {
  (void)comm; (void)mhandle;
  if (dptr_mhandle) *dptr_mhandle = NULL;
  return ncclInvalidUsage; /* no device offload */
}

static ncclResult_t p_irecvConsumed(void* recvComm, int n, void* request) {
  (void)recvComm; (void)n; (void)request;
  return ncclSuccess;
}

static ncclResult_t p_makeVDevice(int* d, ncclNetVDeviceProps_v11_t* props) {
  if (props->ndevs != 1) return ncclInvalidArgument;
  *d = props->devs[0];
  return ncclSuccess;
}

static ncclResult_t p_finalize(void* ctx) {
  (void)ctx;
  return ncclSuccess;
}

static ncclResult_t p_setNetAttr(void* ctx, ncclNetAttr_v11_t* netAttr) {
  (void)ctx; (void)netAttr;
  return ncclInvalidUsage; /* we do not advertise traffic-class control */
}

/* The symbol NCCL looks for: ncclNetPlugin_v11.  v12 is deliberately NOT
   exported -- 2.32 prefers v12 when present (m3d.4), and this step is the v11
   acceptance.  A future step can add v12 with its own ABI measurement. */
__attribute__((visibility("default")))
ncclNet_v11_t ncclNetPlugin_v11 = {
  .name = PLUGIN_NAME,
  .init = p_init,
  .devices = p_devices,
  .getProperties = p_getProperties,
  .listen = p_listen,
  .connect = p_connect,
  .accept = p_accept,
  .regMr = p_regMr,
  .regMrDmaBuf = p_regMrDmaBuf,
  .deregMr = p_deregMr,
  .isend = p_isend,
  .irecv = p_irecv,
  .iflush = p_iflush,
  .test = p_test,
  .closeSend = p_closeSend,
  .closeRecv = p_closeRecv,
  .closeListen = p_closeListen,
  .getDeviceMr = p_getDeviceMr,
  .irecvConsumed = p_irecvConsumed,
  .makeVDevice = p_makeVDevice,
  .finalize = p_finalize,
  .setNetAttr = p_setNetAttr,
};
