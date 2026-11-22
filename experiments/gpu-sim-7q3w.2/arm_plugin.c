/* gpu-sim-7q3w.2: 2 ranks, 2 processes, our v11 net plugin, real libnccl.
 *
 * This arm answers ONE question: did NCCL take our plugin and drive it (accept /
 * isend / irecv), or did it refuse it and fall back to Socket?  It is therefore
 * deliberately NOT a numerics test -- the simulator does not execute NCCL's
 * device kernels (gpu-sim-7q3w.3 measured that), so d_out is expected to hold
 * garbage and the bad count is PRINTED, never asserted.  What is asserted is the
 * plugin's own counters, which a refused plugin cannot move.
 *
 * Two ranks on one box must look like two nodes to NCCL: NCCL_HOSTID (present in
 * libnccl 2.32.6) does that without UTS namespaces, and the fake CUDA layer gives
 * each rank its own virtual device (this box has exactly one real GPU).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <dlfcn.h>
#include <cuda_runtime.h>
#include <nccl.h>

#define NCCLCHECK(c)                                                          \
  do {                                                                        \
    ncclResult_t r_ = (c);                                                    \
    if (r_ != ncclSuccess) {                                                  \
      fprintf(stderr, "ARM FAILED: %s at %s:%d: %s\n", #c, __FILE__, __LINE__, \
              ncclGetErrorString(r_));                                        \
      return 1;                                                               \
    }                                                                         \
  } while (0)

#define CUDACHECK(c)                                                          \
  do {                                                                        \
    cudaError_t e_ = (c);                                                     \
    if (e_ != cudaSuccess) {                                                   \
      fprintf(stderr, "ARM FAILED: %s at %s:%d: %s\n", #c, __FILE__, __LINE__, \
              cudaGetErrorString(e_));                                        \
      return 1;                                                               \
    }                                                                         \
  } while (0)

typedef struct {
  unsigned long long getProperties_calls, devices_calls, init_calls;
  unsigned long long listen_calls, connect_calls, accept_calls;
  unsigned long long regMr_calls, deregMr_calls;
  unsigned long long isend_calls, irecv_calls, test_calls, close_calls;
  unsigned long long bytes_claimed, devs_reported;
  unsigned long long props_speed, props_guid;
} counters_t;

static int read_id(const char *dir, ncclUniqueId *uid) {
  char p[512];
  snprintf(p, sizeof p, "%s/id", dir);
  FILE *f = fopen(p, "rb");
  if (!f) return -1;
  size_t n = fread(uid, sizeof *uid, 1, f);
  fclose(f);
  return n == 1 ? 0 : -1;
}

static int wait_id(const char *dir, ncclUniqueId *uid, int seconds) {
  for (int i = 0; i < seconds * 100; i++) {
    if (read_id(dir, uid) == 0) return 0;
    usleep(10000);
  }
  return -1;
}

int main(int argc, char **argv) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s <rank> <world> [n_f32]\n", argv[0]);
    return 2;
  }
  int rank = atoi(argv[1]), world = atoi(argv[2]);
  size_t n = argc > 3 ? (size_t)strtoull(argv[3], NULL, 10) : 1048576;

  const char *dir = getenv("ARM_ID_DIR");
  ncclUniqueId uid;
  if (rank == 0) {
    const char *hostid = getenv("NCCL_HOSTID");
    printf("ARM hostid=%s plugin=%s\n", hostid ? hostid : "(unset)",
           getenv("NCCL_NET_PLUGIN") ? getenv("NCCL_NET_PLUGIN") : "(unset)");
    NCCLCHECK(ncclGetUniqueId(&uid));
    char p[512];
    snprintf(p, sizeof p, "%s/id.tmp", dir);
    FILE *f = fopen(p, "wb");
    if (!f) { perror("fopen id"); return 2; }
    int ok = fwrite(&uid, sizeof uid, 1, f) == 1;
    fclose(f);
    if (!ok) { fprintf(stderr, "write id failed\n"); return 2; }
    char final_p[512];
    snprintf(final_p, sizeof final_p, "%s/id", dir);
    /* atomic publish: a peer must never read a half-written id */
    if (rename(p, final_p) != 0) { perror("rename id"); return 2; }
  } else if (wait_id(dir, &uid, 120) != 0) {
    fprintf(stderr, "ARM FAILED: no unique id from rank0\n");
    return 2;
  }

  int dev = 0;
  if (getenv("ARM_CUDA_DEV")) dev = atoi(getenv("ARM_CUDA_DEV"));
  CUDACHECK(cudaSetDevice(dev));

  size_t bytes = n * sizeof(float);
  float *d_in = NULL, *d_out = NULL, *h = NULL;
  CUDACHECK(cudaMalloc((void**)&d_in, bytes));
  CUDACHECK(cudaMalloc((void**)&d_out, bytes));
  h = malloc(bytes);
  if (!h) { perror("malloc"); return 2; }
  for (size_t i = 0; i < n; i++) h[i] = (float)(rank + 1);
  CUDACHECK(cudaMemcpy(d_in, h, bytes, cudaMemcpyHostToDevice));

  ncclComm_t comm = NULL;
  NCCLCHECK(ncclCommInitRank(&comm, world, uid, rank));
  printf("ARM ok init rank=%d world=%d dev=%d\n", rank, world, dev);

  NCCLCHECK(ncclAllReduce(d_in, d_out, n, ncclFloat, ncclSum, comm, NULL));
  CUDACHECK(cudaDeviceSynchronize());
  size_t bad = 0;
  CUDACHECK(cudaMemcpy(h, d_out, bytes, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < n; i++) if (h[i] != 3.0f) bad++;
  /* NOT a failure: the simulator has no kernels. Printed so nobody reads the
     collective's completion as proof that bytes were reduced. */
  printf("ARM info allreduce n=%zu bad=%zu value0=%.3f expected=3.0 "
         "(kernels are not executed by the simulator)\n", n, bad, (double)h[0]);

  /* THE COUNTERS ARE READ AFTER ncclCommDestroy, and that is the whole fix: the
     data path is driven during comm teardown (with the simulator's kernels being
     no-ops, the proxy only runs then), so the earlier pre-destroy print read
     isend=0 while the atexit dump read 16. The plugin's nccl_plugin_dump() takes
     ONE snapshot, writes it and returns it, so the line below and the JSON file
     the harness asserts on are the same numbers by construction. */
  ncclCommDestroy(comm);

  const char *plugin = getenv("NCCL_NET_PLUGIN");
  void *h_plug = plugin ? dlopen(plugin, RTLD_NOW | RTLD_NOLOAD) : NULL;
  counters_t snap;
  int have = 0;
  if (h_plug) {
    int (*dump)(const char*, counters_t*) =
        (int (*)(const char*, counters_t*)) dlsym(h_plug, "nccl_plugin_dump");
    if (dump && dump(getenv("FAKEIB_PLUGIN_COUNTERS"), &snap) == 0) have = 1;
  }
  if (have) {
    printf("ARM counters rank=%d getProperties=%llu devices=%llu init=%llu "
           "listen=%llu connect=%llu accept=%llu regMr=%llu isend=%llu "
           "irecv=%llu test=%llu bytes_claimed=%llu speed=%llu guid=0x%llx\n",
           rank, snap.getProperties_calls, snap.devices_calls, snap.init_calls,
           snap.listen_calls, snap.connect_calls, snap.accept_calls,
           snap.regMr_calls, snap.isend_calls, snap.irecv_calls, snap.test_calls,
           snap.bytes_claimed, snap.props_speed, snap.props_guid);
  } else {
    printf("ARM counters rank=%d UNAVAILABLE (plugin not resident?)\n", rank);
  }

  cudaFree(d_in);
  cudaFree(d_out);
  free(h);
  printf("ARM OK rank=%d\n", rank);
  return 0;
}
