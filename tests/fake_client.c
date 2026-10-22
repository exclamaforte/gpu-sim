/*
 * fake_client.c — self-test workload for libfakegpu.so.
 *
 * Compiled against shim/cuda_runtime.h + shim/nccl.h (no real CUDA needed),
 * this exercises a representative slice of the API. Run directly: it resolves
 * the symbols from libfakegpu.so (which the interposer provides).
 *
 * Exit code: 0 on all-assert success, non-zero on any failure.
 */
#include "cuda_runtime.h"
#include "nccl.h"

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define CHECK(call) do { cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "FAIL %s:%d: '%s' err %d\n", __FILE__, __LINE__, #call, e); return 1; } } while (0)

#define CHECK_NCCL(call) do { ncclResult_t r = (call); \
    if (r != ncclSuccess) { \
        fprintf(stderr, "FAIL %s:%d: '%s' err %d\n", __FILE__, __LINE__, #call, r); return 1; } } while (0)

static int test_device_query(void)
{
    int count = -1;
    CHECK(cudaGetDeviceCount(&count));
    printf("  cudaGetDeviceCount -> %d\n", count);
    if (count < 1) return 2;

    int dev = -1;
    CHECK(cudaGetDevice(&dev));
    printf("  cudaGetDevice -> %d\n", dev);

    struct cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("  cudaGetDeviceProperties[0] name=\"%s\" totalGlobalMem=%zu\n",
           prop.name, prop.totalGlobalMem);
    if (strncmp(prop.name, "FakeGPU", 7) != 0) return 2;
    return 0;
}

static int test_memory(void)
{
    void *dptr = NULL;
    CHECK(cudaMalloc(&dptr, 64));
    if (!dptr) { fprintf(stderr, "FAIL cudaMalloc returned NULL\n"); return 1; }
    printf("  cudaMalloc -> %p\n", dptr);

    /* Value-level round-trip check (gpu-sim-2qe): non-uniform pattern so a
       shifted copy fails too. */
    char host[64], orig[64];
    for (size_t i = 0; i < sizeof(host); i++) { host[i] = (char)(i * 37 + 11); orig[i] = host[i]; }
    CHECK(cudaMemcpyHtoD(dptr, host, sizeof(host)));
    memset(host, 0, sizeof(host));
    CHECK(cudaMemcpyDtoH(host, dptr, sizeof(host)));
    int bad = 0;
    for (size_t i = 0; i < sizeof(host); i++) if (host[i] != orig[i]) { bad = 1; break; }
    if (bad) { fprintf(stderr, "FAIL: memcpy round-trip value mismatch\n"); return 1; }
    printf("  memcpy rounds ok; value round-trip preserved\n");

    char *big = NULL;
    CHECK(cudaMalloc((void **)&big, 1024 * 1024));
    if (!big) return 1;
    CHECK(cudaFree(dptr));
    CHECK(cudaFreeAsync(big, NULL));
    printf("  cudaFree / cudaFreeAsync ok\n");
    return 0;
}

static int test_streams_events(void)
{
    cudaStream_t stream;
    cudaEvent_t event;
    CHECK(cudaStreamCreate(&stream));
    CHECK(cudaEventCreate(&event));
    CHECK(cudaEventRecord(event, stream));
    CHECK(cudaStreamWaitEvent(stream, event, 0));
    CHECK(cudaStreamSynchronize(stream));
    CHECK(cudaEventSynchronize(event));
    CHECK(cudaStreamDestroy(stream));
    CHECK(cudaEventDestroy(event));
    printf("  streams/events ok\n");
    return 0;
}

static int test_kernel_launch(void)
{
    void *fn = (void *)0xdeadbeef;
    CHECK(cudaLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL));
    CHECK(cudaDeviceSynchronize());
    printf("  cudaLaunchKernel + device sync ok\n");
    return 0;
}

static int test_module(void)
{
    cudaModule_t mod;
    cudaFunction_t fn;
    CHECK(cudaModuleLoad(&mod, "fake.ptx"));
    CHECK(cudaModuleGetFunction(&fn, mod, "fake_kernel"));
    if (!fn) return 1;
    printf("  cudaModuleLoad/GetFunction -> mod=%p fn=%p\n", (void *)mod, (void *)fn);
    CHECK(cudaModuleUnload(mod));
    return 0;
}

static int test_driver(void)
{
    CUdevice dev;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn;
    CHECK(cuDeviceGet(&dev, 0));
    CHECK(cuCtxCreate(&ctx, 0, dev));
    CHECK(cuModuleLoad(&mod, "fake.ptx"));
    CHECK(cuModuleGetFunction(&fn, mod, "fake_kernel"));
    CHECK(cuLaunchKernel(fn, 1, 1, 1, 1, 1, 1, 0, NULL, NULL, NULL));
    CHECK(cuCtxDestroy(ctx));
    printf("  driver API (cuDeviceGet/cuCtx/cuModule/cuLaunch) ok\n");
    return 0;
}

static int test_nccl(void)
{
    ncclComm_t comm;
    unsigned long long id = 0;
    CHECK_NCCL(ncclGetUniqueId(&id));
    int nr = 0;
    CHECK_NCCL(ncclCommInitRank(&comm, 2, 0, id));
    CHECK_NCCL(ncclCommCount(comm, &nr));
    printf("  ncclCommInitRank -> comm=%p worldsize=%d\n", (void *)comm, nr);
    if (nr != 2) return 2;

    float send[4] = {0}, recv[4] = {0};
    CHECK_NCCL(ncclAllReduce(send, recv, 4, ncclFloat32, ncclSum, comm, NULL));
    CHECK_NCCL(ncclGroupStart());
    CHECK_NCCL(ncclBroadcast(send, recv, 4, ncclFloat32, 0, comm, NULL));
    CHECK_NCCL(ncclGroupEnd());
    int rank = -1;
    CHECK_NCCL(ncclCommGetRank(comm, &rank));
    printf("  nccl allreduce/broadcast ok; rank=%d\n", rank);
    CHECK_NCCL(ncclCommDestroy(comm));
    return 0;
}

int main(void)
{
    int rc;
    printf("== fakegpu self-test ==\n");
    printf("[device query]\n");  if ((rc = test_device_query()))  return rc;
    printf("[memory]\n");        if ((rc = test_memory()))         return rc;
    printf("[streams/events]\n");if ((rc = test_streams_events()))return rc;
    printf("[kernel launch]\n"); if ((rc = test_kernel_launch()))  return rc;
    printf("[module/driver]\n"); if ((rc = test_module()))         return rc;
    if ((rc = test_driver()))     return rc;
    printf("[nccl]\n");          if ((rc = test_nccl()))           return rc;
    printf("== all checks passed ==\n");
    return 0;
}
