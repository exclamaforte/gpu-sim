#include <cstdio>
#include <cuda_runtime.h>
// u2o.19 C4/C5 probe: struct read + attribute readthrough on the REAL 5090.
// maxThreadsPerMultiProcessor has no megakern cross-check, so this probe is
// its evidence (struct AND attribute channels must agree with each other and
// with the fake's single-sourced value 1536). max_smem_per_block_optin is
// read here so C5 can be SET from gpu-sim's own measurement, not borrowed.
#define EMIT(label, expr) do { int _v = (expr); \
  if (_e == cudaSuccess) printf(label " : %d\n", _v); } while (0)
int main(){
  cudaDeviceProp p; cudaError_t e = cudaGetDeviceProperties(&p,0);
  if(e!=cudaSuccess){ printf("ERR %s\n",cudaGetErrorString(e)); return 2; }
  printf("device            : %s\n", p.name);
  printf("S.maxThreadsPerSM : %d\n", p.maxThreadsPerMultiProcessor);
  printf("S.sharedMemPerSM  : %zu\n", p.sharedMemPerMultiprocessor);
  printf("S.l2CacheSize     : %d\n", p.l2CacheSize);
  int v = 0; cudaError_t _e;
  /* All four attributes exist in driver_types.h (38/39/81/97). An earlier
     revision of this probe wrongly claimed 39/L2 had none (the enum lives
     in driver_types.h, not cuda_runtime_api.h where the check looked) and
     read one attribute through an uninitialised status -- both fixed here. */
  _e = cudaDeviceGetAttribute(&v, cudaDevAttrMaxThreadsPerMultiProcessor, 0);
  EMIT("A.maxThreadsPerSM", v);
  v = 0;
  _e = cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerMultiprocessor, 0);
  EMIT("A.sharedMemPerSM", v);
  v = 0;
  _e = cudaDeviceGetAttribute(&v, cudaDevAttrL2CacheSize, 0);
  EMIT("A.l2CacheSize", v);
  v = -1;
  _e = cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
  if (_e == cudaSuccess) printf("A.optinPerBlock   : %d\n", v);
  else printf("A.optinPerBlock   : ATTR-ERR %s\n", cudaGetErrorString(_e));
  return 0;
}
