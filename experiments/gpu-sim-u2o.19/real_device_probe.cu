#include <cstdio>
#include <cuda_runtime.h>
int main(){
  cudaDeviceProp p; cudaError_t e = cudaGetDeviceProperties(&p,0);
  if(e!=cudaSuccess){ printf("ERR %s\n",cudaGetErrorString(e)); return 2; }
  printf("device            : %s\n", p.name);
  printf("maxThreadsPerSM   : %d\n", p.maxThreadsPerMultiProcessor);
  printf("sharedMemPerSM    : %zu\n", p.sharedMemPerMultiprocessor);
  printf("l2CacheSize       : %d\n", p.l2CacheSize);
  printf("sharedMemPerBlock : %zu\n", p.sharedMemPerBlock);
  printf("maxThreadsPerBlock: %d\n", p.maxThreadsPerBlock);
  printf("regsPerBlock      : %d\n", p.regsPerBlock);
  printf("regsPerSM         : %d\n", p.regsPerMultiprocessor);
  printf("multiProcessorCnt : %d\n", p.multiProcessorCount);
  printf("cc                : %d.%d\n", p.major, p.minor);
  printf("totalGlobalMem    : %zu\n", p.totalGlobalMem);
  return 0;
}
