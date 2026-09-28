#include <cstdio>
#include <cstddef>
#include <cuda_runtime.h>
int main(){
#define P(f) printf("%-32s offset=%zu size=%zu\n", #f, offsetof(cudaDeviceProp,f), sizeof(((cudaDeviceProp*)0)->f))
  P(totalGlobalMem); P(sharedMemPerBlock); P(regsPerBlock); P(warpSize);
  P(maxThreadsPerBlock); P(maxThreadsPerMultiProcessor);
  P(sharedMemPerMultiprocessor); P(l2CacheSize);
  P(regsPerMultiprocessor); P(multiProcessorCount); P(major); P(minor);
  printf("sizeof(cudaDeviceProp) = %zu\n", sizeof(cudaDeviceProp));
  return 0;
}
