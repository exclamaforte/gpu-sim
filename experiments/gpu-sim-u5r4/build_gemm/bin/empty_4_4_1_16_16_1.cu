#include <cuda_runtime.h>
#include <stdio.h>
__global__ void k_empty() {}
int main(){
  dim3 grid(4,4,1), block(16,16,1);
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  for(int w=0;w<20;w++){ k_empty<<<grid,block>>>(); } cudaDeviceSynchronize();
  for(int t=0;t<20;t++){
    float ms=0; cudaEventRecord(s); k_empty<<<grid,block>>>(); cudaEventRecord(e); cudaEventSynchronize(e); cudaEventElapsedTime(&ms,s,e);
    printf("{\"trial\": %d, \"time_ms\": %.6f}\n", t, ms);
  }
  return 0;
}
