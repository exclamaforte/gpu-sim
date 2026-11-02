#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){fprintf(stderr,"{\"error\":\"%s:%d %s\"}\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 2;} } while(0)

__global__ void k_empty(void){}

__global__ void k_strided_read(const float*in,float*out,int n,int stride){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < n) { int idx = i*stride; if(idx < n) out[i]=in[idx]; else out[i]=0.0f; }
}

__global__ void k_interleaved_read(const float*in,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < n){ int idx = (i*7) % n; out[i]=in[idx]; }
}

__global__ void k_block_cyclic(const float*in,float*out,int n,int block){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i < n){ int idx = (i*block) % n; if(idx < n) out[i]=in[idx]; else out[i]=0.0f; }
}

static int cmp_d(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}

int main(int argc,char**argv){
    if(argc<6){fprintf(stderr,"usage: %s <family> <n> <block> <trials> <batch>\n",argv[0]);return 1;}
    const char*family=argv[1];
    int n=atoi(argv[2]);
    int block=atoi(argv[3]);
    int trials=atoi(argv[4]);
    int batch=atoi(argv[5]);
    if(n<1||block<1||trials<1||batch<1){fprintf(stderr,"{\"error\":\"bad args\"}\n");return 1;}

    int grid=(n+block-1)/block; if(grid<1)grid=1; if(grid>65535)grid=65535;
    int grid_capped = ((n+block-1)/block > 65535) ? 1 : 0;

    float *d_in=NULL,*d_out=NULL;
    float *h=NULL;
    CHECK(cudaMalloc((void**)&d_in,(size_t)n*sizeof(float)));
    CHECK(cudaMalloc((void**)&d_out,(size_t)n*sizeof(float)));
    h=(float*)malloc((size_t)n*sizeof(float));
    for(int i=0;i<n;i++) h[i]=(float)(i&1023);
    CHECK(cudaMemcpy(d_in,h,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));

    cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));

#define LAUNCH_EMPTY() k_empty<<<grid,block>>>()

#define LAUNCH_SCALE() \
    do { \
        if(strcmp(family,"strided_read")==0) k_strided_read<<<grid,block>>>(d_in,d_out,n,4); \
        else if(strcmp(family,"interleaved_read")==0) k_interleaved_read<<<grid,block>>>(d_in,d_out,n); \
        else if(strcmp(family,"block_cyclic")==0) k_block_cyclic<<<grid,block>>>(d_in,d_out,n,block); \
    } while(0)

    for(int w=0;w<20;w++){ LAUNCH_EMPTY(); }
    CHECK(cudaDeviceSynchronize());
    for(int w=0;w<20;w++){ LAUNCH_SCALE(); }
    CHECK(cudaDeviceSynchronize());

    for(int phase=0;phase<2;phase++){
        int is_empty = (phase==0);
        double *us=(double*)malloc((size_t)trials*sizeof(double));
        for(int t=0;t<trials;t++){
            float ms=0.0f;
            CHECK(cudaEventRecord(start,0));
            for(int b=0;b<batch;b++){
                if(is_empty) LAUNCH_EMPTY(); else LAUNCH_SCALE();
            }
            CHECK(cudaEventRecord(stop,0));
            CHECK(cudaEventSynchronize(stop));
            CHECK(cudaEventElapsedTime(&ms,start,stop));
            us[t]=(double)ms*1000.0/(double)batch;
            printf("{\"family\":\"%s\",\"mode\":\"%s\",\"trial\":%d,\"batch\":%d,\"us\":%.6f,\"n\":%d,\"grid_capped\":%d}\n",
                family, is_empty?"empty":"scale", t, batch, us[t], n, grid_capped);
            fflush(stdout);
        }
        double *sorted=(double*)malloc((size_t)trials*sizeof(double));
        memcpy(sorted,us,(size_t)trials*sizeof(double));
        qsort(sorted,(size_t)trials,sizeof(double),cmp_d);
        int p05=(int)(trials*0.05), p95=(int)(trials*0.95);
        if(p95>=trials)p95=trials-1; if(p05<0)p05=0;
        printf("{\"summary\":true,\"family\":\"%s\",\"mode\":\"%s\",\"n\":%d,\"block\":%d,\"grid\":%d,\"batch\":%d,\"trials\":%d,\"median_us\":%.6f,\"p05_us\":%.6f,\"p95_us\":%.6f,\"min_us\":%.6f,\"max_us\":%.6f,\"grid_capped\":%d}\n",
            family, is_empty?"empty":"scale", n, block, grid, batch, trials,
            sorted[trials/2], sorted[p05], sorted[p95], sorted[0], sorted[trials-1], grid_capped);
        fflush(stdout);
        free(us); free(sorted);
    }

    if(d_in)cudaFree(d_in); if(d_out)cudaFree(d_out);
    if(h)free(h);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
