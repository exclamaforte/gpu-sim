#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define CHECK(call) do { cudaError_t e_=(call); if(e_!=cudaSuccess){fprintf(stderr,"{\"error\":\"%s:%d %s\"}\n",__FILE__,__LINE__,cudaGetErrorString(e_)); return 2;} } while(0)

__global__ void k_empty(void){}

__global__ void k_elementwise(const float*in,float*out,float alpha,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n) out[i]=in[i]*alpha+in[i]*0.5f+1.0f;
}

__global__ void k_reduce(const float*in,float*out,int n){
    int tid=blockIdx.x*blockDim.x+threadIdx.x;
    int stride=gridDim.x*blockDim.x;
    float s=0.0f;
    for(int i=tid;i<n;i+=stride) s+=in[i];
    if(tid < gridDim.x*blockDim.x) out[tid]=s;
}

__global__ void k_transpose(const float*in,float*out,int m){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int y=blockIdx.y*blockDim.y+threadIdx.y;
    if(x<m && y<m) out[y*m+x]=in[x*m+y];
}

__global__ void k_softmax(const float*in,float*out,int n){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){ float e=expf(in[i]); out[i]=e/(e+1.0f); }
}

// matmul-like: C (m x m) = A (m x K) * B (K x m), K=64
#define K 64
__global__ void k_matmul(const float*A,const float*B,float*C,int m){
    int row=blockIdx.y*blockDim.y+threadIdx.y;
    int col=blockIdx.x*blockDim.x+threadIdx.x;
    if(row<m && col<m){
        float s=0.0f;
        for(int k=0;k<K;k++) s+=A[row*K+k]*B[k*m+col];
        C[row*m+col]=s;
    }
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
    int is_matmul = (strcmp(family,"matmul")==0);
    int m = (int)sqrtf((float)n);
    if(m<1) m=1;
    if(is_matmul && m*m!=n){ /* adjust n to perfect square */ n=m*m; }

    int grid;
    if(strcmp(family,"transpose")==0||is_matmul){
        int g=(m+block-1)/block; if(g<1)g=1; if(g>65535)g=65535; grid=g; // grid is per-dim; we use 2D
    } else {
        grid=(n+block-1)/block; if(grid<1)grid=1; if(grid>65535)grid=65535;
    }
    int grid_capped = 0;
    if(!is_matmul && !strcmp(family,"transpose")==0){
        if((n+block-1)/block > 65535) grid_capped=1;
    } else {
        if((m+block-1)/block > 65535) grid_capped=1;
    }

    // allocations
    float *d_in=NULL,*d_out=NULL,*d_A=NULL,*d_B=NULL,*d_C=NULL;
    float *h=NULL;
    if(strcmp(family,"elementwise")==0||strcmp(family,"reduction")==0||strcmp(family,"softmax")==0){
        CHECK(cudaMalloc((void**)&d_in,(size_t)n*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_out,(size_t)n*sizeof(float)));
        h=(float*)malloc((size_t)n*sizeof(float));
        for(int i=0;i<n;i++) h[i]=(float)(i&1023);
        CHECK(cudaMemcpy(d_in,h,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
    } else if(strcmp(family,"transpose")==0){
        CHECK(cudaMalloc((void**)&d_in,(size_t)n*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_out,(size_t)n*sizeof(float)));
        h=(float*)malloc((size_t)n*sizeof(float));
        for(int i=0;i<n;i++) h[i]=(float)(i&1023);
        CHECK(cudaMemcpy(d_in,h,(size_t)n*sizeof(float),cudaMemcpyHostToDevice));
    } else if(is_matmul){
        int sz=m*m;
        CHECK(cudaMalloc((void**)&d_A,(size_t)sz*sizeof(float)*0+ (size_t)m*K*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_B,(size_t)K*m*sizeof(float)));
        CHECK(cudaMalloc((void**)&d_C,(size_t)sz*sizeof(float)));
        // init
        float *ha=(float*)malloc((size_t)m*K*sizeof(float));
        float *hb=(float*)malloc((size_t)K*m*sizeof(float));
        for(int i=0;i<m*K;i++) ha[i]=1.0f;
        for(int i=0;i<K*m;i++) hb[i]=1.0f;
        CHECK(cudaMemcpy(d_A,ha,(size_t)m*K*sizeof(float),cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_B,hb,(size_t)K*m*sizeof(float),cudaMemcpyHostToDevice));
        free(ha); free(hb);
    }

    cudaEvent_t start,stop; CHECK(cudaEventCreate(&start)); CHECK(cudaEventCreate(&stop));

#define LAUNCH_EMPTY() k_empty<<<grid,block>>>()

#define LAUNCH_SCALE() \
    do { \
        if(strcmp(family,"elementwise")==0) k_elementwise<<<grid,block>>>(d_in,d_out,2.0f,n); \
        else if(strcmp(family,"reduction")==0) k_reduce<<<grid,block>>>(d_in,d_out,n); \
        else if(strcmp(family,"transpose")==0){ dim3 g((m+15)/16,(m+15)/16); dim3 b(16,16); k_transpose<<<g,b>>>(d_in,d_out,m); } \
        else if(strcmp(family,"softmax")==0) k_softmax<<<grid,block>>>(d_in,d_out,n); \
        else if(is_matmul){ dim3 g((m+15)/16,(m+15)/16); dim3 b(16,16); k_matmul<<<g,b>>>(d_A,d_B,d_C,m); } \
    } while(0)

    // warmup
    for(int w=0;w<20;w++){ LAUNCH_EMPTY(); }
    CHECK(cudaDeviceSynchronize());
    for(int w=0;w<20;w++){ LAUNCH_SCALE(); }
    CHECK(cudaDeviceSynchronize());

    // run empty then scale
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

    // cleanup
    if(d_in)cudaFree(d_in); if(d_out)cudaFree(d_out);
    if(d_A)cudaFree(d_A); if(d_B)cudaFree(d_B); if(d_C)cudaFree(d_C);
    if(h)free(h);
    cudaEventDestroy(start); cudaEventDestroy(stop);
    return 0;
}
