#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>
#include <nccl.h>
#include <cuda_runtime.h>
int main(int argc,char**argv){int n=(argc>1)?atoi(argv[1]):4;ncclUniqueId u;ncclGetUniqueId(&u);pid_t pids[64];for(int r=0;r<n;r++){pid_t p=fork();if(p==0){cudaSetDevice(r%4);ncclComm_t c;ncclResult_t res=ncclCommInitRank(&c,n,u,r);printf("rank %d init=%d %s\n",r,(int)res,ncclGetErrorString(res));fflush(stdout);if(res==ncclSuccess){int dc=0;cudaGetDeviceCount(&dc);printf("rank %d devices=%d\n",r,dc);fflush(stdout);ncclCommDestroy(c);} _exit(res==ncclSuccess?0:1);}pids[r]=p;}int fail=0;for(int r=0;r<n;r++){int st=0;waitpid(pids[r],&st,0);if(!WIFEXITED(st)||WEXITSTATUS(st)!=0){fail=1;fprintf(stderr,"rank %d bad status %d\n",r,st);}}printf("ALL RANKS %s\n",fail?"FAILED":"OK");fflush(stdout);return fail;}