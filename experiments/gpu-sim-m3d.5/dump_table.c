#define _GNU_SOURCE
#include <stdio.h>
#include <dlfcn.h>
#include <string.h>
typedef int (*ce_t)(const void**, const void*);
int main(void){
  void *h = dlopen("libcuda.so.1", RTLD_NOW|RTLD_LOCAL);
  if(!h){ printf("dlopen failed: %s\n", dlerror()); return 1; }
  ce_t ce = (ce_t)dlsym(h, "cuGetExportTable");
  if(!ce){ printf("no cuGetExportTable\n"); return 1; }
  unsigned char ids[2][16] = {
    {0xf8,0xcf,0xf9,0x51,0x21,0x46,0x8b,0x4e,0xb9,0xe2,0xfb,0x46,0x9e,0x7c,0x0d,0xd9},
    {0x6b,0xd5,0xfb,0x6c,0x5b,0xf4,0xe7,0x4a,0x89,0x87,0xd9,0x39,0x12,0xfd,0x9d,0xf9},
  };
  for(int k=0;k<2;k++){
    const void *tbl=NULL;
    int rc = ce(&tbl, ids[k]);
    printf("uuid[%d] rc=%d table=%p\n", k, rc, tbl);
    if(rc==0 && tbl){
      void **e = (void**)tbl;
      for(int i=0;i<24;i++){
        Dl_info di; const char *nm="?";
        if(e[i] && dladdr(e[i], &di) && di.dli_sname) nm=di.dli_sname;
        printf("  [%2d] %p %s\n", i, e[i], nm);
      }
    }
  }
  return 0;
}
