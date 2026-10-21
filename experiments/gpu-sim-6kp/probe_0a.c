/* probe_0a: does the interposer reach the runtime calls? */
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <cuda_runtime.h>

int main() {
    /* Check if cudaGetDeviceCount resolves to libfakegpu.so */
    void *h = dlopen("libfakegpu.so", RTLD_NOW | RTLD_NOLOAD);
    if (!h) {
        /* try loading it */
        h = dlopen("libfakegpu.so", RTLD_NOW);
    }
    printf("libfakegpu.so handle: %p\n", h);
    
    void *sym = dlsym(RTLD_DEFAULT, "cudaGetDeviceCount");
    printf("cudaGetDeviceCount resolved to: %p\n", sym);
    
    /* Get the library name for this symbol */
    Dl_info info;
    if (dladdr(sym, &info)) {
        printf("  library: %s\n", info.dli_fname);
        printf("  symbol:  %s\n", info.dli_sname);
    }
    
    /* Call it */
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    printf("cudaGetDeviceCount() = %d (err=%d)\n", count, err);
    
    /* Check cudaSetDevice */
    sym = dlsym(RTLD_DEFAULT, "cudaSetDevice");
    if (dladdr(sym, &info)) {
        printf("cudaSetDevice resolved to: %s (%p)\n", info.dli_fname, sym);
    }
    
    /* Check cudaMalloc */
    sym = dlsym(RTLD_DEFAULT, "cudaMalloc");
    if (dladdr(sym, &info)) {
        printf("cudaMalloc resolved to: %s (%p)\n", info.dli_fname, sym);
    }
    
    return 0;
}
