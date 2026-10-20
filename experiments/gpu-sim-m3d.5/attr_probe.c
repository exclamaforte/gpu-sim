/* Positive control for the shim's audit facility: call a shim-served driver
   entry point directly and prove its counter is NONZERO, so a zero counter in
   the option-(a) audit means "not called", not "counter broken". */
#include <stdio.h>
#include <cuda.h>
#include <dlfcn.h>
#include <stdlib.h>
typedef CUresult (*attr_fn)(int *, CUdevice_attribute, CUdevice);
int main(void) {
    const char *lib = getenv("FAKECUDA_SO");
    void *h = dlopen(lib ? lib : "libfakecuda.so", RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("dlopen libfakecuda.so failed: %s\n", dlerror()); return 1; }
    attr_fn f = (attr_fn)dlsym(h, "cuDeviceGetAttribute");
    if (!f) { printf("dlsym cuDeviceGetAttribute failed\n"); return 1; }
    int v = -1;
    CUresult r = f(&v, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0);
    printf("direct cuDeviceGetAttribute rc=%d value=%d (shim audit counter increments below)\n", (int)r, v);
    return 0;
}
