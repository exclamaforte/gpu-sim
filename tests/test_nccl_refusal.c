#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fakegpu_nccl.h"

int main() {
    char err[256];
    fg_topo_t *t = fg_topo_load("tests/topo_p1.bin", err, sizeof(err));
    if (!t) { printf("load failed: %s\n", err); return 1; }
    int bl = -1;
    char regime[64];
    double pred = fg_nccl_predict(t, "ring_allreduce", 1000, 0, 2, &bl, regime, sizeof(regime));
    printf("fg_nccl_predict (loss_ppm=1e6): pred=%.1f (expect FG_NCCL_REFUSED=%.1f) regime=%s\n",
           pred, (double)FG_NCCL_REFUSED, regime);
    fg_topo_free(t);
    if (pred != FG_NCCL_REFUSED) {
        printf("FAIL: C model did not refuse\n");
        return 1;
    }
    printf("PASS: C model refuses p>=1\n");
    return 0;
}
