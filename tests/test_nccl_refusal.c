#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fakegpu_nccl.h"

static int check_regime(const char *label, const char *envval) {
    if (envval) setenv("GPUSIM_NCCL_LEGACY_TABLE", envval, 1);
    else unsetenv("GPUSIM_NCCL_LEGACY_TABLE");

    char err[256];
    fg_topo_t *t = fg_topo_load("tests/topo_p1.bin", err, sizeof(err));
    if (!t) { printf("load failed: %s\n", err); return 1; }
    int bl = -1;
    char regime[64];
    double pred = fg_nccl_predict(t, "ring_allreduce", 1000, 0, 2, &bl, regime, sizeof(regime));
    printf("[%s] fg_nccl_predict (loss_ppm=1e6): pred=%.1f (expect FG_NCCL_REFUSED=%.1f) regime=%s\n",
           label, pred, (double)FG_NCCL_REFUSED, regime);
    fg_topo_free(t);
    if (pred != FG_NCCL_REFUSED) {
        printf("FAIL [%s]: C model did not refuse\n", label);
        return 1;
    }
    if (strcmp(regime, "refused") != 0) {
        printf("FAIL [%s]: regime != refused (got %s)\n", label, regime);
        return 1;
    }
    printf("PASS [%s]: C model refuses p>=1\n", label);
    return 0;
}

int main() {
    int rc = 0;
    rc |= check_regime("default", NULL);
    rc |= check_regime("legacy", "1");
    /* restore */
    unsetenv("GPUSIM_NCCL_LEGACY_TABLE");
    return rc;
}
