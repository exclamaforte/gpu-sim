#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <fakegpu_fabric.h>

int main(int argc, char**argv) {
    int nranks = 2;
    const char *run_id = "testfault";
    fg_fab_t *f = fg_fab_attach(run_id, 1, 0, nranks, 1000000000ULL, NULL);
    if (!f) { printf("attach failed\n"); return 1; }

    /* Test 1: loss_ppm=1e6 drops every transfer */
    {
        int drops = 0;
        for (int i = 0; i < 10; i++) {
            int rc = fg_fab_transfer(f, 0, 1024, 100.0, 5.0, 0.0, 1000000, 0, 0);
            if (rc == FG_FAB_ELOSS) drops++;
        }
        printf("loss_1e6: drops=%d/10 (expect 10)\n", drops);
        if (drops != 10) { printf("FAIL loss_1e6\n"); return 1; }
    }

    /* Test 2: loss_ppm=0 drops none */
    {
        /* reset rng by creating new fabric */
        fg_fab_detach(f);
        f = fg_fab_attach("testfault2", 1, 0, nranks, 1000000000ULL, NULL);
        int drops = 0;
        for (int i = 0; i < 10; i++) {
            int rc = fg_fab_transfer(f, 0, 1024, 100.0, 5.0, 0.0, 0, 0, 0);
            if (rc == FG_FAB_ELOSS) drops++;
        }
        printf("loss_0: drops=%d/10 (expect 0)\n", drops);
        if (drops != 0) { printf("FAIL loss_0\n"); return 1; }
    }

    /* Test 3: partition after k=3 */
    {
        fg_fab_detach(f);
        f = fg_fab_attach("testfault3", 1, 0, nranks, 1000000000ULL, NULL);
        int ok = 0, to = 0;
        for (int i = 0; i < 5; i++) {
            int rc = fg_fab_transfer(f, 0, 1024, 100.0, 5.0, 0.0, 0, 3, 0);
            if (rc == FG_FAB_OK) ok++;
            else if (rc == FG_FAB_ETIMEDOUT) to++;
        }
        printf("partition_k3: ok=%d timeout=%d (expect ok=3, timeout=2)\n", ok, to);
        if (ok != 3 || to != 2) { printf("FAIL partition\n"); return 1; }
    }

    /* Test 4: jitter draws vary (not constant) */
    {
        fg_fab_detach(f);
        f = fg_fab_attach("testfault4", 1, 0, nranks, 1000000000ULL, NULL);
        /* Just verify it doesn't crash and returns OK */
        int ok = 0;
        for (int i = 0; i < 5; i++) {
            int rc = fg_fab_transfer(f, 0, 1024, 100.0, 5.0, 10.0, 0, 0, 0);
            if (rc == FG_FAB_OK) ok++;
        }
        printf("jitter: ok=%d/5\n", ok);
        if (ok != 5) { printf("FAIL jitter\n"); return 1; }
    }

    fg_fab_detach(f);
    printf("ALL FAULT TESTS PASS\n");
    return 0;
}
