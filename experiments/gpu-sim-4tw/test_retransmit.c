/* gpu-sim-4tw: does the fabric's measured retransmit behaviour match the model's
 * 1/(1-p) counterfactual?  See experiments/gpu-sim-4tw/pre_registration.md.
 *
 * Delivers 1000 pieces per (loss_ppm, seed); ratio = attempts/deliveries, compared
 * with the model prediction 1/(1-p) at 5% relative tolerance on the per-point median.
 *
 * Build/run (from repo root, after `make`):
 *   cc -O2 -Iinclude -o build/test_retransmit experiments/gpu-sim-4tw/test_retransmit.c \
 *      -L. -lfakegpu -Wl,-rpath,.
 *   ./build/test_retransmit
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <fakegpu_fabric.h>

#define DELIVERIES 1000u
#define ATTEMPT_BOUND 200000u
#define TOL 0.05

static const int LOSS_PPM[] = {0, 50000, 100000, 250000, 500000};
static const int NPOINTS = (int)(sizeof(LOSS_PPM) / sizeof(LOSS_PPM[0]));
static const int SEEDS[] = {1, 2, 3};
static const int NSEEDS = (int)(sizeof(SEEDS) / sizeof(SEEDS[0]));

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* counter sanity checks (pre-registration section 7) */
static int instrument_ok(void) {
    char id[64];

    setenv("GPUSIM_SEED", "1", 1);
    snprintf(id, sizeof id, "tw4_check_noloss");
    fg_fab_t *f = fg_fab_attach(id, 0, 0, 2, 1000000000ULL, NULL);
    if (!f) return 0;
    for (int i = 0; i < 100; i++) (void)fg_fab_transfer(f, 0, 1024, 100.0, 5.0, 0.0, 0, 0, 0.0, 0);
    uint64_t a = fg_fab_attempt_count(f), s = fg_fab_success_count(f);
    fg_fab_detach(f);
    if (a != 100 || s != 100) {
        printf("INSTRUMENT-FAILED: loss=0 gave attempts=%llu successes=%llu (expect 100/100)\n",
               (unsigned long long)a, (unsigned long long)s);
        return 0;
    }

    snprintf(id, sizeof id, "tw4_check_allloss");
    f = fg_fab_attach(id, 0, 0, 2, 1000000000ULL, NULL);
    if (!f) return 0;
    for (int i = 0; i < 100; i++) (void)fg_fab_transfer(f, 0, 1024, 100.0, 5.0, 0.0, 1000000, 0, 0.0, 0);
    uint64_t a2 = fg_fab_attempt_count(f), s2 = fg_fab_success_count(f);
    fg_fab_detach(f);
    if (a2 != 100 || s2 != 0) {
        printf("INSTRUMENT-FAILED: loss=1e6 gave attempts=%llu successes=%llu (expect 100/0)\n",
               (unsigned long long)a2, (unsigned long long)s2);
        return 0;
    }
    printf("instrument check PASSED (loss=0: 100/100; loss=1e6: 100/0)\n");
    return 1;
}

int main(void) {
    if (!instrument_ok()) return 2;

    int failures = 0;
    printf("\n%-10s %-6s %-10s %-10s %-12s %-10s\n",
           "loss_ppm", "seed", "attempts", "successes", "ratio", "model 1/(1-p)");
    for (int i = 0; i < NPOINTS; i++) {
        int ppm = LOSS_PPM[i];
        double p = ppm / 1000000.0;
        double model = (p >= 1.0) ? 0.0 : 1.0 / (1.0 - p);
        double ratios[NSEEDS];
        for (int k = 0; k < NSEEDS; k++) {
            char seedbuf[32];
            snprintf(seedbuf, sizeof seedbuf, "%d", SEEDS[k]);
            setenv("GPUSIM_SEED", seedbuf, 1);
            char id[64];
            snprintf(id, sizeof id, "tw4_%d_%d", ppm, SEEDS[k]);
            fg_fab_t *f = fg_fab_attach(id, 0, 0, 2, 1000000000ULL, NULL);
            if (!f) { printf("attach failed\n"); return 2; }
            unsigned attempts = 0;
            while (fg_fab_success_count(f) < DELIVERIES) {
                (void)fg_fab_transfer(f, 0, 1024, 100.0, 5.0, 0.0, ppm, 0, 0.0, 0);
                if (++attempts > ATTEMPT_BOUND) {
                    printf("INSTRUMENT-FAILED: attempt bound hit at loss_ppm=%d seed=%d\n",
                           ppm, SEEDS[k]);
                    fg_fab_detach(f);
                    return 2;
                }
            }
            double ratio = (double)attempts / (double)DELIVERIES;
            ratios[k] = ratio;
            printf("%-10d %-6d %-10u %-10u %-12.6f %-10.6f\n",
                   ppm, SEEDS[k], attempts, DELIVERIES, ratio, model);
            fg_fab_detach(f);
        }
        double sorted[NSEEDS];
        memcpy(sorted, ratios, sizeof ratios);
        qsort(sorted, NSEEDS, sizeof(double), cmp_double);
        double med = sorted[NSEEDS / 2];
        double rel = model > 0 ? (med - model) / model : 0.0;
        int ok = (rel <= TOL && rel >= -TOL);
        printf("  -> median ratio %.6f vs model %.6f  rel err %+.4f  %s\n\n",
               med, model, rel, ok ? "WITHIN" : "OUTSIDE");
        if (!ok) failures++;
    }

    if (failures == 0) {
        printf("VERDICT: MODEL-SUPPORTED (all %d points within %d%% tolerance)\n",
               NPOINTS, (int)(TOL * 100));
        return 0;
    }
    printf("VERDICT: MODEL-REFUTED (%d of %d points outside tolerance)\n",
           failures, NPOINTS);
    return 1;
}
