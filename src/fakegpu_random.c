/*
 * fakegpu_random.c — thread-local xorshift64** PRNG + random fill.
 *
 * Seed comes from FAKEGPU_SEED (for reproducibility) or falls back to
 * nanoseconds-of-time so different runs differ. State is per-thread.
 */
#include "fakegpu.h"

#include <errno.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

static __thread uint64_t g_state = 0;

static uint64_t fg_entropy(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    /* mix in pid + a counter-ish value for more entropy */
    uint64_t v = ((uint64_t)ts.tv_sec * 1000000000ULL) ^
                 ((uint64_t)ts.tv_nsec) ^
                 ((uint64_t)getpid() << 16);
    if (v == 0) v = 0x9e3779b97f4a7c15ULL;
    return v;
}

static void fg_seed_once(void)
{
    uint64_t s = fg_cfg()->seed;
    if (s == 0) s = fg_entropy();
    /* xorshift** needs a nonzero state; if the seed is 0, nudge it. */
    if (s == 0) s = 0xdeadbeefcafebabeULL;
    g_state = s;
}

uint64_t fg_rng_next(void)
{
    if (g_state == 0) fg_seed_once();
    uint64_t x = g_state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    g_state = x;
    return x * 5ULL >> 12; /* rotate-free constant mul variant */
}

void fg_fill(void *buf, size_t nbytes)
{
    if (!buf || nbytes == 0) return;
    unsigned char *p = (unsigned char *)buf;
    size_t i;
    for (i = 0; i + 8 <= nbytes; i += 8) {
        uint64_t v = fg_rng_next();
        memcpy(p + i, &v, 8);
    }
    /* remaining tail */
    if (i < nbytes) {
        uint64_t v = fg_rng_next();
        memcpy(p + i, &v, nbytes - i);
    }
}
