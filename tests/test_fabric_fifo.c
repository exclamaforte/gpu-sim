/* test_fabric_fifo.c -- RED test for the fabric ordering defect (gpu-sim-6n6).
 *
 * Symptom in the field: with the sim fabric in the path, every payload arrives
 * byte-identical to what was sent (32/32 size and digest match in the committed
 * traces) yet the values land at wrong, chunk-aligned offsets -- and at 384KiB
 * one piece never arrives at all (N sent, N-1 received, then timeout).
 *
 * Mechanism this test pins down: fg_fab_recv_max_ex takes the FIRST ready slot
 * whose len fits cap and SILENTLY SKIPS over a queued message that is too large
 * ("continue" on len > cap).  Skipping reorders: the next message is handed to a
 * caller that has not yet received the previous one, so the consumer's notion of
 * which message it is holding is wrong for ever after.  Nothing is corrupt; the
 * pieces are simply matched to the wrong requests.
 *
 * The contract asserted here is the one NCCL requires of a net plugin: delivery
 * is FIFO per peer, and a message that does not fit the caller's capacity is an
 * ERROR (ESIZE), never a reason to hand over the message behind it.
 *
 * Expected today: FAIL, because the first recv (cap 4) receives the SMALL second
 * message instead of reporting that the head message does not fit.
 * Expected after the fix: PASS.
 */
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "fakegpu_fabric.h"

int main(void) {
    char runid[64];
    snprintf(runid, sizeof runid, "fifo-%d-%ld", (int)getpid(), (long)time(NULL));
    uint64_t epoch = 1;
    const int nranks = 2;

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    int rc = 0;
    if (pid == 0) {
        /* ------------- rank 1: sender ------------- */
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, 1, nranks, 5000000000ULL, &err);
        if (!f) { fprintf(stderr, "rank1 attach: %s\n", fg_fab_errstr(err)); _exit(1); }
        if (fg_fab_barrier(f) != FG_FAB_OK) _exit(1);

        uint64_t big = 0xAAAAAAAAAAAAAAAAULL;      /* 8 bytes */
        uint32_t small = 0xBBBBBBBBu;              /* 4 bytes */

        if (fg_fab_send(f, 0, &big, sizeof big) != FG_FAB_OK) { fprintf(stderr, "rank1 send big failed\n"); _exit(1); }
        if (fg_fab_send(f, 0, &small, sizeof small) != FG_FAB_OK) { fprintf(stderr, "rank1 send small failed\n"); _exit(1); }

        /* scenario 2 (reviewer's strengthening): both queued messages exceed the
         * receiver's capacity, so nothing on the queue fits at all. */
        if (fg_fab_barrier(f) != FG_FAB_OK) _exit(1);
        uint64_t big_a = 0xCCCCCCCCCCCCCCCCULL, big_b = 0xDDDDDDDDDDDDDDDDULL;
        if (fg_fab_send(f, 0, &big_a, sizeof big_a) != FG_FAB_OK) { fprintf(stderr, "rank1 send big_a failed\n"); _exit(1); }
        if (fg_fab_send(f, 0, &big_b, sizeof big_b) != FG_FAB_OK) { fprintf(stderr, "rank1 send big_b failed\n"); _exit(1); }

        fg_fab_barrier(f);
        fg_fab_detach(f);
        _exit(0);
    }

    /* ------------- rank 0: receiver ------------- */
    int err = 0;
    fg_fab_t *f = fg_fab_attach(runid, epoch, 0, nranks, 3000000000ULL, &err);
    if (!f) { fprintf(stderr, "rank0 attach: %s\n", fg_fab_errstr(err)); return 1; }
    if (fg_fab_barrier(f) != FG_FAB_OK) return 1;

    /* First recv offers only 4 bytes while the head message is 8.
     * FIFO says: report ESIZE (or wait), never jump past the head. */
    uint32_t buf_small = 0; size_t n = 0;
    int r1 = fg_fab_recv_max(f, 1, &buf_small, sizeof buf_small, &n);
    if (r1 == FG_FAB_OK) {
        fprintf(stderr, "ORDER VIOLATION: recv(cap=4) delivered %zu bytes (value 0x%08x) while an "
                        "8-byte message was still queued at the head; it should have refused with ESIZE\n",
                n, buf_small);
        rc = 1;
    } else if (r1 != FG_FAB_ESIZE) {
        fprintf(stderr, "UNEXPECTED: recv(cap=4) returned %s, expected ESIZE\n", fg_fab_errstr(r1));
        rc = 1;
    }

    /* Now take the head message with a capacity that fits it. */
    uint64_t got_big = 0; n = 0;
    int r2 = fg_fab_recv_max(f, 1, &got_big, sizeof got_big, &n);
    if (r2 != FG_FAB_OK || n != sizeof got_big || got_big != 0xAAAAAAAAAAAAAAAAULL) {
        fprintf(stderr, "head message wrong: rc=%s n=%zu value=0x%016llx (expected 0xaaaaaaaaaaaaaaaa)\n",
                fg_fab_errstr(r2), n, (unsigned long long)got_big);
        rc = 1;
    }

    /* And then the small one, in order. */
    uint32_t got_small = 0; n = 0;
    int r3 = fg_fab_recv_max(f, 1, &got_small, sizeof got_small, &n);
    if (r3 != FG_FAB_OK || n != sizeof got_small || got_small != 0xBBBBBBBBu) {
        fprintf(stderr, "second message wrong: rc=%s n=%zu value=0x%08x (expected 0xbbbbbbbb)\n",
                fg_fab_errstr(r3), n, got_small);
        rc = 1;
    }

    if (fg_fab_barrier(f) != FG_FAB_OK) return 1;   /* scenario 1 complete on both sides */

    /* ------- scenario 2 (reviewer's strengthening): BOTH queued messages are
     * oversized, so nothing on the queue fits the caller's capacity at all.
     * The contract is still ESIZE, promptly: an oversize head is a reportable
     * error, not a reason to wait for the timeout and not a reason to consume
     * anything.  This also pins that returning ESIZE leaves the head queued,
     * because both messages are recovered afterwards, in order. */
    uint8_t big2_a = 0;
    int r4 = fg_fab_recv_max(f, 1, &big2_a, 4, &n);
    if (r4 == FG_FAB_OK) {
        fprintf(stderr, "ORDER VIOLATION: recv(cap=4) delivered %zu bytes while BOTH queued messages "
                        "were 8 bytes; it should have reported ESIZE\n", n);
        rc = 1;
    } else if (r4 == FG_FAB_ETIMEDOUT) {
        fprintf(stderr, "BLOCKED INSTEAD OF REFUSING: recv(cap=4) waited for the timeout while an "
                        "8-byte head was already queued; an oversize head must report ESIZE immediately "
                        "(blocking is what stalls a polled ncclNet test())\n");
        rc = 1;
    } else if (r4 != FG_FAB_ESIZE) {
        fprintf(stderr, "UNEXPECTED: recv(cap=4) returned %s, expected ESIZE\n", fg_fab_errstr(r4));
        rc = 1;
    }

    /* Both oversized messages must survive the ESIZE and come back in order. */
    uint64_t pair[2] = {0, 0}; n = 0;
    for (int k = 0; k < 2; k++) {
        int rk2 = fg_fab_recv_max(f, 1, &pair[k], sizeof pair[k], &n);
        uint64_t want = k == 0 ? 0xCCCCCCCCCCCCCCCCULL : 0xDDDDDDDDDDDDDDDDULL;
        if (rk2 != FG_FAB_OK || n != sizeof pair[k] || pair[k] != want) {
            fprintf(stderr, "ESIZE CONSUMED THE HEAD or order broke: message %d rc=%s n=%zu got=0x%016llx want=0x%016llx\n",
                    k, fg_fab_errstr(rk2), n, (unsigned long long)pair[k], (unsigned long long)want);
            rc = 1;
        }
    }

    fg_fab_barrier(f);
    fg_fab_detach(f);
    fg_fab_unlink(runid);

    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) { rc = 1; }

    if (rc == 0) printf("fabric fifo: PASS (oversize head refused with ESIZE, messages delivered in order)\n");
    else         printf("fabric fifo: FAIL (see the ORDER VIOLATION / wrong-message lines above)\n");
    return rc;
}
