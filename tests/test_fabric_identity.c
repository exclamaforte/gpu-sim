/* test_fabric_identity.c -- pins the two fabric properties that broke field
 * runs and that no other test covered (gpu-sim-ast, gpu-sim-z3a).
 *
 * 1. CHANNEL ISOLATION.  NCCL runs one message stream per channel, and every
 *    channel of a peer shares one mailbox.  Before 3f33f0e the receive side
 *    matched on tag alone, so a request for channel 1 was satisfied by the
 *    oldest message of that tag whatever channel it came from -- a constant
 *    whole-chunk rotation in the result.  This test queues (tag0,chan0) then
 *    (tag0,chan1) and asks for channel 1 FIRST.  A correct fabric returns the
 *    channel-1 payload; a tag-only fabric returns the channel-0 one.
 *
 * 2. ZERO-LENGTH MESSAGES.  At small sizes NCCL issues a 0-byte isend for an
 *    empty chunk.  Before 575a186 the fabric returned FG_FAB_EINVAL for n == 0
 *    (and for cap == 0), which stalled the collective with the sender a few
 *    pieces in and the receiver never posting.  A 0-byte message is a legal
 *    message: it must round-trip and it must keep its place in the stream.
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

#define PAYLOAD_A 0x11111111u   /* (tag 0, chan 0) */
#define PAYLOAD_B 0x22222222u   /* (tag 0, chan 1) */

int main(void) {
    char runid[64];
    snprintf(runid, sizeof runid, "ident-%d-%ld", (int)getpid(), (long)time(NULL));
    const uint64_t epoch = 1;
    const int nranks = 2;

    pid_t pid = fork();
    if (pid < 0) { perror("fork"); return 1; }

    if (pid == 0) {
        /* ---------------- rank 1: sender ---------------- */
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, 1, nranks, 5000000000ULL, &err);
        if (!f) { fprintf(stderr, "rank1 attach: %s\n", fg_fab_errstr(err)); _exit(1); }
        if (fg_fab_barrier(f) != FG_FAB_OK) _exit(1);

        /* (tag 0, chan 0) first, then (tag 0, chan 1): the receiver will ask for
           channel 1 first, so a tag-only matcher hands it the wrong message. */
        if (fg_fab_send_id(f, 0, &(uint32_t){PAYLOAD_A}, sizeof(uint32_t), 0, 0) != FG_FAB_OK) {
            fprintf(stderr, "rank1 send A failed\n"); _exit(1);
        }
        if (fg_fab_send_id(f, 0, &(uint32_t){PAYLOAD_B}, sizeof(uint32_t), 0, 1) != FG_FAB_OK) {
            fprintf(stderr, "rank1 send B failed\n"); _exit(1);
        }
        /* zero-length message, tag 0, channel 2 */
        if (fg_fab_send_id(f, 0, "", 0, 0, 2) != FG_FAB_OK) {
            fprintf(stderr, "rank1 zero-length send failed: %s\n", fg_fab_errstr(FG_FAB_EINVAL)); _exit(1);
        }

        if (fg_fab_barrier(f) != FG_FAB_OK) _exit(1);
        fg_fab_detach(f);
        _exit(0);
    }

    /* ---------------- rank 0: receiver ---------------- */
    int rc = 0;
    int err = 0;
    fg_fab_t *f = fg_fab_attach(runid, epoch, 0, nranks, 5000000000ULL, &err);
    if (!f) { fprintf(stderr, "rank0 attach: %s\n", fg_fab_errstr(err)); return 1; }
    if (fg_fab_barrier(f) != FG_FAB_OK) return 1;

    /* 1. channel isolation: ask for channel 1 while a channel-0 message is queued first */
    uint32_t got = 0; size_t n = 0;
    int r = fg_fab_recv_id_ex(f, 1, &got, sizeof got, 0, 1, &n, NULL, NULL, NULL);
    if (r != FG_FAB_OK || n != sizeof got || got != PAYLOAD_B) {
        fprintf(stderr, "CHANNEL BLEED: request for (tag0,chan1) returned rc=%s n=%zu value=0x%08x "
                        "(expected 0x%08x from chan1; 0x%08x would mean it took the chan0 message)\n",
                fg_fab_errstr(r), n, got, PAYLOAD_B, PAYLOAD_A);
        rc = 1;
    }

    /* 2. the channel-0 message is still there, and still its own payload */
    got = 0; n = 0;
    r = fg_fab_recv_id_ex(f, 1, &got, sizeof got, 0, 0, &n, NULL, NULL, NULL);
    if (r != FG_FAB_OK || got != PAYLOAD_A) {
        fprintf(stderr, "channel 0 message lost or wrong: rc=%s n=%zu value=0x%08x (expected 0x%08x)\n",
                fg_fab_errstr(r), n, got, PAYLOAD_A);
        rc = 1;
    }

    /* 3. zero-length message on channel 2.  A real caller hands a valid buffer
       with capacity 0, not a null pointer, so that is what is tested. */
    uint8_t dummy = 0;
    n = 12345;
    r = fg_fab_recv_id_ex(f, 1, &dummy, 0, 0, 2, &n, NULL, NULL, NULL);
    if (r != FG_FAB_OK || n != 0) {
        fprintf(stderr, "ZERO-LENGTH REJECTED: rc=%s n=%zu (expected FG_FAB_OK and n=0)\n",
                fg_fab_errstr(r), n);
        rc = 1;
    }

    fg_fab_barrier(f);
    fg_fab_detach(f);
    fg_fab_unlink(runid);

    int st = 0;
    waitpid(pid, &st, 0);
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) rc = 1;

    if (rc == 0) printf("fabric identity: PASS (channels isolated, zero-length message delivered)\n");
    else         printf("fabric identity: FAIL (see CHANNEL BLEED / ZERO-LENGTH lines)\n");
    return rc;
}
