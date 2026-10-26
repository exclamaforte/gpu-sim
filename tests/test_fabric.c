#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

#include "fakegpu_fabric.h"

static int run_worker(const char *test, int rank, int nranks, const char *runid, uint64_t epoch) {
    /* basic */
    if (strcmp(test, "basic") == 0) {
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 5000000000ULL, &err);
        if (!f) { fprintf(stderr, "rank %d attach failed: %s\n", rank, fg_fab_errstr(err)); return 1; }
        if (fg_fab_barrier(f) != FG_FAB_OK) { fprintf(stderr, "rank %d barrier1 fail\n", rank); return 1; }

        size_t N = 1 << 20;
        char *buf = malloc(N);
        char *rbuf = malloc(N);
        if (!buf || !rbuf) return 1;

        if (rank == 0) {
            for (size_t i = 0; i < N; i++) buf[i] = (char)(i * 31 + 7);
            if (fg_fab_send(f, 1, buf, N) != FG_FAB_OK) { fprintf(stderr, "r0 send fail\n"); return 1; }
            if (fg_fab_recv(f, 1, rbuf, N) != FG_FAB_OK) { fprintf(stderr, "r0 recv fail\n"); return 1; }
            for (size_t i = 0; i < N; i++) {
                char expected = (char)(i * 17 + 3);
                if (rbuf[i] != expected) { fprintf(stderr, "r0 data mismatch at %zu\n", i); return 1; }
            }
        } else {
            for (size_t i = 0; i < N; i++) buf[i] = (char)(i * 17 + 3);
            if (fg_fab_recv(f, 0, rbuf, N) != FG_FAB_OK) { fprintf(stderr, "r1 recv fail\n"); return 1; }
            for (size_t i = 0; i < N; i++) {
                char expected = (char)(i * 31 + 7);
                if (rbuf[i] != expected) { fprintf(stderr, "r1 data mismatch at %zu\n", i); return 1; }
            }
            if (fg_fab_send(f, 0, buf, N) != FG_FAB_OK) { fprintf(stderr, "r1 send fail\n"); return 1; }
        }
        if (fg_fab_barrier(f) != FG_FAB_OK) { fprintf(stderr, "rank %d barrier2 fail\n", rank); return 1; }
        fg_fab_detach(f);
        if (rank == 0) fg_fab_unlink(runid);
        free(buf); free(rbuf);
        return 0;
    }

    if (strcmp(test, "barrier8") == 0) {
        if (rank == 7) {
            /* straggler: attach but never barrier */
            int err = 0;
            fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 2000000000ULL, &err);
            if (!f) return 1;
            sleep(3);
            fg_fab_detach(f);
            return 0;
        }
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 2000000000ULL, &err);
        if (!f) { fprintf(stderr, "rank %d attach fail\n", rank); return 1; }
        int rc = fg_fab_barrier(f);
        fg_fab_detach(f);
        if (rc != FG_FAB_ETIMEDOUT) {
            fprintf(stderr, "rank %d expected ETIMEDOUT got %s\n", rank, fg_fab_errstr(rc));
            return 1;
        }
        return 0;
    }

    if (strcmp(test, "killpeer") == 0) {
        if (rank == 1) {
            /* die quickly */
            int err = 0;
            fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 2000000000ULL, &err);
            if (!f) return 0; /* still exit */
            /* detach not called; process exits -> peer death */
            (void)f;
            return 0;
        }
        /* rank 0 */
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 2000000000ULL, &err);
        if (!f) { fprintf(stderr, "r0 attach fail\n"); return 1; }
        /* wait for peer to die */
        sleep(1);
        char rbuf[64];
        int rc = fg_fab_recv(f, 1, rbuf, 5);
        fg_fab_detach(f);
        fg_fab_unlink(runid);
        if (rc != FG_FAB_EPEERDEAD) {
            fprintf(stderr, "r0 expected EPEERDEAD on recv from dead peer, got %s\n", fg_fab_errstr(rc));
            return 1;
        }
        fprintf(stderr, "r0 got EPEERDEAD (expected) on recv from dead peer\n");
        return 0;
    }

    if (strcmp(test, "stale") == 0) {
        if (rank != 0) return 0;
        int err = 0;
        fg_fab_t *f1 = fg_fab_attach(runid, 1, 0, 1, 2000000000ULL, &err);
        if (!f1) { fprintf(stderr, "first attach fail: %s\n", fg_fab_errstr(err)); return 1; }
        /* second attach with different epoch should fail ESTALE */
        int err2 = 0;
        fg_fab_t *f2 = fg_fab_attach(runid, 2, 0, 1, 2000000000ULL, &err2);
        if (f2 != NULL) { fprintf(stderr, "stale attach unexpectedly succeeded\n"); fg_fab_detach(f2); return 1; }
        if (err2 != FG_FAB_ESTALE) { fprintf(stderr, "expected ESTALE got %s\n", fg_fab_errstr(err2)); fg_fab_detach(f1); fg_fab_unlink(runid); return 1; }
        fg_fab_detach(f1);
        fg_fab_unlink(runid);
        return 0;
    }

    if (strcmp(test, "size") == 0) {
        if (rank == 0) {
            int err = 0;
            fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 5000000000ULL, &err);
            if (!f) return 1;
            if (fg_fab_barrier(f) != FG_FAB_OK) return 1;
            char buf[128];
            memset(buf, 'A', 100);
            if (fg_fab_send(f, 1, buf, 100) != FG_FAB_OK) { fprintf(stderr, "r0 send fail\n"); return 1; }
            /* second message */
            if (fg_fab_send(f, 1, buf, 100) != FG_FAB_OK) { fprintf(stderr, "r0 send2 fail\n"); return 1; }
            fg_fab_barrier(f);
            fg_fab_detach(f);
            fg_fab_unlink(runid);
            return 0;
        } else {
            int err = 0;
            fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 5000000000ULL, &err);
            if (!f) return 1;
            if (fg_fab_barrier(f) != FG_FAB_OK) return 1;
            char rbuf[128];
            /* wrong size: expect ESIZE */
            int rc = fg_fab_recv(f, 0, rbuf, 50);
            if (rc != FG_FAB_ESIZE) { fprintf(stderr, "r1 expected ESIZE got %s\n", fg_fab_errstr(rc)); return 1; }
            /* second message: correct size */
            if (fg_fab_barrier(f) != FG_FAB_OK) { /* sync not needed */ }
            /* wait for second send: use recv with timeout */
            /* second send happens after first recv consumed; but sender sends both before barrier? Actually sender sends two back-to-back. First recv consumed first. Need recv second. */
            /* We already consumed first (ready cleared on ESIZE). Wait for second. */
            /* busy wait a bit */
            int tries = 0;
            while (tries < 1000) {
                int rc2 = fg_fab_recv(f, 0, rbuf, 100);
                if (rc2 == FG_FAB_OK) break;
                if (rc2 == FG_FAB_ESIZE) { fprintf(stderr, "unexpected ESIZE on second\n"); return 1; }
                /* ETIMEDOUT/EINVAL -> retry? ready may not be set yet */
                usleep(1000);
                tries++;
            }
            fg_fab_barrier(f);
            fg_fab_detach(f);
            return 0;
        }
    }

    if (strcmp(test, "reg") == 0) {
        /* gpu-sim-mp7r (ported from gpu-sim-7vw): the registered-buffer path.
           reg_mr returns off+1 so 0 means failure; the sender passes off-1.
           Adapted: the 7vw version carried a chan argument that gpu-sim-ast
           superseded with msg_tag/msg_chan, so the chan parameters are gone. */
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 5000000000ULL, &err);
        if (!f) { fprintf(stderr, "rank %d attach failed: %s\n", rank, fg_fab_errstr(err)); return 1; }
        if (fg_fab_barrier(f) != FG_FAB_OK) { fprintf(stderr, "rank %d barrier1 fail\n", rank); return 1; }

        size_t N = 4096;
        char *buf = calloc(N, 1);
        char *rbuf = calloc(N, 1);
        if (!buf || !rbuf) return 1;
        for (size_t i = 0; i < N; i++) buf[i] = (char)(i * 37 + 13);

        if (rank == 0) {
            /* production never sends pool offset 0 via the reg path (the
               plugin falls back to an inline send when reg_off <= 1), so
               occupy offset 0 first and register the real payload after it. */
            char dummy[8] = {0};
            if (fg_fab_reg_mr(f, dummy, sizeof dummy) == 0) { fprintf(stderr, "r0 dummy reg failed\n"); return 1; }
            size_t off = fg_fab_reg_mr(f, buf, N);
            if (off <= 1) { fprintf(stderr, "r0 reg_mr failed\n"); return 1; }
            /* plugin logic: reg_off = (size_t)mhandle; send_reg(peer, reg_off-1) */
            int slot = -1; unsigned seq = 0;
            if (fg_fab_send_reg(f, 1, off - 1, N, &slot, &seq) != FG_FAB_OK) {
                fprintf(stderr, "r0 send_reg fail\n"); return 1;
            }
        } else {
            /* registered arrivals resolve via the id path (any tag/chan),
               the same call the plugin uses; plain recv_max_ex would hand
               back the slot's offset bytes instead of the pool payload. */
            size_t got = 0; int slot = -1; unsigned seq = 0, consumed = 0;
            int rc = fg_fab_recv_id_ex(f, 0, rbuf, N, -1, -1, &got, &slot, &seq, &consumed);
            if (rc != FG_FAB_OK) { fprintf(stderr, "r1 recv fail: %s\n", fg_fab_errstr(rc)); return 1; }
            if (got != N) { fprintf(stderr, "r1 got %zu != %zu\n", got, N); return 1; }
            if (memcmp(rbuf, buf, N) != 0) {
                for (size_t i = 0; i < N; i++) {
                    if (rbuf[i] != buf[i]) { fprintf(stderr, "r1 MISMATCH at %zu: got %d expect %d\n", i, (unsigned char)rbuf[i], (unsigned char)buf[i]); break; }
                }
                return 1;
            }
        }
        if (fg_fab_barrier(f) != FG_FAB_OK) { fprintf(stderr, "rank %d barrier2 fail\n", rank); return 1; }
        fg_fab_detach(f);
        if (rank == 0) fg_fab_unlink(runid);
        free(buf); free(rbuf);
        return 0;
    }

    if (strcmp(test, "slowstart") == 0) {
        /* gpu-sim-mp7r: a peer that attaches LATE must be waited for, not
           reported dead. Rank 0 is already inside recv while rank 1 has not
           attached yet; without the attached==0 rule peer_alive reads pid 0
           as dead and recv fails fast with EPEERDEAD instead of delivering
           rank 1's message. This FAILS on the pre-fix fabric. */
        if (rank == 1) {
            sleep(2); /* attach late: rank 0 is already waiting */
            int err = 0;
            fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 5000000000ULL, &err);
            if (!f) { fprintf(stderr, "r1 late attach failed: %s\n", fg_fab_errstr(err)); return 1; }
            char buf[64];
            memset(buf, 'S', sizeof buf);
            if (fg_fab_send(f, 0, buf, sizeof buf) != FG_FAB_OK) { fprintf(stderr, "r1 send fail\n"); return 1; }
            fg_fab_detach(f);
            return 0;
        }
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 10000000000ULL, &err);
        if (!f) { fprintf(stderr, "r0 attach fail\n"); return 1; }
        char rbuf[64];
        int rc = fg_fab_recv(f, 1, rbuf, sizeof rbuf);
        if (rc != FG_FAB_OK) { fprintf(stderr, "r0 expected late peer's message, got %s\n", fg_fab_errstr(rc)); fg_fab_detach(f); return 1; }
        for (size_t i = 0; i < sizeof rbuf; i++) {
            if (rbuf[i] != 'S') { fprintf(stderr, "r0 payload corrupt at %zu\n", i); fg_fab_detach(f); return 1; }
        }
        fg_fab_detach(f);
        fg_fab_unlink(runid);
        return 0;
    }

    if (strcmp(test, "neverstart") == 0) {
        /* gpu-sim-mp7r negative control: a peer that NEVER attaches must still
           surface as an error WITHIN the timeout -- the waiter must not hang
           forever. Rank 1 exits without attaching; rank 0's recv must return
           non-OK (ETIMEDOUT where attach failure waits, EPEERDEAD where it
           still reads dead) well before the wall-clock guard. The alarm makes
           a true hang FAIL instead of hanging the suite. */
        if (rank == 1) return 0; /* never attaches */
        alarm(20);
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, epoch, rank, nranks, 3000000000ULL, &err);
        if (!f) { fprintf(stderr, "r0 attach fail\n"); return 1; }
        char rbuf[64];
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        int rc = fg_fab_recv(f, 1, rbuf, sizeof rbuf);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double dt = (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) / 1e9;
        fg_fab_detach(f);
        fg_fab_unlink(runid);
        if (rc == FG_FAB_OK) { fprintf(stderr, "r0 recv from never-attached peer unexpectedly succeeded\n"); return 1; }
        if (dt > 8.0) { fprintf(stderr, "r0 recv took %.1fs: waiter hung past the 3s timeout\n", dt); return 1; }
        fprintf(stderr, "r0 got %s in %.1fs (expected, within timeout)\n", fg_fab_errstr(rc), dt);
        return 0;
    }

    fprintf(stderr, "unknown test %s\n", test);
    return 1;
}

static int fork_workers(const char *test, int nranks, const char *runid, uint64_t epoch) {
    pid_t pids[16];
    for (int r = 0; r < nranks; r++) {
        pid_t pid = fork();
        if (pid < 0) { perror("fork"); return 1; }
        if (pid == 0) {
            char rbuf[16], nbuf[16], ebuf[32];
            snprintf(rbuf, sizeof(rbuf), "%d", r);
            snprintf(nbuf, sizeof(nbuf), "%d", nranks);
            snprintf(ebuf, sizeof(ebuf), "%llu", (unsigned long long)epoch);
            execlp("./build/test_fabric", "test_fabric", "--worker", test, rbuf, nbuf, runid, ebuf, (char*)NULL);
            /* if exec fails, try relative path */
            execl("./build/test_fabric", "test_fabric", "--worker", test, rbuf, nbuf, runid, ebuf, (char*)NULL);
            perror("exec");
            _exit(127);
        }
        pids[r] = pid;
    }
    int fail = 0;
    for (int r = 0; r < nranks; r++) {
        int status = 0;
        waitpid(pids[r], &status, 0);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            fprintf(stderr, "worker %d failed (status=%d)\n", r, status);
            fail = 1;
        }
    }
    return fail;
}

int main(int argc, char **argv) {
    if (argc >= 7 && strcmp(argv[1], "--worker") == 0) {
        const char *test = argv[2];
        int rank = atoi(argv[3]);
        int nranks = atoi(argv[4]);
        const char *runid = argv[5];
        uint64_t epoch = strtoull(argv[6], NULL, 10);
        return run_worker(test, rank, nranks, runid, epoch);
    }

    /* driver mode */
    if (argc < 2) {
        fprintf(stderr, "usage: %s <basic|barrier8|killpeer|stale|size|reg|slowstart|neverstart>\n", argv[0]);
        return 2;
    }
    const char *test = argv[1];
    int nranks = 2;
    if (strcmp(test, "barrier8") == 0) nranks = 8;
    if (strcmp(test, "stale") == 0) nranks = 1;
    char runid[64];
    snprintf(runid, sizeof(runid), "tf_%s_%d", test, (int)getpid());

    /* stale test runs single process, no fork */
    if (strcmp(test, "stale") == 0) {
        int rc = run_worker(test, 0, 1, runid, 1);
        if (rc == 0) printf("TEST stale: PASS\n");
        else printf("TEST stale: FAIL\n");
        return rc;
    }
    int rc = fork_workers(test, nranks, runid, 1);
    if (rc == 0) printf("TEST %s: PASS\n", test);
    else printf("TEST %s: FAIL\n", test);
    return rc;
}