/* gpu-sim-m3d.8 acceptance 3b: prove the simulator's peer-death detection
 * INDEPENDENTLY of NCCL. Two processes attach to the fabric; one sends a
 * message and dies; the survivor's pending transfer must return
 * FG_FAB_EPEERDEAD, and the plugin's error-class mapping must turn that into
 * ncclSystemError (asserted here through the same mapping function the plugin
 * uses, pinned to the numeric value 2 from the real nccl.h). No NCCL, no GPU.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <signal.h>
#include <sys/wait.h>
#include "fakegpu_fabric.h"
#include "nccl_net.h"

/* The plugin's REAL mapping (experiments/gpu-sim-m3d.8/src/ncclnetsim_fabric.c),
 * declared extern so this test links against the plugin's own code rather than
 * a copy of it. */
extern "C" int ncclnetsim_map_fab_error(int rc);

int main(void) {
    const char *runid = "m3d8_peerdeath";
    pid_t pid = fork();
    if (pid == 0) {
        /* child = victim: attach, send one message, then die hard */
        int err = 0;
        fg_fab_t *f = fg_fab_attach(runid, 1, 1, 2, 2000000000ULL, &err);
        if (!f) _exit(2);
        if (fg_fab_barrier(f) != FG_FAB_OK) _exit(3);
        char buf[64]; memset(buf, 'V', sizeof(buf));
        if (fg_fab_send(f, 0, buf, sizeof(buf)) != FG_FAB_OK) _exit(4);
        raise(SIGKILL);
        _exit(0);
    }
    int err = 0;
    fg_fab_t *f = fg_fab_attach(runid, 1, 0, 2, 2000000000ULL, &err);
    if (!f) { fprintf(stderr, "survivor attach failed: %s\n", fg_fab_errstr(err)); return 1; }
    if (fg_fab_barrier(f) != FG_FAB_OK) { fprintf(stderr, "barrier failed\n"); return 1; }
    char buf[64];
    /* drain the victim's message */
    size_t got = 0;
    fg_fab_recv_max(f, 1, buf, sizeof(buf), &got);
    /* now the pending transfer: nothing more will ever arrive, and the peer is dead */
    int rc = fg_fab_recv_max(f, 1, buf, sizeof(buf), &got);
    printf("nccl.h resolved: ncclSystemError = %d (plugin uses the real header, value 2)\n", (int)ncclSystemError);
    int mapped = ncclnetsim_map_fab_error(rc);
    printf("mapping function called: ncclnetsim_map_fab_error (plugin's real code)\n");
    printf("pending transfer rc=%d (%s), mapped to NCCL code %d\n", rc, fg_fab_errstr(rc), mapped);
    int ok = (rc == FG_FAB_EPEERDEAD) && (mapped == ncclSystemError)
            && (mapped == 2) && ((int)ncclSystemError == 2);
    printf("peer-death mapping %s\n", ok ? "PASS" : "FAIL");
    int st = 0; waitpid(pid, &st, 0);
    fg_fab_detach(f); fg_fab_unlink(runid);
    return ok ? 0 : 1;
}
