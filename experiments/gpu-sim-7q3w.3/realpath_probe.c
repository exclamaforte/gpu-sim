/* realpath_probe.c -- does the spoof give NCCL what init.cc asks for, and
 * nothing else?  Run with LD_PRELOAD=sysfs_spoof.so FAKEIB_SYSFS=<tree>.
 *
 * The negative half is the part that matters: a prefix rewrite that also
 * catches /sys/class/net or /sys/bus/pci would silently corrupt NCCL's PCI
 * scan, and that failure would look like "no path between GPUs" three layers
 * later.
 */
#define _GNU_SOURCE
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(const char *path, int want_ok, const char *want_prefix) {
    char buf[PATH_MAX];
    char *r = realpath(path, buf);
    if (want_ok && !r) {
        printf("FAIL %s: realpath returned NULL\n", path);
        failures++;
        return;
    }
    if (!want_ok) {
        if (r) { printf("FAIL %s: expected NULL, got %s\n", path, r); failures++; }
        else printf("ok   %s -> NULL (untouched)\n", path);
        return;
    }
    if (want_prefix && strncmp(r, want_prefix, strlen(want_prefix)) != 0) {
        printf("FAIL %s -> %s: want prefix %s\n", path, r, want_prefix);
        failures++;
        return;
    }
    printf("ok   %s -> %s\n", path, r);
}

int main(void) {
    if (!getenv("FAKEIB_SYSFS")) { printf("FAIL FAKEIB_SYSFS unset\n"); return 1; }
    char devp[PATH_MAX], node[PATH_MAX];
    snprintf(devp, sizeof devp, "/sys/class/infiniband/mlx5_0/device");
    snprintf(node, sizeof node, "/sys/class/infiniband/mlx5_0/node_guid");

    /* the one lookup NCCL's IB path makes */
    check(devp, 1, "/sys/");
    /* Stated as a limitation, not as a pass: this interposer rewrites realpath
     * only.  An attribute file in the tree is NOT readable through open(), so if
     * a future NCCL reads /sys/class/infiniband/<dev>/ports/... on the IB path
     * (today it does so on the RoCE path only, connect.cc:279) the shim will
     * appear to have a device with no attributes and NCCL will report
     * "Failed to get RoCE gid type" or similar -- extend sysfs_spoof.c then. */
    FILE *f = fopen(node, "r");
    if (f) { printf("note %s: attribute open() works\n", node); fclose(f); }
    else printf("note %s: attribute open() not intercepted (realpath only), as documented\n", node);
    /* a device that does not exist must still fail, even under the prefix */
    check("/sys/class/infiniband/mlx5_9/device", 0, NULL);
    /* unrelated prefixes are not ours to touch */
    check("/sys/class/net", 1, "/sys/");
    check("/sys/bus/pci/devices", 1, "/sys/");
    check("/sys/class/infiniband/../net", 0, NULL); /* lexical ../ must not escape-prefix */

    if (failures) { printf("REALPATH FAILED: %d failure(s)\n", failures); return 1; }
    printf("REALPATH OK\n");
    return 0;
}
