/* sysfs_spoof.c -- make /sys/class/infiniband/<dev> exist, without root.
 *
 * NCCL's IB transport needs exactly ONE sysfs fact.  src/transport/net_ib/
 * init.cc:73 (v2.32.6-1, sha256 of the file recorded in provenance.md):
 *
 *     snprintf(devicePath, PATH_MAX, "/sys/class/infiniband/%s/device", devName);
 *     char* p = realpath(devicePath, NULL);
 *
 * and then it truncates the last character of that path to '0' to merge the
 * ports of one PCI device.  Everything else the IB path reads comes from the
 * verbs calls our shim answers (the /sys/.../gid_attrs/types read in
 * connect.cc:279 is on the RoCE code path only).
 *
 * The bead allows two routes for the fake sysfs: a mount namespace, or
 * LD_PRELOAD.  The mount namespace is a measured dead end on this box --
 * `unshare -m --map-root-user` can create a namespace, but /sys/class has no
 * `infiniband` entry to bind over, cannot be mkdir'd, and an overlay over /sys
 * is refused -- so this is the LD_PRELOAD route, and it is 30 lines because
 * the surface is one function and one prefix.
 *
 * FAKEIB_SYSFS points at a directory built by fake_sysfs.sh holding
 *   <root>/infiniband/<dev>/device -> <real sysfs path of a PCI bridge>
 * We rewrite the prefix and hand the rest to the real realpath, so the path
 * NCCL gets back is a REAL path under /sys/devices/... (the symlink target
 * resolves normally), which is what its last-character truncation expects.
 *
 * Deliberately not intercepted: stat/lstat/opendir on /sys/class/infiniband.
 * libibverbs' own device list is synthesized by our shim from
 * FAKEIB_NRANKS, never from sysfs, and NCCL does not stat the directory.
 * If a future NCCL starts to, this is the file to extend -- and the probe
 * below will not catch it, so read the NCCL version's init.cc again.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SYSFS_IB "/sys/class/infiniband/"

static char *(*real_realpath)(const char *, char *);

static const char *fake_root(void) {
    static char root[PATH_MAX];
    static int ready;
    if (!ready) {
        const char *e = getenv("FAKEIB_SYSFS");
        if (e && e[0]) snprintf(root, sizeof root, "%s", e);
        ready = 1;
    }
    return root;
}

char *realpath(const char *path, char *resolved) {
    if (!real_realpath) {
        *(void **)(&real_realpath) = dlsym(RTLD_NEXT, "realpath");
        if (!real_realpath) { errno = ENOSYS; return NULL; }
    }
    if (path && fake_root()[0] && strncmp(path, SYSFS_IB, sizeof SYSFS_IB - 1) == 0) {
        /* /sys/class/infiniband/<rest> -> <root>/infiniband/<rest> */
        char alt[PATH_MAX];
        int n = snprintf(alt, sizeof alt, "%s/infiniband/%s", fake_root(),
                         path + sizeof SYSFS_IB - 1);
        if (n > 0 && (size_t)n < sizeof alt) return real_realpath(alt, resolved);
        errno = ENAMETOOLONG;
        return NULL;
    }
    return real_realpath(path, resolved);
}
