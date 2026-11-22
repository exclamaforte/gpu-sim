/* gpu-sim-7q3w.3: load the shim the way NCCL loads it, and dlsym its surface.
 *
 * NCCL does dlopen("libibverbs.so.1", RTLD_NOW|RTLD_LOCAL) then dlsym on a
 * fixed name list (src/misc/ibvwrap.cc buildIbvSymbols).  Linking the probe
 * against the shim is NOT evidence for that path: a missing DT_SONAME or a
 * symbol that is only link-visible would still pass.  This program takes the
 * SONAME as argv[1] and reports every missing name.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>

/* The 32 names in NCCL v2.32.6-1's own dlsym list (measured from the binary,
 * see the bead), plus the four inline-op helpers NCCL reaches through the
 * context ops table and the two WCs helpers our harness uses. */
static const char *const names[] = {
    "ibv_get_device_list", "ibv_free_device_list", "ibv_get_device_name",
    "ibv_open_device", "ibv_close_device", "ibv_query_device",
    "ibv_query_port", "ibv_query_port_speed", "ibv_query_gid",
    "ibv_query_pkey", "ibv_query_qp", "ibv_alloc_pd", "ibv_dealloc_pd",
    "ibv_reg_mr", "ibv_reg_mr_iova2", "ibv_reg_dmabuf_mr", "ibv_dereg_mr",
    "ibv_create_comp_channel", "ibv_destroy_comp_channel",
    "ibv_create_cq", "ibv_destroy_cq", "ibv_create_qp", "ibv_destroy_qp",
    "ibv_modify_qp", "ibv_create_ah", "ibv_destroy_ah", "ibv_create_srq",
    "ibv_destroy_srq", "ibv_fork_init", "ibv_get_async_event",
    "ibv_ack_async_event", "ibv_event_type_str", "ibv_set_ece",
    "ibv_query_ece",
    /* reached through ctx->ops in NCCL's inline wrappers, exported for our
     * own probe and for a caller that links rather than dlsyms */
    "ibv_post_send", "ibv_post_recv", "ibv_poll_cq", "ibv_req_notify_cq",
};

int main(int argc, char **argv) {
    const char *soname = argc > 1 ? argv[1] : "libibverbs.so.1";
    void *h = dlopen(soname, RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "FAIL dlopen(%s): %s\n", soname, dlerror()); return 1; }
    int missing = 0;
    for (unsigned i = 0; i < sizeof names / sizeof names[0]; i++) {
        if (!dlsym(h, names[i])) { fprintf(stderr, "FAIL missing symbol: %s\n", names[i]); missing++; }
    }
    dlclose(h);
    if (missing) { printf("DLSYM FAILED: %d missing\n", missing); return 1; }
    printf("DLSYM OK: %zu names, soname %s\n", sizeof names / sizeof names[0], soname);
    return 0;
}
