/* dlvsym_probe.c -- the arm-(a) preflight, on the CPU.
 *
 * NCCL's verbs table is built with dlvsym(handle, name, VERSION), not dlsym
 * (src/misc/ibvsymbols.cc).  The mandatory names are looked up at IBVERBS_1.1
 * and a single NULL tears the whole table down, which makes NCCL fall back to
 * Socket with no error the operator sees unless NCCL_DEBUG is on.  That is
 * exactly what happened on the first GPU run of this bead; this probe is the
 * reason it cannot happen again silently.
 *
 * Mirrors both halves of the source: LOAD_SYM (mandatory, fatal) and
 * LOAD_SYM_VERSION (cherry-picked, silent on failure, but arm (c) needs the
 * dmabuf one, and arm (b) needs query_port_speed to be honest about absence).
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

static int failures;

static const char *mandatory[] = {
    "ibv_get_device_list", "ibv_free_device_list", "ibv_get_device_name",
    "ibv_open_device", "ibv_close_device", "ibv_get_async_event",
    "ibv_ack_async_event", "ibv_query_device", "ibv_query_port",
    "ibv_query_gid", "ibv_query_pkey", "ibv_query_qp", "ibv_alloc_pd",
    "ibv_dealloc_pd", "ibv_reg_mr", "ibv_dereg_mr", "ibv_create_cq",
    "ibv_destroy_cq", "ibv_create_qp", "ibv_modify_qp", "ibv_destroy_qp",
    "ibv_fork_init", "ibv_event_type_str",
};

struct versioned { const char *name, *version, *why; };
static const struct versioned optional[] = {
    {"ibv_reg_mr_iova2",   "IBVERBS_1.8",  "ibv_reg_mr_iova2 (1.8): nice to have"},
    {"ibv_reg_dmabuf_mr",  "IBVERBS_1.12", "ARM (c) needs this: dmabuf registration"},
    {"ibv_query_ece",      "IBVERBS_1.10", "ECE query (1.10)"},
    {"ibv_set_ece",        "IBVERBS_1.10", "ECE set (1.10)"},
    {"ibv_query_port_speed", "IBVERBS_1.16", "port speed (1.16); NCCL falls back to active_speed"},
};

int main(int argc, char **argv) {
    const char *soname = argc > 1 ? argv[1] : "libibverbs.so";
    void *h = dlopen(soname, RTLD_NOW);
    if (!h) { printf("DLVSYM FAILED: dlopen(%s): %s\n", soname, dlerror()); return 1; }

    for (size_t i = 0; i < sizeof mandatory / sizeof *mandatory; i++) {
        dlerror();
        void *p = dlvsym(h, mandatory[i], "IBVERBS_1.1");
        if (!p) {
            printf("DLVSYM FAIL %s@IBVERBS_1.1: %s\n", mandatory[i], dlerror());
            failures++;
        }
    }
    for (size_t i = 0; i < sizeof optional / sizeof *optional; i++) {
        void *p = dlvsym(h, optional[i].name, optional[i].version);
        printf("dlvsym %-22s @%-12s %s  (%s)\n", optional[i].name,
               optional[i].version, p ? "present" : "ABSENT", optional[i].why);
        if (!p && strcmp(optional[i].version, "IBVERBS_1.12") == 0) {
            printf("DLVSYM FAIL %s@%s is required for arm (c)\n",
                   optional[i].name, optional[i].version);
            failures++;
        }
    }
    /* A version that does not exist must be reported as absent, not silently
     * satisfied by the unversioned symbol -- that would mean the map is not
     * doing what we think. */
    if (dlvsym(h, "ibv_reg_dmabuf_mr", "IBVERBS_1.1") != NULL) {
        printf("DLVSYM FAIL ibv_reg_dmabuf_mr resolved at IBVERBS_1.1; the version "
               "script is not assigning versions\n");
        failures++;
    }
    if (failures) { printf("DLVSYM FAILED: %d\n", failures); return 1; }
    printf("DLVSYM OK: %zu mandatory at IBVERBS_1.1, dmabuf at 1.12\n",
           sizeof mandatory / sizeof *mandatory);
    return 0;
}
