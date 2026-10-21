#ifndef FAKEGPU_NCCL_PREDICT_H
#define FAKEGPU_NCCL_PREDICT_H

/* gpu-sim-5zv: NCCL prediction contract (C mirror of epic_cluster/link_model.py).

   FG_NCCL_REFUSED is the sentinel returned by fg_nccl_predict when the spec
   cannot be predicted (n>=3 unmeasured, or loss_ppm>=1e6 / p>=1). A duration
   is never negative, so -1.0 is safe as a sentinel. A future API should
   return a status enum rather than overloading a double.

   CROSS-ROUTE MAPPING (gpu-sim-5zv):
     C returns FG_NCCL_REFUSED (-1.0) for exactly the specs where Python raises
     ValueError. The two routes agree on refusing, not just on avoiding a crash.
     - n>=3 (unmeasured pieces-per-phase): Python raises ValueError; C returns FG_NCCL_REFUSED.
     - loss_ppm>=1e6 (p>=1, every piece dropped): Python raises ValueError; C returns FG_NCCL_REFUSED.
*/
#define FG_NCCL_REFUSED (-1.0)

#include "fakegpu_topology.h"

double fg_nccl_predict(const fg_topo_t *t, const char *algo, size_t count,
                       int datatype, int nranks, int *bottleneck_link,
                       char *regime, size_t regimesz);

#endif /* FAKEGPU_NCCL_PREDICT_H */
