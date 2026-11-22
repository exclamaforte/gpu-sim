"""Canonical checkpoint hash and cross-rank agreement checker for 7q3w.4.

One definition, used by the training probe and by the ledger-bound test, so the
number the probe reports and the number the test checks cannot diverge:

  * canonical_state_dict_sha256 hashes sorted keys, dtype, shape and CPU tensor
    bytes.  Container metadata is deliberately excluded: the claim is that the
    TENSORS are equal, not that two torch.save files have equal zip metadata.
  * check_checkpoint_hashes returns whether every rank's hash is the same, with
    the distinct values named.  It is fail-able by construction; a synthetic
    disagreement is the test's positive-negative arm.
"""
import hashlib


def canonical_state_dict_sha256(state_dict):
    """SHA-256 over sorted key/dtype/shape/contiguous CPU tensor bytes."""
    h = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        h.update(key.encode())
        h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode())
        h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def check_checkpoint_hashes(hashes):
    """Return {agrees, distinct, values, reason} for one hash per rank.

    The checker must never pass vacuously: an empty hash list is a failure, and
    a disagreement keeps every value so the caller can print it verbatim.
    """
    values = list(hashes)
    if not values:
        return {
            "agrees": False,
            "distinct": [],
            "values": [],
            "reason": "no rank hashes supplied",
        }
    distinct = sorted(set(values))
    agrees = len(distinct) == 1
    return {
        "agrees": agrees,
        "distinct": distinct,
        "values": values,
        "reason": (
            "all ranks agree" if agrees
            else "checkpoint hashes disagree across ranks"
        ),
    }
