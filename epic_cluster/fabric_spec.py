"""Canonical fabric spec: one source, three consumers (gpu-sim-3v6d.5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]

KIND_TO_FABRIC = {
    "nvlink": "nvlink4_h100",
    "ib": "ib_400g",
    "pcie": "pcie5_x16",
}


def load_spec(path) -> dict:
    p = Path(path)
    doc = json.loads(p.read_text())
    if not isinstance(doc, dict):
        raise ValueError(f"{p}: spec must be a JSON object")
    if "sources" not in doc or not isinstance(doc["sources"], dict):
        raise ValueError(f"{p}: missing 'sources' provenance block")
    for name, model in (doc.get("models") or {}).items():
        if not isinstance(model, dict) or not model.get("source"):
            raise ValueError(f"{p}: model {name!r} missing provenance 'source'")
    kinds = {l.get("kind") for l in (doc.get("links") or []) if isinstance(l, dict)}
    for kind in kinds:
        key = f"link.{kind}"
        if key not in doc["sources"] or not doc["sources"].get(key):
            raise ValueError(f"{p}: missing provenance for link kind {kind!r}")
    return doc


def classify_links(spec: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for link in spec.get("links") or []:
        out.setdefault(link["kind"], []).append(link)
    return out


def _one_way_bw(link: dict) -> float:
    bw = float(link["bw_gbps"])
    if link.get("bw_is_bidirectional"):
        bw = bw / 2.0
    return bw


def derive_fabrics(spec: dict) -> dict[str, dict[str, float]]:
    classified = classify_links(spec)
    result = {}
    for kind, links in classified.items():
        bws = {_one_way_bw(l) for l in links}
        alphas = {float(l["latency_us"]) / 1e6 for l in links}
        if len(bws) != 1 or len(alphas) != 1:
            raise ValueError(
                f"link kind {kind!r} is heterogeneous (bw={bws}, alpha={alphas})")
        result[kind] = {"alpha_s": alphas.pop(), "bw_gbps": bws.pop()}
    return result


def fabric_name(kind: str) -> str:
    return KIND_TO_FABRIC.get(kind, kind)


def provenance_for(spec: dict, kind: str) -> str:
    return spec["sources"].get(f"link.{kind}", "")
