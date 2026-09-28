# Repo Scaffolding (v1)

Parent: gpu-sim-u2o.5.3

---

## 1. Config / Seeding

- `seed_config.json`: fixed random seed per experiment run (determinism)
- `FAKEGPU_SEED` used for synthetic validation; real GPU measurements do NOT use fakegpu for timing (only structural pre-validation per v1_scope.md)

---

## 2. Dataset / Run Tracking Layout

```
runs/
  <run_id>/
    config.json
    dataset.csv
    measurement_log.json
    beads_jsonl_export
```

Every data-generation / measurement run creates a `bd` bead (`bd create --title="Run <run_id>: ..."`) per epic protocol `gpu-sim-u2o.11.6`.
