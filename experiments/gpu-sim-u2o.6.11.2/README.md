# gpu-sim-u2o.6.11.2 — inventory regeneration (2026-10-04)

The guard worked as designed: `test_the_generated_rm_lists_every_tracked_path`
failed because master grew `epic_u2o/resource_limits.py` (6.3 merge) under
the committed artifact.

- Ran `python3 experiments/gpu-sim-u2o.6.11/generate_inventory.py` on master
  tip: 59 tracked, 0 not tracked, 0 added by this bead (pure regeneration;
  6.11 itself is merged).
- `tests/test_generated_recipe.py`: 4/4 green.
- The regenerated file lives where the generator lives
  (`experiments/gpu-sim-u2o.6.11/tracks_inventory.md`); this note is the
  bead's own record that the run happened here.
