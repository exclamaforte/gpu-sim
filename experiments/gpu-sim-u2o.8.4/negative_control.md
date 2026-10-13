# Negative control: matched-size assertion (compare_84.py)

The verdict depends on the matched-size contract (identical encoder,
hparams, split, seed -- representation the only difference). `compare_84.py`
asserts it before scoring. Demonstrated failing run (mixed-scale scratch
dir: 199K source + big ptx/hybrid):

```
$ python3 experiments/gpu-sim-u2o.8.4/compare_84.py /tmp/84neg
AssertionError: arms not matched: [
 {'n_params': 199169, ..., 'd_model': None, 'n_heads': None,
  'n_layers': None, 'd_ff': None, 'n_train': 2494},
 {'n_params': 4936449, ..., 'd_model': 256, 'n_heads': 8,
  'n_layers': 6, 'd_ff': 1024, 'n_train': 2494},
 ...]
```

Green controls on the same code: pure-199K dir and pure-big dir both
compare clean (metrics_199k.json / metrics_big.json). Note the guard was
sharpened once during this work: the first mixed-scale probe died with
`KeyError: 'd_model'` (pre-dims metrics lack the key) instead of the
assertion; the `.get` form above makes the guard itself fire.
