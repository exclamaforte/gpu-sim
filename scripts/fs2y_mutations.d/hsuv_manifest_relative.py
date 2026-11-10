MUTATIONS = {"hsuv_manifest_relative": "hsuv manifest relative paths (gpu-sim-hsuv)"}

def apply(root):
    """Mutation for hsuv_manifest_relative: write an absolute exe path into the
    committed manifest so the relative-path guard fails."""
    import json, os
    path = os.path.join(root, "experiments", "gpu-sim-u5r4", "build_div_sizescale", "manifest.json")
    backup = open(path).read()
    d = json.loads(backup)
    specs = d.get("specs", [])
    if specs:
        specs[0]["exe"] = "/home/gabe/Documents/gpu-sim/.lab/wt/gpu-sim-hsuv/experiments/gpu-sim-u5r4/build_div_sizescale/bin/k_divergent_branch_0000.out"
    with open(path, "w") as f:
        json.dump(d, f, indent=2)
    return backup, path
