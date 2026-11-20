MUTATIONS = {"3s8w_gate_log": "3s8w gate stage report prints ids and copies log (gpu-sim-3s8w)"}

def apply(root):
    import os
    path = os.path.join(root, "scripts", "gate_stage_report.py")
    backup = open(path).read()
    old = '''    if id_lines:
        print("--- failing ids / summary ---")
        for ln in id_lines[-40:]:
            print("    " + ln)
'''
    mutated = backup.replace(old, "    if id_lines:\n        pass  # mutated: drop ids\n", 1)
    if mutated == backup:
        raise RuntimeError("3s8w mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
