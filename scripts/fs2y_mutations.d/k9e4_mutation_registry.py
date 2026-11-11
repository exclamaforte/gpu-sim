def apply(root):
    """Mutation for k9e4_mutation_registry: remove the loader from fs2y_mutations.py."""
    import os
    path = os.path.join(root, "scripts", "fs2y_mutations.py")
    backup = open(path).read()
    # Remove everything up to and including the for loop that populates MUTATIONS
    marker = 'MUTATIONS = {}\nMUTATION_FUNCTIONS = {}'
    idx = backup.find(marker)
    if idx == -1:
        raise RuntimeError("k9e4 mutation did not apply: marker not found")
    # Keep only from MUTATIONS = {} onwards, but empty
    mutated = 'MUTATIONS = {}\nMUTATION_FUNCTIONS = {}\n'
    if mutated == backup:
        raise RuntimeError("k9e4 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
