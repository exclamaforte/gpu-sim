MUTATIONS = {"rb66_ambient_init": "rb66 fixture: throwaway repo left on the ambient default branch"}

def apply(root):
    """Mutation for rb66_ambient_init: init the throwaway repo on the WRONG branch.

    The real fix is that the fixture names its branch explicitly, so the
    master/side hazard it builds does not depend on the caller's
    init.defaultBranch. Pointing init at 'main' instead reproduces the bug
    deterministically: the following 'git checkout master' has no such branch and
    the eight tests in the file error out.
    """
    import os
    path = os.path.join(root, "tests", "test_safe_git_add.py")
    backup = open(path).read()
    old = 'git(self.repo, "init", "-q", "-b", "master")'
    new = 'git(self.repo, "init", "-q", "-b", "main")'
    mutated = backup.replace(old, new, 1)
    if mutated == backup:
        raise RuntimeError("rb66 mutation did not apply")
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
