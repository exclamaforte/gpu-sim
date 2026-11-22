"""Mutation for f1sl_worktree_fixture_fails (gpu-sim-f1sl).

Makes the scratch-worktree fixture silently succeed when `git worktree add`
fails: the fixture returns instead of raising pytest.fail.  That is a mutation
of the RULE the guard exists to enforce, not of a helper.  The bound guard test
then goes red because pytest.raises sees no pytest.fail to catch (the fixture
returns before yielding), which is exactly the guard going yellow on the
regression it exists for.
"""
import os

MUTATIONS = {
    "f1sl_worktree_fixture_fails":
    "f1sl: make the broken scratch-worktree fixture return instead of failing"
}


def apply(root):
    path = os.path.join(root, "tests", "test_verify_committed_run.py")
    backup = open(path).read()
    anchor = "        pytest.fail(\n"
    mutated = backup.replace(
        anchor,
        "        return  # MUTATION: fixture silently succeeds on worktree failure\n",
        1,
    )
    if mutated == backup:
        raise RuntimeError(
            "f1sl_worktree_fixture_fails: fixture pytest.fail anchor not found"
        )
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
