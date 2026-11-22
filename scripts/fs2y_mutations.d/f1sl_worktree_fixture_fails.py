"""Mutation for f1sl_worktree_fixture_fails (gpu-sim-f1sl).

Makes the scratch-worktree fixture silently succeed when `git worktree add`
fails: the fixture does nothing instead of raising pytest.fail, then continues
to `yield d`.  That is a mutation of the RULE the guard exists to enforce, not
of a helper.  The bound guard's `next(sandbox.__wrapped__())` therefore returns
normally, and pytest.raises fails with `DID NOT RAISE` -- an assertion-shaped
failure, not StopIteration and not a collection error.
"""
import os

MUTATIONS = {
    "f1sl_worktree_fixture_fails":
    "f1sl: make the broken scratch-worktree fixture return instead of failing"
}


def apply(root):
    path = os.path.join(root, "tests", "test_verify_committed_run.py")
    backup = open(path).read()
    # Replace the WHOLE call, including its two f-string arguments and its
    # closing parenthesis.  Replacing only the first line leaves the arguments
    # dangling and turns the plant into an IndentationError: the module never
    # collects, so "the rule was blinded" and "the module no longer parses"
    # would be indistinguishable (reviewer finding on c3e3f8d0).
    anchor = '''        pytest.fail(
            f"cannot create scratch worktree at {d}: git worktree add "
            f"rc={add.returncode}\\nstdout={add.stdout}\\nstderr={add.stderr}"
        )
'''
    mutated = backup.replace(
        anchor,
        "        pass  # MUTATION: fixture silently succeeds and yields\n",
        1,
    )
    if mutated == backup:
        raise RuntimeError(
            "f1sl_worktree_fixture_fails: fixture pytest.fail anchor not found"
        )
    with open(path, "w") as f:
        f.write(mutated)
    return backup, path
