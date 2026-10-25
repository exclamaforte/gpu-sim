"""Repo-wide pytest configuration (gpu-sim-6gu).

Registers a 'slow' marker and enforces a per-test wall-clock guard so no
test can hang the suite the way test_overfit_111 used to.
"""
import signal

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")


def _handler(signum, frame):
    raise TimeoutError("per-test wall budget exceeded (120s)")


@pytest.fixture(autouse=True)
def _per_test_timeout():
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(120)
    yield
    signal.alarm(0)
