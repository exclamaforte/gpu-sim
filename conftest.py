"""Repo-wide pytest configuration (gpu-sim-6gu).

Registers a 'slow' marker and enforces a per-test wall-clock guard so no
test can hang the suite the way test_overfit_111 used to.
"""
import signal
import sys
from pathlib import Path

import pytest

# gpu-sim-x1v: refuse to run tests against a stale libfakegpu.so
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripts.check_libfakegpu_fresh import assert_fresh


def pytest_configure(config):
    assert_fresh()
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")


def _handler(signum, frame):
    raise TimeoutError("per-test wall budget exceeded (120s)")


@pytest.fixture(autouse=True)
def _per_test_timeout():
    signal.signal(signal.SIGALRM, _handler)
    signal.alarm(120)
    yield
    signal.alarm(0)
