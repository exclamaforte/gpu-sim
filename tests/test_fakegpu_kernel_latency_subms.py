import ctypes
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LIB = REPO / "libfakegpu.so"


def test_submillisecond_kernel_latency_is_active():
    os.environ["FAKEGPU_KERNEL_LATENCY_MS"] = "0.000893"
    # reload by fresh CDLL
    lib = ctypes.CDLL(str(LIB))
    lib.fg_simulate_kernel()
    assert lib.fg_latency_active() == 1
