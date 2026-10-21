#!/usr/bin/env python3
"""Executable assertion for net-plugin-over-interposed (gpu-sim-6kp item 2)."""
import os, sys

EXP = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(EXP, "net_plugin_over_interposed.log")

def test_net_plugin_loaded():
    assert os.path.exists(LOG), LOG
    txt = open(LOG).read()
    assert "NET/Plugin: Loaded net plugin sim" in txt, "plugin not loaded"
    assert "via NET/sim" in txt, "no via NET/sim channel"
    assert "via P2P" not in txt, "P2P transport leaked"
    assert "via SHM" not in txt, "SHM transport leaked"
    assert "via Socket" not in txt and "NET/Socket" not in txt, "Socket transport leaked"
    print("PASS net_plugin_loaded")

if __name__ == "__main__":
    test_net_plugin_loaded()
