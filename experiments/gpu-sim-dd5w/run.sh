#!/usr/bin/env bash
set -euo pipefail
python -m unittest tests.test_7uz2_raw_to_facts tests.test_3fog_raw_to_facts tests.test_37xk_raw_to_facts
