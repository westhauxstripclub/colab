#!/usr/bin/env bash
# Runs the whole pipeline (round 1 + round 2). Results are cached per run, so re-running resumes where it stopped.
# Roughly 13 h on 4 CPU cores. E5/E6 read the configs frozen by `e4_tuning.py select` (see PREREGISTRATION.md).
set -euo pipefail
cd "$(dirname "$0")"
python3 e1_mechanisms.py --exp e1_mechanisms --seeds 8   # round 1
python3 e2_fixes.py
python3 e3_lm.py --preset round1
python3 e1_mechanisms.py                                 # round 2: 20 seeds + SGD factorial
python3 e4_tuning.py tune
python3 e4_tuning.py select
python3 e4_tuning.py eval
python3 e5_heldout.py
python3 e6_deep.py
python3 e3_lm.py --preset cpu
python3 analyze.py
python3 analyze2.py
