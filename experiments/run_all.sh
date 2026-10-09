#!/usr/bin/env bash
# Runs the whole pipeline. Results are cached per run, so re-running resumes where it stopped.
# CPU defaults take roughly 1.5-2 hours on 4 cores; on a Colab GPU pass --device to E3 via e3_lm.py.
set -euo pipefail
cd "$(dirname "$0")"
python3 e1_mechanisms.py "$@"
python3 e2_fixes.py "$@"
python3 e3_lm.py
python3 analyze.py
