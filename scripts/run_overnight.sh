#!/usr/bin/env bash
# Launch the overnight residual-recovery suite under the `a2l` conda env,
# detached, with all output tee'd to the run directory.
#
# Usage:
#   bash scripts/run_overnight.sh            # full run -> output/overnight_<date>/
#   bash scripts/run_overnight.sh --fast     # tiny end-to-end validation
#   bash scripts/run_overnight.sh --force    # re-run all phases
#
# Resume after interruption: just re-run; completed phases (whose outputs exist)
# are skipped automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
STAMP=$(date +%Y%m%d)
OUT="output/overnight_${STAMP}"
mkdir -p "$OUT"
echo "Launching overnight suite -> $OUT (log: $OUT/run.log)"
nohup conda run -n a2l python3 scripts/run_overnight.py "$@" > "$OUT/nohup.out" 2>&1 &
echo "PID $! ; tail -f $OUT/run.log"
