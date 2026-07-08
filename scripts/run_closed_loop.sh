#!/usr/bin/env bash
# Run the closed-loop detector experiment (and the deferred diffusion FSM cells) when
# the GPU is FREE. Polls nvidia-smi and only starts once no other process holds the GPU,
# so it never contends with another job (e.g. the lob-mae scaling-laws run).
#
# Usage:
#   nohup bash scripts/run_closed_loop.sh > output/closed_loop/launch.log 2>&1 &
#   tail -f output/closed_loop/launch.log
#
# Env knobs: POLL_SECS (default 120), FREE_MIB (treat GPU as free if usage < this, 400),
#            STABLE_CHECKS (consecutive free polls required, 3).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p output/closed_loop
POLL_SECS="${POLL_SECS:-120}"
FREE_MIB="${FREE_MIB:-400}"
STABLE_CHECKS="${STABLE_CHECKS:-3}"

gpu_used_mib() {
  nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk 'BEGIN{s=0}{s+=$1}END{print s+0}'
}

echo "[$(date '+%F %T')] waiting for GPU to free (poll=${POLL_SECS}s, free<${FREE_MIB}MiB x${STABLE_CHECKS})"
free=0
while :; do
  used=$(gpu_used_mib)
  if [ "$used" -lt "$FREE_MIB" ]; then
    free=$((free+1))
    echo "[$(date '+%F %T')] GPU ~${used}MiB free ($free/$STABLE_CHECKS)"
    [ "$free" -ge "$STABLE_CHECKS" ] && break
  else
    [ "$free" -ne 0 ] && echo "[$(date '+%F %T')] GPU busy again (~${used}MiB), resetting"
    free=0
  fi
  sleep "$POLL_SECS"
done

echo "[$(date '+%F %T')] GPU free — starting closed-loop detector experiment"
conda run -n a2l python3 scripts/run_closed_loop_detector.py --policies bc,diffusion \
  --n_clean 20 --n_injected 10 --epochs 40

echo "[$(date '+%F %T')] closed-loop done — running deferred diffusion FSM oracle cells"
conda run -n a2l python3 src/a2l_pr/eval/fsm_recovery.py --policies diffusion \
  --n_initial_states 30 --trigger oracle --output_dir output/fsm_recovery/oracle_gentle_diff
# Use the CLOSED-LOOP detector (clean-FP ~0.035) — the whole point of this round.
# The earlier harmful FSM used the offline detector (clean-FP 0.24).
conda run -n a2l python3 src/a2l_pr/eval/fsm_recovery.py --policies bc,diffusion \
  --detector_checkpoint notebooks/gated_residual_closed_loop.pth \
  --n_initial_states 30 --trigger detector --gate_threshold 0.85 --trigger_persist 3 \
  --max_recoveries 2 --output_dir output/fsm_recovery/detector_closed_loop

echo "[$(date '+%F %T')] ALL DONE. See output/closed_loop/ and output/fsm_recovery/"
