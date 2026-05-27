#!/usr/bin/env bash

set -e

THRESHOLDS=(
  0.00
  0.05
  0.10
  0.15
  0.20
  0.25
  0.30
  0.35
  0.40
  0.45
  0.50
  0.55
  0.60
  0.65
  0.70
  0.75
  0.80
  0.85
  0.90
  0.95
  1.00
)

for THRESH in "${THRESHOLDS[@]}"; do
  echo "======================================="
  echo "Running with gate_threshold=${THRESH}"
  echo "======================================="

  python src/a2l_pr/eval/official_policy_evaluation.py \
    --n_rollouts 20 \
    --gate_threshold "${THRESH}" \
    --output_dir "output/official_eval_results/gate_threshold_${THRESH}" \
    --seed 0

  echo
done