#!/usr/bin/env python3
"""Closed-loop detector experiment: collect live-rollout data, train a detector on it,
and compare against the offline-trained detector on a held-out CLOSED-LOOP set.

Hypothesis: an offline-trained detector has a high false-positive rate on the live
policy's NORMAL windows; a detector trained on closed-loop data should drive that down
while keeping injected-failure recall, fixing the spurious-trigger problem that made
FSM/residual recovery harmful.

GPU-aware: collection drives the simulator (diffusion is GPU-heavy). By default this
ABORTS if another process is already using the GPU (e.g. a separate training job), so it
never steals compute. Override with --force. Collection is cached to .npz so training/
eval can rerun cheaply.

Usage:
  python scripts/run_closed_loop_detector.py                 # full (guards GPU)
  python scripts/run_closed_loop_detector.py --policies bc   # bc-only (GPU-light)
  python scripts/run_closed_loop_detector.py --force         # ignore GPU guard
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
sys.path.append(str(A2L_PR_ROOT / "src"))
sys.path.append(str(A2L_PR_ROOT.parent / "robomimic" / "robomimic"))

from a2l_pr.learning.closed_loop_data import collect_records, save_records, load_records, split_records  # noqa: E402
from a2l_pr.learning.residual_training import TrainConfig, train_gated_residual  # noqa: E402
from a2l_pr.eval.residual_runtime import load_gated_residual, predict_window  # noqa: E402
from a2l_pr.eval.detection_eval import auroc, ID_TO_FAILURE  # noqa: E402
from a2l_pr.learning.residual_data import FAILURE_TYPE_TO_ID  # noqa: E402


def gpu_busy(min_mib=500):
    """Return (busy, info). True if another process holds > min_mib on the GPU."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
            text=True).strip()
    except Exception:
        return False, "nvidia-smi unavailable"
    rows = [r for r in out.splitlines() if r.strip()]
    mine = set()
    heavy = [r for r in rows if int(r.split(",")[1]) > min_mib]
    return (len(heavy) > 0, "; ".join(rows) if rows else "idle")


def evaluate_on_records(checkpoint, records, device, gate_threshold=0.5):
    model = load_gated_residual(checkpoint, device)
    labels, scores = [], []
    clean_scores, pos_total, fail_correct = [], 0, 0
    nft = len(FAILURE_TYPE_TO_ID)
    confusion = np.zeros((nft, nft), dtype=int)
    for r in records:
        pred = predict_window(model, r.past_states, r.past_actions, device)
        g = pred["gate_prob"]
        lab = int(r.gate_target > 0.5)
        labels.append(lab); scores.append(g)
        if r.bucket == "clean":
            clean_scores.append(g)
        if lab == 1:
            fp = int(np.argmax(pred["failure_probs"]))
            confusion[r.failure_type, fp] += 1
            pos_total += 1; fail_correct += int(fp == r.failure_type)
    labels = np.array(labels); scores = np.array(scores)
    return {
        "checkpoint": str(checkpoint),
        "gate_auroc": auroc(labels, scores),
        "clean_false_positive_rate_live": float(np.mean(np.array(clean_scores) >= gate_threshold)) if clean_scores else None,
        "positive_recall": float(np.mean(scores[labels == 1] >= gate_threshold)) if (labels == 1).any() else None,
        "failure_type_accuracy_positive": fail_correct / max(1, pos_total),
        "n_records": len(records), "n_clean_live": len(clean_scores), "n_positive": pos_total,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", default="bc,diffusion")
    p.add_argument("--n_clean", type=int, default=20)
    p.add_argument("--n_injected", type=int, default=10)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--offline_checkpoint", default=str(A2L_PR_ROOT / "notebooks" / "gated_residual_severity_norm.pth"))
    p.add_argument("--out_checkpoint", default=str(A2L_PR_ROOT / "notebooks" / "gated_residual_closed_loop.pth"))
    p.add_argument("--records_npz", default=str(A2L_PR_ROOT / "output" / "closed_loop" / "cl_records.npz"))
    p.add_argument("--output_dir", default=str(A2L_PR_ROOT / "output" / "closed_loop"))
    p.add_argument("--force", action="store_true", help="run even if the GPU is busy")
    args = p.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. collect (or load cached) closed-loop records
    npz = Path(args.records_npz)
    if npz.exists():
        print(f"[cl] loading cached records {npz}")
        records = load_records(npz)
    else:
        busy, info = gpu_busy()
        if busy and not args.force:
            print(f"[cl] ABORT: GPU busy ({info}). Collection drives the sim and would "
                  f"contend. Re-run when free, or pass --force / --policies bc.")
            return
        print(f"[cl] collecting live rollouts (GPU: {info})...")
        records = collect_records(args.policies.split(","), n_clean=args.n_clean,
                                  n_injected=args.n_injected, horizon=args.horizon,
                                  seed=args.seed, stride=args.stride, device=device)
        npz.parent.mkdir(parents=True, exist_ok=True)
        save_records(records, npz)
        print(f"[cl] saved {len(records)} records -> {npz}")

    train_recs, val_recs = split_records(records, val_frac=0.2, seed=args.seed)
    pos = sum(int(r.gate_target > 0.5) for r in records)
    print(f"[cl] {len(records)} records ({pos} pos / {len(records)-pos} neg); "
          f"train={len(train_recs)} val={len(val_recs)}")

    # 2. train detector on closed-loop records
    cfg = TrainConfig(out_path=args.out_checkpoint, epochs=args.epochs, normalize=True,
                      predict_severity=True, seed=args.seed, label="closed_loop")
    train_gated_residual(cfg, train_records=train_recs, val_records=val_recs)

    # 3. compare offline vs closed-loop detector on the SAME held-out closed-loop val set
    result = {
        "offline_detector": evaluate_on_records(args.offline_checkpoint, val_recs, device),
        "closed_loop_detector": evaluate_on_records(args.out_checkpoint, val_recs, device),
        "val_set": "held-out closed-loop rollouts (live policy states)",
    }
    (out_dir / "closed_loop_detector_comparison.json").write_text(json.dumps(result, indent=2))
    print("\n=== Closed-loop detection comparison (held-out live rollouts) ===")
    for name in ["offline_detector", "closed_loop_detector"]:
        r = result[name]
        print(f"{name:22s} AUROC={r['gate_auroc']:.3f}  "
              f"clean_FP_live={r['clean_false_positive_rate_live']}  "
              f"recall={r['positive_recall']}  type_acc={r['failure_type_accuracy_positive']:.3f}")
    print(f"wrote {out_dir}/closed_loop_detector_comparison.json")


if __name__ == "__main__":
    main()
