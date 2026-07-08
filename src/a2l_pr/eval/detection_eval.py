#!/usr/bin/env python3
"""Offline detection evaluation for the gated residual policy.

Answers goal #1 ("can it notice it is failing?") on the model's own training
distribution, with NO simulation: take held-out demos, apply each of the four
perturbations, slide the model's gate over every window, and report:
  - gate AUROC (perturbed-window vs clean-window separability),
  - clean false-positive rate at the deployment threshold,
  - detection rate + latency (steps after perturbation onset until first fire),
  - per-failure-type classification confusion / accuracy on positive windows.

This is fast (seconds) and definitive about detectability, independent of any
recovery/correction question.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

import sys
A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
sys.path.append(str(A2L_PR_ROOT / "src"))

from a2l_pr.learning.residual_data import (  # noqa: E402
    DEFAULT_HDF5, FAILURE_TYPE_TO_ID, generate_records, list_demo_keys,
)
from a2l_pr.eval.residual_runtime import load_gated_residual, predict_window  # noqa: E402

ID_TO_FAILURE = {v: k for k, v in FAILURE_TYPE_TO_ID.items()}


def auroc(labels, scores):
    labels = np.asarray(labels); scores = np.asarray(scores)
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # rank-based Mann-Whitney U
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), dtype=float)
    ranks[order] = np.arange(1, len(order) + 1)
    # handle ties by average rank
    allv = np.concatenate([pos, neg])
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt)); np.add.at(sums, inv, ranks)
    avg = sums / cnt
    ranks = avg[inv]
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def evaluate(checkpoint, hdf5_path, demo_keys, gate_threshold, history, horizon, stride, severity, seed, device):
    model = load_gated_residual(checkpoint, device)
    records, counts = generate_records(demo_keys, hdf5_path, history, horizon, stride, severity, seed, verbose=False)

    labels, scores = [], []
    # per-demo, per-failure-type window streams for latency
    streams = defaultdict(list)  # (demo, failure_type) -> list[(onset_offset, gate_prob)]
    confusion = np.zeros((len(FAILURE_TYPE_TO_ID), len(FAILURE_TYPE_TO_ID)), dtype=int)
    pos_pred_correct = pos_total = 0

    for r in records:
        pred = predict_window(model, r.past_states, r.past_actions, device)
        g = pred["gate_prob"]
        labels.append(int(r.gate_target > 0.5)); scores.append(g)
        if r.bucket == "positive":
            streams[(r.demo_key, r.failure_type)].append((r.onset_offset, g))
            fpred = int(np.argmax(pred["failure_probs"]))
            confusion[r.failure_type, fpred] += 1
            pos_total += 1
            pos_pred_correct += int(fpred == r.failure_type)

    labels = np.array(labels); scores = np.array(scores)
    clean_mask = labels == 0
    pos_mask = labels == 1
    fp_rate = float(np.mean(scores[clean_mask] >= gate_threshold)) if clean_mask.any() else float("nan")
    recall = float(np.mean(scores[pos_mask] >= gate_threshold)) if pos_mask.any() else float("nan")

    # detection latency per (demo,failure): first onset_offset>=0 window with gate>=thr
    per_type_lat = defaultdict(list)
    per_type_detected = defaultdict(lambda: [0, 0])
    for (demo, ftype), windows in streams.items():
        windows = sorted(windows)
        post = [(off, g) for off, g in windows if off >= 0]
        per_type_detected[ftype][1] += 1
        fired = [off for off, g in post if g >= gate_threshold]
        if fired:
            per_type_detected[ftype][0] += 1
            per_type_lat[ftype].append(min(fired))

    per_type = {}
    for ftype in sorted(set(list(per_type_detected.keys()))):
        det, tot = per_type_detected[ftype]
        lat = per_type_lat[ftype]
        # per-class F1 from confusion
        tp = confusion[ftype, ftype]
        fn = confusion[ftype, :].sum() - tp
        fp = confusion[:, ftype].sum() - tp
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_type[ID_TO_FAILURE.get(ftype, str(ftype))] = {
            "detection_rate": det / max(1, tot),
            "median_latency_steps": float(np.median(lat)) if lat else None,
            "mean_latency_steps": float(np.mean(lat)) if lat else None,
            "type_precision": prec, "type_recall": rec, "type_f1": f1,
            "n_demos": tot,
        }

    return {
        "checkpoint": str(checkpoint),
        "n_demos": len(demo_keys),
        "record_counts": counts,
        "gate_threshold": gate_threshold,
        "gate_auroc": auroc(labels, scores),
        "clean_false_positive_rate": fp_rate,
        "positive_recall": recall,
        "failure_type_accuracy_positive": pos_pred_correct / max(1, pos_total),
        "per_failure_type": per_type,
        "confusion_matrix": confusion.tolist(),
        "confusion_labels": [ID_TO_FAILURE[i] for i in range(len(FAILURE_TYPE_TO_ID))],
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--hdf5_path", default=DEFAULT_HDF5)
    p.add_argument("--n_demos", type=int, default=40)
    p.add_argument("--demo_offset", type=int, default=160, help="start index into demo list (held-out by default)")
    p.add_argument("--gate_threshold", type=float, default=0.5)
    p.add_argument("--history", type=int, default=12)
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--stride", type=int, default=6)
    p.add_argument("--severity", type=float, default=0.75)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    keys = list_demo_keys(args.hdf5_path)[args.demo_offset: args.demo_offset + args.n_demos]
    res = evaluate(args.checkpoint, args.hdf5_path, keys, args.gate_threshold,
                   args.history, args.horizon, args.stride, args.severity, args.seed, device)
    print(json.dumps(res, indent=2))
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(res, indent=2))
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
