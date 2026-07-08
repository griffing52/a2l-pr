#!/usr/bin/env python3
"""Train the GatedResidualRecoveryPolicy from offline perturbation records.

Importable as `train_gated_residual(cfg)` and runnable as a CLI. This replaces
the ad-hoc notebook cells so overnight runs are reproducible.

Improvements over the original 3-epoch / 20-demo notebook run:
- configurable demo count and epochs (defaults use the full square/ph split),
- optional input normalization (state/action history) with stats saved into the
  checkpoint and reapplied identically at eval time,
- held-out validation metrics (gate AUROC-free proxies + failure accuracy),
- residual targets kept in RAW action units so the eval-time application
  (base_action + residual) is unchanged.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys
A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
sys.path.append(str(A2L_PR_ROOT / "src"))

from a2l_pr.learning.residual_data import (  # noqa: E402
    DEFAULT_HDF5, FAILURE_TYPE_TO_ID, NUM_FAILURE_TYPES,
    NormStats, ResidualRecord, compute_norm_stats, generate_records, list_demo_keys,
)
from a2l_pr.models import GatedResidualRecoveryPolicy  # noqa: E402


@dataclass
class TrainConfig:
    hdf5_path: str = DEFAULT_HDF5
    out_path: str = str(A2L_PR_ROOT / "notebooks" / "gated_residual_retrained.pth")
    history: int = 12
    horizon: int = 30
    stride: int = 6
    severity: float = 0.75
    max_train_demos: int = 160
    max_val_demos: int = 40
    epochs: int = 40
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    normalize: bool = True
    failure_loss_weight: float = 0.5
    predict_severity: bool = False
    severity_loss_weight: float = 1.0
    severity_min: float = 0.25
    severity_max: float = 1.0
    seed: int = 0
    label: str = "retrained"


class _RecordDataset(Dataset):
    def __init__(self, records: List[ResidualRecord], norm: Optional[NormStats]):
        self.records = records
        self.norm = norm

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        ps = r.past_states.astype(np.float32)
        pa = r.past_actions.astype(np.float32)
        if self.norm is not None:
            ps = (ps - self.norm.state_mean) / self.norm.state_std
            pa = (pa - self.norm.action_mean) / self.norm.action_std
        return {
            "past_states": torch.from_numpy(ps),
            "past_actions": torch.from_numpy(pa),
            "residual_target": torch.from_numpy(r.residual_target.astype(np.float32)),
            "gate_target": torch.tensor(r.gate_target, dtype=torch.float32),
            "failure_type": torch.tensor(r.failure_type, dtype=torch.long),
            "severity": torch.tensor(r.severity, dtype=torch.float32),
        }


def _epoch_metrics(model, loader, device, horizon) -> Dict[str, float]:
    model.eval()
    gate_correct = gate_total = 0
    clean_total = clean_fp = 0
    pos_total = fail_correct = 0
    res_se = res_n = 0.0
    sev_ae = sev_n = 0.0
    with torch.no_grad():
        for b in loader:
            ps = b["past_states"].to(device); pa = b["past_actions"].to(device)
            gt = b["gate_target"].to(device); ft = b["failure_type"].to(device)
            tgt = b["residual_target"].to(device)
            out = model(ps, pa, prediction_horizon=horizon)
            gate_prob = torch.sigmoid(out["gate_logits"][:, 0])
            pred = (gate_prob >= 0.5).float()
            gate_correct += (pred == gt).sum().item(); gate_total += gt.numel()
            clean_mask = gt < 0.5
            clean_total += clean_mask.sum().item()
            clean_fp += (pred[clean_mask] > 0.5).sum().item()
            pos_mask = gt > 0.5
            if pos_mask.any():
                fl = out["failure_logits"][:, 0, :]
                fp = fl.argmax(-1)
                fail_correct += (fp[pos_mask] == ft[pos_mask]).sum().item()
                pos_total += pos_mask.sum().item()
                res = out["residuals"][pos_mask]
                res_se += ((res - tgt[pos_mask]) ** 2).mean(-1).sum().item()
                res_n += pos_mask.sum().item()
                if "severity_logits" in out:
                    sev = b["severity"].to(device)
                    sp = torch.sigmoid(out["severity_logits"][:, 0])
                    sev_ae += (sp[pos_mask] - sev[pos_mask]).abs().sum().item()
                    sev_n += pos_mask.sum().item()
    return {
        "gate_acc": gate_correct / max(1, gate_total),
        "clean_false_positive_rate": clean_fp / max(1, clean_total),
        "failure_acc_positive": fail_correct / max(1, pos_total),
        "residual_mse_positive": res_se / max(1.0, res_n),
        "severity_mae_positive": sev_ae / max(1.0, sev_n) if sev_n else None,
    }


def _count_buckets(records):
    counts = {}
    for r in records:
        counts[r.bucket] = counts.get(r.bucket, 0) + 1
    return counts


def train_gated_residual(cfg: TrainConfig, log=print, train_records=None, val_records=None) -> Dict:
    """Train the detector. If train_records/val_records are provided (e.g. collected from
    closed-loop rollouts), they are used directly; otherwise records are generated from
    offline demo edits via generate_records."""
    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if train_records is None:
        demo_keys = list_demo_keys(cfg.hdf5_path)
        train_keys = demo_keys[: cfg.max_train_demos]
        val_keys = demo_keys[cfg.max_train_demos: cfg.max_train_demos + cfg.max_val_demos]
        log(f"[train] {len(train_keys)} train demos, {len(val_keys)} val demos")
        sev_range = (cfg.severity_min, cfg.severity_max) if cfg.predict_severity else None
        train_records, train_counts = generate_records(
            train_keys, cfg.hdf5_path, cfg.history, cfg.horizon, cfg.stride, cfg.severity, cfg.seed,
            severity_range=sev_range)
        val_records, val_counts = generate_records(
            val_keys, cfg.hdf5_path, cfg.history, cfg.horizon, cfg.stride, cfg.severity, cfg.seed + 1,
            severity_range=sev_range)
    else:
        train_counts = _count_buckets(train_records)
        val_counts = _count_buckets(val_records) if val_records else {}
        log(f"[train] using {len(train_records)} prebuilt train records, "
            f"{len(val_records) if val_records else 0} val records")

    norm = compute_norm_stats(train_records) if cfg.normalize else None
    state_dim = train_records[0].past_states.shape[-1]
    action_dim = train_records[0].past_actions.shape[-1]

    train_loader = DataLoader(_RecordDataset(train_records, norm), batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(_RecordDataset(val_records, norm), batch_size=cfg.batch_size, shuffle=False)

    model = GatedResidualRecoveryPolicy(
        state_dim=state_dim, action_dim=action_dim,
        history_length=cfg.history, prediction_horizon=cfg.horizon,
        num_failure_types=NUM_FAILURE_TYPES, predict_severity=cfg.predict_severity).to(device)

    num_pos = sum(int(r.gate_target > 0.5) for r in train_records)
    num_neg = max(1, len(train_records) - num_pos)
    pos_weight = torch.tensor([num_neg / max(1, num_pos)], dtype=torch.float32, device=device)
    log(f"[train] gate pos={num_pos} neg={num_neg} pos_weight={float(pos_weight):.2f}")

    crit_res = torch.nn.SmoothL1Loss(reduction="none")
    crit_gate = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    crit_fail = torch.nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    history = []
    best_val = None
    best_state = None
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for b in train_loader:
            ps = b["past_states"].to(device); pa = b["past_actions"].to(device)
            gt = b["gate_target"].to(device); ft = b["failure_type"].to(device)
            tgt = b["residual_target"].to(device)
            out = model(ps, pa, prediction_horizon=cfg.horizon)
            gate_loss = crit_gate(out["gate_logits"][:, 0], gt)
            fail_loss = crit_fail(out["failure_logits"][:, 0, :], ft)
            pos_mask = gt > 0.5
            res_full = crit_res(out["residuals"], tgt).mean(-1)
            res_loss = res_full[pos_mask].mean() if pos_mask.any() else res_full.mean() * 0.0
            loss = res_loss + gate_loss + cfg.failure_loss_weight * fail_loss
            if cfg.predict_severity:
                sev = b["severity"].to(device)
                sev_pred = torch.sigmoid(out["severity_logits"][:, 0])
                sev_loss = ((sev_pred[pos_mask] - sev[pos_mask]) ** 2).mean() if pos_mask.any() else sev_pred.mean() * 0.0
                loss = loss + cfg.severity_loss_weight * sev_loss
            opt.zero_grad(); loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            running += loss.item()
        tr_loss = running / max(1, len(train_loader))
        val_m = _epoch_metrics(model, val_loader, device, cfg.horizon)
        rec = {"epoch": epoch, "train_loss": tr_loss, **{f"val_{k}": v for k, v in val_m.items()}}
        history.append(rec)
        log(f"[train] ep{epoch:02d} loss={tr_loss:.4f} val_gate_acc={val_m['gate_acc']:.3f} "
            f"val_clean_fp={val_m['clean_false_positive_rate']:.3f} "
            f"val_fail_acc={val_m['failure_acc_positive']:.3f} "
            f"val_res_mse={val_m['residual_mse_positive']:.4f}")
        # model-selection score: low clean FP, high gate acc, high failure acc
        score = val_m["gate_acc"] + val_m["failure_acc_positive"] - val_m["clean_false_positive_rate"]
        if best_val is None or score > best_val:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_class": "GatedResidualRecoveryPolicy",
        "state_dim": state_dim,
        "action_dim": action_dim,
        "history_length": cfg.history,
        "prediction_horizon": cfg.horizon,
        "num_failure_types": NUM_FAILURE_TYPES,
        "failure_type_to_id": FAILURE_TYPE_TO_ID,
        "predict_severity": cfg.predict_severity,
        "training_history": history,
        "config": asdict(cfg),
        "train_counts": train_counts,
        "val_counts": val_counts,
    }
    if norm is not None:
        ckpt["norm_stats"] = norm.to_ckpt()
    Path(cfg.out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, cfg.out_path)
    log(f"[train] saved {cfg.out_path} (best score={best_val:.3f})")
    return {"out_path": cfg.out_path, "history": history, "best_score": best_val,
            "train_counts": train_counts, "val_counts": val_counts}


def parse_args() -> TrainConfig:
    p = argparse.ArgumentParser()
    for f, default in asdict(TrainConfig()).items():
        if isinstance(default, bool):
            p.add_argument(f"--{f}", type=lambda x: x.lower() in {"1", "true", "yes"}, default=default)
        else:
            p.add_argument(f"--{f}", type=type(default), default=default)
    a = p.parse_args()
    return TrainConfig(**vars(a))


if __name__ == "__main__":
    train_gated_residual(parse_args())
