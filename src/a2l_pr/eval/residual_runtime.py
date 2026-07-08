"""Shared runtime for residual-policy evaluation (detection + closed-loop).

Provides a single source of truth for:
- loading a GatedResidualRecoveryPolicy checkpoint (with optional norm stats),
- normalizing a (states, actions) history exactly as in training,
- an improved closed-loop wrapper that fixes the eval-time issues identified in
  the diagnosis: optional input normalization, an optional `scale_by_gate` flag
  (the original wrapper multiplied the residual by both residual_weight and the
  gate probability -> ~0.003 effective magnitude), an `apply_residual` flag for
  observe-only baselines, and an `inject_fn` hook so we can corrupt the executed
  action mid-rollout to test on-distribution recovery.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
import torch

from a2l_pr.learning.residual_data import NormStats
from a2l_pr.models import GatedResidualRecoveryPolicy
from a2l_pr.utils.failure_labels import merge_failure_id_to_type, select_failure_label

STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
]


def load_gated_residual(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if not isinstance(ckpt, dict) or ckpt.get("model_class") != "GatedResidualRecoveryPolicy":
        raise ValueError(f"Expected GatedResidualRecoveryPolicy checkpoint, got {checkpoint_path}")
    model = GatedResidualRecoveryPolicy(
        state_dim=int(ckpt["state_dim"]),
        action_dim=int(ckpt["action_dim"]),
        history_length=int(ckpt.get("history_length", 12)),
        prediction_horizon=int(ckpt.get("prediction_horizon", 30)),
        num_failure_types=int(ckpt.get("num_failure_types", 5)),
        predict_severity=bool(ckpt.get("predict_severity", False)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.checkpoint_metadata = ckpt
    model.norm_stats = NormStats.from_ckpt(ckpt.get("norm_stats"))
    return model


def normalize_history(states: np.ndarray, actions: np.ndarray, norm: Optional[NormStats]):
    if norm is None:
        return states, actions
    s = (states - norm.state_mean) / norm.state_std
    a = (actions - norm.action_mean) / norm.action_std
    return s.astype(np.float32), a.astype(np.float32)


def predict_window(model, states: np.ndarray, actions: np.ndarray, device) -> Dict:
    """Run the model on one (H,state)/(H,action) window. Returns numpy outputs."""
    s, a = normalize_history(states, actions, getattr(model, "norm_stats", None))
    with torch.no_grad():
        ps = torch.tensor(s, dtype=torch.float32, device=device).unsqueeze(0)
        pa = torch.tensor(a, dtype=torch.float32, device=device).unsqueeze(0)
        out = model.predict_first_step(ps, pa)
    result = {
        "gate_prob": float(out["gate_probs"][0].item()),
        "residual": out["residuals"][0].cpu().numpy(),
        "failure_probs": torch.softmax(out["failure_logits"][0], dim=-1).cpu().numpy(),
    }
    if "severity" in out:
        result["severity"] = float(out["severity"][0].item())
    return result


def extract_state(ob) -> Optional[np.ndarray]:
    parts = []
    for key in STATE_KEYS:
        if key in ob:
            value = np.asarray(ob[key])
            if value.ndim > 1:
                value = value[-1]
            parts.append(value.reshape(-1).astype(np.float32))
    return np.concatenate(parts) if parts else None


class ResidualWrapper:
    """Closed-loop policy wrapper around a base robomimic policy.

    executed_action = inject_fn(base_action)              # optional corruption
                    + (residual term if applied & gated)

    The residual term is `residual_weight * residual` (clamped). If
    `scale_by_gate` is True it is additionally multiplied by the gate prob
    (legacy behaviour). The corrupted action is what gets recorded into the
    history, matching the training distribution (perturbed action stream).
    """

    def __init__(self, base_policy, residual_policy, device,
                 residual_weight=0.5, clamp_residual=0.5, gate_threshold=0.5,
                 scale_by_gate=False, apply_residual=True,
                 inject_fn: Optional[Callable[[int, np.ndarray], np.ndarray]] = None):
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.device = device
        self.residual_weight = residual_weight
        self.clamp_residual = clamp_residual
        self.gate_threshold = gate_threshold
        self.scale_by_gate = scale_by_gate
        self.apply_residual = apply_residual
        self.inject_fn = inject_fn
        self.history_length = int(getattr(residual_policy, "history_length", 12))
        meta = getattr(residual_policy, "checkpoint_metadata", {})
        n = int(meta.get("num_failure_types", getattr(residual_policy, "num_failure_types", 5)))
        mapping = meta.get("failure_type_to_id", {}) or {}
        id_to_type = {int(v): str(k) for k, v in mapping.items()}
        self.failure_id_to_type = merge_failure_id_to_type(id_to_type, num_failure_types=n)
        self.reset_diagnostics()

    def start_episode(self):
        if hasattr(self.base_policy, "start_episode"):
            self.base_policy.start_episode()
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.state_history: List[np.ndarray] = []
        self.action_history: List[np.ndarray] = []
        self.action_step = 0
        self.interventions = 0
        self.gate_probs: List[float] = []
        self.residual_norms: List[float] = []
        self.failure_type_names: List[str] = []
        self.applied_interventions: List[bool] = []
        self.injected_steps: List[bool] = []

    def __call__(self, ob, goal=None, batched_ob=False):
        base = np.asarray(self.base_policy(ob=ob, goal=goal, batched_ob=batched_ob)).reshape(-1).astype(np.float32)
        if base.shape[0] != self.residual_policy.action_dim:
            base = base[: self.residual_policy.action_dim].copy()
        step = self.action_step
        self.action_step += 1

        injected = False
        executed = base.copy()
        if self.inject_fn is not None:
            corrupted = np.asarray(self.inject_fn(step, base), dtype=np.float32).reshape(-1)
            injected = not np.allclose(corrupted, base)
            executed = corrupted
        self.injected_steps.append(injected)

        state = extract_state(ob)
        if state is None:
            return executed
        self.state_history.append(state)
        self.action_history.append(executed.copy())
        self.state_history = self.state_history[-self.history_length:]
        self.action_history = self.action_history[-self.history_length:]
        if len(self.state_history) < self.history_length:
            return executed

        pred = predict_window(self.residual_policy, np.stack(self.state_history),
                              np.stack(self.action_history), self.device)
        gate_prob = pred["gate_prob"]
        residual = np.clip(pred["residual"], -self.clamp_residual, self.clamp_residual)
        sel = select_failure_label(pred["failure_probs"], self.failure_id_to_type,
                                   prefer_non_no_failure=gate_prob >= self.gate_threshold)
        self.gate_probs.append(gate_prob)
        self.residual_norms.append(float(np.linalg.norm(residual)))
        self.failure_type_names.append(str(sel["label"]))

        applied = self.apply_residual and gate_prob >= self.gate_threshold
        self.applied_interventions.append(applied)
        if applied:
            self.interventions += 1
            scale = self.residual_weight * (gate_prob if self.scale_by_gate else 1.0)
            return executed + scale * residual
        return executed

    def diagnostics_summary(self) -> Dict:
        if not self.gate_probs:
            return {"interventions": 0, "mean_gate": 0.0, "max_gate": 0.0}
        return {
            "interventions": int(self.interventions),
            "mean_gate": float(np.mean(self.gate_probs)),
            "max_gate": float(np.max(self.gate_probs)),
            "mean_residual_norm": float(np.mean(self.residual_norms)),
        }
