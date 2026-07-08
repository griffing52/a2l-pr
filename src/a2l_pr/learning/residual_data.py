"""Offline data construction for the gated residual recovery policy.

This module ports the data pipeline that previously lived only inside
`notebooks/robomimic_residual_training.ipynb` into an importable, testable
module so that training and offline detection evaluation share one code path.

Key design (mirrors the notebook so existing checkpoints stay comparable):
- State vector per step = concat of STATE_KEYS observations (eef pos/quat,
  gripper qpos, joint pos/vel) -> 23 dims for robomimic square.
- For each demo we emit sliding windows:
    * clean demo windows                -> gate_target=0, failure_type=0
    * perturbed demo, before onset      -> gate_target=0, failure_type=0
    * perturbed demo, within window     -> gate_target=1, failure_type=<id>,
      residual_target = original_action - perturbed_action over HORIZON.
- NEW vs notebook: optional input normalization. We compute per-dim mean/std
  over the *training* state/action stream and expose them so they can be saved
  into the checkpoint and reapplied identically at eval time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from a2l_pr.adapters.robomimic import RobomimicAdapter
from a2l_pr.perturbations.generator import PerturbationGenerator, PerturbationType


STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
]

# Canonical ordering shared with a2l_pr.utils.failure_labels.default_failure_id_to_type
PERTURBATION_TYPES = [
    PerturbationType.UNDERREACH_IDLE,
    PerturbationType.PREMATURE_CLOSE,
    PerturbationType.PREMATURE_OPEN,
    PerturbationType.LATERAL_DRIFT,
]
FAILURE_TYPE_TO_ID: Dict[str, int] = {"no_failure": 0}
FAILURE_TYPE_TO_ID.update({p.value: i + 1 for i, p in enumerate(PERTURBATION_TYPES)})
NUM_FAILURE_TYPES = len(FAILURE_TYPE_TO_ID)

DEFAULT_HDF5 = (
    "/home/griffing52/vail/bot2bot/bot2bot/a2l/robomimic/robomimic/"
    "datasets/square/ph/low_dim_v15.hdf5"
)

_adapter = RobomimicAdapter()


def load_demo_trajectory(hdf5_path: str, demo_key: str) -> Dict:
    """Load one robomimic demo into the standard trajectory dict."""
    with h5py.File(hdf5_path, "r") as f:
        grp = f["data"][demo_key]
        obs = {k: np.asarray(grp["obs"][k][:], dtype=np.float32) for k in STATE_KEYS if k in grp["obs"]}
        actions = np.asarray(grp["actions"][:], dtype=np.float32)
    traj = {"observations": obs, "actions": actions}
    return _adapter.load(traj)


def list_demo_keys(hdf5_path: str) -> List[str]:
    with h5py.File(hdf5_path, "r") as f:
        keys = list(f["data"].keys())
    # robomimic stores demo_0, demo_1 ...; sort numerically for stable splits.
    def _idx(k: str) -> int:
        try:
            return int(k.split("_")[-1])
        except ValueError:
            return 0
    return sorted(keys, key=_idx)


def build_state_vector(traj: Dict, t: int) -> Optional[np.ndarray]:
    parts = []
    obs = traj["observations"]
    for key in STATE_KEYS:
        if key in obs and t < len(obs[key]):
            parts.append(np.asarray(obs[key][t]).reshape(-1))
    if not parts:
        return None
    return np.concatenate(parts).astype(np.float32)


def _traj_len(traj: Dict) -> int:
    return int(len(traj["actions"]))


@dataclass
class ResidualRecord:
    past_states: np.ndarray      # (H, state_dim)
    past_actions: np.ndarray     # (H, action_dim)
    residual_target: np.ndarray  # (HORIZON, action_dim)
    gate_target: float
    failure_type: int
    severity: float = 0.0        # perturbation severity in [0,1] (0 for clean/pre windows)
    # bookkeeping for analysis / detection eval
    demo_key: str = ""
    anchor_t: int = 0
    bucket: str = "clean"
    onset_offset: int = -1       # anchor_t - perturb_start (<0 means before onset)


@dataclass
class NormStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_ckpt(self) -> Dict[str, list]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @staticmethod
    def from_ckpt(d: Dict) -> Optional["NormStats"]:
        if not d or "state_mean" not in d:
            return None
        return NormStats(
            state_mean=np.asarray(d["state_mean"], dtype=np.float32),
            state_std=np.asarray(d["state_std"], dtype=np.float32),
            action_mean=np.asarray(d["action_mean"], dtype=np.float32),
            action_std=np.asarray(d["action_std"], dtype=np.float32),
        )


def _window_starts(traj_len: int, history: int, horizon: int, stride: int) -> List[int]:
    last = traj_len - 1
    return list(range(history, max(history + 1, last - 1), stride))


def _history_for_window(traj: Dict, anchor_t: int, history: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    past_states, past_actions = [], []
    for t in range(anchor_t - history, anchor_t):
        s = build_state_vector(traj, t)
        if s is None or t < 0 or t >= len(traj["actions"]):
            return None, None
        past_states.append(s)
        past_actions.append(np.asarray(traj["actions"][t]).reshape(-1).astype(np.float32))
    return np.stack(past_states), np.stack(past_actions)


def _target_residuals(original: Dict, perturbed: Dict, anchor_t: int, horizon: int, action_dim: int) -> np.ndarray:
    out = []
    for t in range(anchor_t, anchor_t + horizon):
        if t < len(original["actions"]) and t < len(perturbed["actions"]):
            orig = np.asarray(original["actions"][t]).reshape(-1).astype(np.float32)
            pert = np.asarray(perturbed["actions"][t]).reshape(-1).astype(np.float32)
            out.append(orig - pert)
        else:
            out.append(np.zeros(action_dim, dtype=np.float32))
    return np.stack(out)


def generate_records(
    demo_keys: List[str],
    hdf5_path: str,
    history: int = 12,
    horizon: int = 30,
    stride: int = 6,
    severity: float = 0.75,
    seed: int = 0,
    verbose: bool = True,
    severity_range: Optional[Tuple[float, float]] = None,
) -> Tuple[List[ResidualRecord], Dict[str, int]]:
    """Build sliding-window records for clean + 4 perturbed variants of each demo.

    If `severity_range` is given, each (demo, perturbation) draws a severity uniformly
    from that range instead of the fixed `severity`, so a severity head has signal.
    """
    rng = np.random.default_rng(seed)
    generator = PerturbationGenerator(rng=rng)
    records: List[ResidualRecord] = []
    counts = {"clean": 0, "pre_perturb": 0, "positive": 0}

    for demo_key in demo_keys:
        traj = load_demo_trajectory(hdf5_path, demo_key)
        traj_len = _traj_len(traj)
        if traj_len < history + horizon + 2:
            continue
        action_dim = np.asarray(traj["actions"][0]).reshape(-1).shape[0]
        zero_res = np.zeros((horizon, action_dim), dtype=np.float32)

        # clean windows
        for anchor_t in _window_starts(traj_len, history, horizon, stride):
            ps, pa = _history_for_window(traj, anchor_t, history)
            if ps is None:
                continue
            records.append(ResidualRecord(ps, pa, zero_res.copy(), 0.0, 0, 0.0, demo_key, anchor_t, "clean", -1))
            counts["clean"] += 1

        # perturbed variants
        for p_type in PERTURBATION_TYPES:
            sev = float(rng.uniform(*severity_range)) if severity_range else severity
            res = generator.apply_perturbation(traj, p_type, severity=sev)
            if res is None:
                continue
            pert = res.perturbed_trajectory
            p_start, p_end = map(int, res.perturbation_window)
            failure_type = FAILURE_TYPE_TO_ID[p_type.value]
            n = min(len(traj["actions"]), len(pert["actions"]))
            for anchor_t in _window_starts(n, history, horizon, stride):
                ps, pa = _history_for_window(pert, anchor_t, history)
                if ps is None:
                    continue
                if anchor_t < p_start:
                    records.append(ResidualRecord(ps, pa, zero_res.copy(), 0.0, 0, 0.0, demo_key, anchor_t, "pre_perturb", anchor_t - p_start))
                    counts["pre_perturb"] += 1
                elif anchor_t <= p_end + horizon:
                    tgt = _target_residuals(traj, pert, anchor_t, horizon, action_dim)
                    records.append(ResidualRecord(ps, pa, tgt, 1.0, failure_type, sev, demo_key, anchor_t, "positive", anchor_t - p_start))
                    counts["positive"] += 1

    if verbose:
        print(f"record counts: {counts} (total={sum(counts.values())})")
    return records, counts


def compute_norm_stats(records: List[ResidualRecord]) -> NormStats:
    states = np.concatenate([r.past_states for r in records], axis=0)
    actions = np.concatenate([r.past_actions for r in records], axis=0)
    s_mean, s_std = states.mean(0), states.std(0)
    a_mean, a_std = actions.mean(0), actions.std(0)
    s_std[s_std < 1e-6] = 1.0
    a_std[a_std < 1e-6] = 1.0
    return NormStats(s_mean.astype(np.float32), s_std.astype(np.float32),
                     a_mean.astype(np.float32), a_std.astype(np.float32))
