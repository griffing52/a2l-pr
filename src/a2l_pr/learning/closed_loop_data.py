#!/usr/bin/env python3
"""Collect CLOSED-LOOP detector training data from live policy rollouts.

Motivation (results.md, Round 2): the detector trained on offline-edited demo windows
detects well offline (AUROC ~0.9) but fires spuriously on the LIVE policy's normal
motion (4-6 false recoveries/episode), because its negatives came from demo states, not
from states the policy actually visits. Fix: build the training set from live rollouts:
  - CLEAN rollouts  -> every window is a NEGATIVE (real normal policy behaviour)
  - INJECTED rollouts -> windows during the injection (+ short tail) are POSITIVE
    (failure type = injected mode); windows before onset are negatives.

Produces a list of `ResidualRecord` (residual_target = zeros; we train gate + failure
type + severity, not the action residual). Reuses the same record/window format as the
offline pipeline so `train_gated_residual(..., train_records=...)` consumes it directly.

This module DOES drive the simulator, so run it when the GPU is free (diffusion).
Collection and training are separated: collect once to an .npz, then train cheaply.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
sys.path.append(str(A2L_PR_ROOT / "src"))
sys.path.append(str(A2L_PR_ROOT.parent / "robomimic" / "robomimic"))

from a2l_pr.learning.residual_data import FAILURE_TYPE_TO_ID, ResidualRecord  # noqa: E402
from a2l_pr.eval.residual_runtime import extract_state  # noqa: E402
from a2l_pr.eval import recovery_eval as RE  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402

MODE_TO_ID = {
    "underreach_idle": FAILURE_TYPE_TO_ID["underreach_idle_before_max_reach"],
    "premature_close": FAILURE_TYPE_TO_ID["premature_gripper_close"],
    "premature_open": FAILURE_TYPE_TO_ID["premature_gripper_open"],
    "lateral_drift": FAILURE_TYPE_TO_ID["lateral_drift"],
}


def _rollout_record(policy, env, horizon, state_dict, det_seed, inject_fn=None):
    """Run one rollout; return per-step (states, executed_actions, injected_flags)."""
    torch.manual_seed(det_seed); np.random.seed(det_seed)
    if hasattr(policy, "start_episode"):
        policy.start_episode()
    env.reset()
    obs = env.reset_to({k: v for k, v in state_dict.items()}) if isinstance(state_dict, dict) else env.reset_to(state_dict)
    states, actions, injected = [], [], []
    action_dim = None
    try:
        for i in range(horizon):
            base = np.asarray(policy(ob=obs)).reshape(-1).astype(np.float32)
            if action_dim is None:
                action_dim = base.shape[0]
            executed = base
            inj = False
            if inject_fn is not None:
                executed = np.asarray(inject_fn(i, base), dtype=np.float32).reshape(-1)
                inj = not np.allclose(executed, base)
            s = extract_state(obs)
            if s is not None:
                states.append(s); actions.append(executed.copy()); injected.append(inj)
            obs, r, done, _ = env.step(executed)
            if done or bool(env.is_success().get("task", False)):
                break
    except env.rollout_exceptions as e:
        print(f"WARNING rollout exception: {e}", flush=True)
    return np.array(states), np.array(actions), np.array(injected, dtype=bool)


def _windows(states, actions, history, stride, gate, ftype, severity, demo_key, onset_ref=None, tail=30,
             inj_window=None):
    """Slide history-length windows; label window-ending-at-t by step t's status."""
    recs = []
    zero_res = np.zeros((1, actions.shape[1]), dtype=np.float32) if len(actions) else None
    n = len(states)
    for t in range(history - 1, n, stride):
        ps = states[t - history + 1: t + 1]
        pa = actions[t - history + 1: t + 1]
        if ps.shape[0] != history:
            continue
        if inj_window is not None:
            a, b = inj_window
            if t < a:
                g, ft, sev, bucket = 0.0, 0, 0.0, "pre_perturb"
            elif a <= t <= b + tail:
                g, ft, sev, bucket = 1.0, ftype, severity, "positive"
            else:
                continue  # ambiguous aftermath -> skip
            onset = t - a
        else:
            g, ft, sev, bucket, onset = 0.0, 0, 0.0, "clean", -1
        recs.append(ResidualRecord(
            ps.astype(np.float32), pa.astype(np.float32),
            np.zeros((30, actions.shape[1]), dtype=np.float32),
            g, ft, sev, demo_key, t, bucket, onset))
    return recs


def collect_records(policies, n_clean=20, n_injected=10, horizon=400, seed=0,
                    history=12, stride=3, inject_start=25, inject_len=20, drift=0.4,
                    device=None, log=print) -> List[ResidualRecord]:
    device = device or TorchUtils.get_torch_device(try_to_use_cuda=True)
    rng = np.random.default_rng(seed)
    records: List[ResidualRecord] = []
    for pol in policies:
        ckpt = RE.find_best_checkpoint(RE.ROBOMIMIC_ROOT / RE.MODEL_DIRS[pol])
        base_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
        env = RE.make_env(ckpt_dict)
        states = RE.capture_states(env, n_clean + n_injected * len(MODE_TO_ID), seed)
        si = 0
        # clean rollouts
        for _ in range(n_clean):
            S, A, _ = _rollout_record(base_policy, env, horizon, states[si], seed + si)
            if len(S) >= history:
                records += _windows(S, A, history, stride, 0.0, 0, 0.0, f"{pol}_clean_{si}")
            si += 1
        # injected rollouts (per mode)
        for mode, ftype in MODE_TO_ID.items():
            for _ in range(n_injected):
                sev = float(rng.uniform(0.25, 1.0))
                ilen = int(round(inject_len * (0.5 + sev)))
                inj = RE.make_injector(mode, inject_start, ilen, (drift * sev, drift * sev))
                S, A, flags = _rollout_record(base_policy, env, horizon, states[si], seed + si, inject_fn=inj)
                if len(S) >= history:
                    # actual injected window from flags (more robust than nominal)
                    idx = np.where(flags)[0]
                    a = int(idx[0]) if len(idx) else inject_start
                    b = int(idx[-1]) if len(idx) else inject_start + ilen
                    records += _windows(S, A, history, stride, 1.0, ftype, sev,
                                        f"{pol}_{mode}_{si}", inj_window=(a, b))
                si += 1
        log(f"[collect] {pol}: {len(records)} cumulative windows", flush=True)
    return records


def save_records(records, path):
    np.savez_compressed(
        path,
        past_states=np.stack([r.past_states for r in records]),
        past_actions=np.stack([r.past_actions for r in records]),
        gate=np.array([r.gate_target for r in records], dtype=np.float32),
        failure_type=np.array([r.failure_type for r in records], dtype=np.int64),
        severity=np.array([r.severity for r in records], dtype=np.float32),
        onset=np.array([r.onset_offset for r in records], dtype=np.int64),
        bucket=np.array([r.bucket for r in records]),
        demo_key=np.array([r.demo_key for r in records]),
    )


def load_records(path) -> List[ResidualRecord]:
    d = np.load(path, allow_pickle=True)
    out = []
    for i in range(len(d["gate"])):
        out.append(ResidualRecord(
            d["past_states"][i], d["past_actions"][i],
            np.zeros((30, d["past_actions"].shape[-1]), dtype=np.float32),
            float(d["gate"][i]), int(d["failure_type"][i]), float(d["severity"][i]),
            str(d["demo_key"][i]), 0, str(d["bucket"][i]), int(d["onset"][i])))
    return out


def split_records(records, val_frac=0.2, seed=0) -> Tuple[List, List]:
    """Split by demo_key so windows from one rollout don't leak across train/val."""
    keys = sorted({r.demo_key for r in records})
    rng = np.random.default_rng(seed); rng.shuffle(keys)
    n_val = max(1, int(len(keys) * val_frac))
    val_keys = set(keys[:n_val])
    train = [r for r in records if r.demo_key not in val_keys]
    val = [r for r in records if r.demo_key in val_keys]
    return train, val


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", default="bc,diffusion")
    p.add_argument("--n_clean", type=int, default=20)
    p.add_argument("--n_injected", type=int, default=10)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--stride", type=int, default=3)
    p.add_argument("--out", default=str(A2L_PR_ROOT / "output" / "closed_loop" / "cl_records.npz"))
    args = p.parse_args()
    recs = collect_records(args.policies.split(","), n_clean=args.n_clean, n_injected=args.n_injected,
                           horizon=args.horizon, seed=args.seed, stride=args.stride)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    save_records(recs, args.out)
    pos = sum(int(r.gate_target > 0.5) for r in recs)
    print(f"collected {len(recs)} windows ({pos} positive, {len(recs)-pos} negative) -> {args.out}")


if __name__ == "__main__":
    main()
