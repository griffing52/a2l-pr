#!/usr/bin/env python3
"""Closed-loop recovery evaluation: paired baseline vs. residual.

Two modes:
  --mode natural : run the base policy from fixed initial states; residual arm
                   may intervene on the policy's own (natural) behaviour. This is
                   the deployment setting (what paired_policy_evaluation did) but
                   with the improved wrapper (normalization, no gate double-scale).
  --mode induced : INJECT one of the four synthetic failure modes into the
                   executed action over a window, then measure whether the
                   residual detects + recovers vs. an un-helped baseline that
                   suffers the same injection. This is the on-distribution test
                   the residual was designed for and isolates "method works" from
                   "natural failures != synthetic failures".

Every comparison is paired: identical initial state, identical injection, and
(for diffusion) a per-state torch seed so sampling matches across arms until the
residual actually changes an action. Reports paired delta, bootstrap 95% CI, and
a sign-test p-value.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
A2L_ROOT = A2L_PR_ROOT.parent
ROBOMIMIC_ROOT = A2L_ROOT / "robomimic" / "robomimic"
sys.path.append(str(A2L_PR_ROOT / "src"))
sys.path.append(str(ROBOMIMIC_ROOT))

from a2l_pr.eval.residual_runtime import ResidualWrapper, load_gated_residual  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402

MODEL_DIRS = {
    "bc": "bc_trained_models/test",
    "diffusion": "diffusion_policy_trained_models/test",
}
FAILURE_MODES = ["underreach_idle", "premature_close", "premature_open", "lateral_drift"]


def find_best_checkpoint(model_dir):
    model_dir = Path(model_dir)
    cands = list(model_dir.glob("**/*.pth"))
    if not cands:
        raise FileNotFoundError(f"No .pth under {model_dir}")
    scored = []
    for p in cands:
        m = re.search(r"success_([0-9.]+)", p.name)
        if m:
            try:
                scored.append((float(m.group(1).rstrip(".")), p))
            except ValueError:
                pass
    if scored:
        return str(max(scored, key=lambda x: x[0])[1])
    return str(max(cands, key=lambda p: p.stat().st_mtime))


def make_env(ckpt_dict):
    if ckpt_dict.get("obs_spec"):
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=ckpt_dict["obs_spec"])
    env, _ = FileUtils.env_from_checkpoint(ckpt_dict=ckpt_dict, render=False, render_offscreen=False, verbose=False)
    return env


def make_injector(mode, start, length, drift_xy, gripper_close_val=1.0, gripper_open_val=-1.0):
    """Return inject_fn(step, base_action)->action for a fixed window."""
    end = start + length

    def inject(step, base):
        a = base.copy()
        if not (start <= step < end):
            return a
        if mode == "underreach_idle":
            a[:6] = 0.0  # freeze motion: hover/idle
        elif mode == "premature_close":
            a[6] = gripper_close_val
        elif mode == "premature_open":
            a[6] = gripper_open_val
        elif mode == "lateral_drift":
            a[0] += drift_xy[0]; a[1] += drift_xy[1]
        return a

    return inject


def rollout(policy, env, horizon, state_dict, det_seed):
    torch.manual_seed(det_seed)
    np.random.seed(det_seed)
    policy.start_episode()
    env.reset()
    obs = env.reset_to(deepcopy(state_dict))
    total = 0.0
    success = False
    steps = 0
    try:
        for i in range(horizon):
            a = policy(ob=obs)
            obs, r, done, _ = env.step(a)
            total += r
            success = bool(env.is_success().get("task", False))
            steps = i + 1
            if done or success:
                break
            obs = deepcopy(obs)
    except env.rollout_exceptions as e:
        print(f"WARNING rollout exception: {e}")
    return {"return": float(total), "success": float(success), "horizon": int(steps)}


def window_detection(wrapper, inject_start, inject_len, lookahead=40):
    """Did the gate fire within the injection window + lookahead? latency from start."""
    gates = wrapper.gate_probs  # aligned to steps where history was full
    applied = wrapper.applied_interventions
    thr = wrapper.gate_threshold
    # gate_probs only recorded once history filled (offset = history_length-1 steps).
    offset = wrapper.history_length - 1
    lo = max(0, inject_start - offset)
    hi = min(len(gates), inject_start + inject_len + lookahead - offset)
    fired_idx = [i for i in range(lo, hi) if gates[i] >= thr]
    detected = len(fired_idx) > 0
    latency = (fired_idx[0] + offset - inject_start) if detected else None
    return {"detected": detected, "latency": latency,
            "max_gate_in_window": float(np.max(gates[lo:hi])) if hi > lo else 0.0,
            "n_interventions_in_window": int(sum(applied[lo:hi]))}


def capture_states(env, n, seed):
    np.random.seed(seed); torch.manual_seed(seed)
    states = []
    for _ in range(n):
        env.reset()
        states.append(deepcopy(env.get_state()))
    return states


def bootstrap_ci(deltas, n_boot=10000, seed=0):
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = rng.choice(deltas, size=(n_boot, len(deltas)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def sign_test_p(deltas):
    """Two-sided sign test on non-zero paired deltas (binomial)."""
    deltas = np.asarray(deltas, dtype=float)
    pos = int(np.sum(deltas > 0)); neg = int(np.sum(deltas < 0))
    n = pos + neg
    if n == 0:
        return 1.0
    from math import comb
    k = min(pos, neg)
    p = sum(comb(n, i) for i in range(0, k + 1)) * (0.5 ** n) * 2
    return float(min(1.0, p))


def paired_stats(rows, key):
    a = np.array([r[f"baseline_{key}"] for r in rows], dtype=float)
    b = np.array([r[f"residual_{key}"] for r in rows], dtype=float)
    d = b - a
    ci = bootstrap_ci(d)
    return {
        "baseline_mean": float(a.mean()) if len(a) else 0.0,
        "residual_mean": float(b.mean()) if len(b) else 0.0,
        "mean_delta": float(d.mean()) if len(d) else 0.0,
        "ci95_delta": ci,
        "fraction_residual_better": float(np.mean(d > 0)) if len(d) else 0.0,
        "fraction_residual_worse": float(np.mean(d < 0)) if len(d) else 0.0,
        "sign_test_p": sign_test_p(d),
        "n": int(len(d)),
    }


def run(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    residual_policy = load_gated_residual(args.residual_checkpoint, device)
    policies = args.policies.split(",")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    drift_xy = (args.drift, args.drift)
    modes = FAILURE_MODES if args.mode == "induced" else ["none"]

    all_rows = []
    summaries = []
    for pol in policies:
        ckpt = find_best_checkpoint(ROBOMIMIC_ROOT / MODEL_DIRS[pol])
        base_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
        env = make_env(ckpt_dict)
        states = capture_states(env, args.n_initial_states, args.seed)
        print(f"[{pol}] checkpoint={ckpt}  {len(states)} states  modes={modes}", flush=True)

        for mode in modes:
            rows = []
            for si, sd in enumerate(states):
                det_seed = args.seed + si
                inj = None
                if mode != "none":
                    inj = make_injector(mode, args.inject_start, args.inject_len, drift_xy)
                # baseline: observe-only residual (logs gate, never applies)
                base_wrap = ResidualWrapper(
                    base_policy, residual_policy, device,
                    residual_weight=args.residual_weight, clamp_residual=args.clamp_residual,
                    gate_threshold=args.gate_threshold, scale_by_gate=args.scale_by_gate,
                    apply_residual=False, inject_fn=inj)
                base_stats = rollout(base_wrap, env, args.horizon, sd, det_seed)
                # residual: applies
                res_wrap = ResidualWrapper(
                    base_policy, residual_policy, device,
                    residual_weight=args.residual_weight, clamp_residual=args.clamp_residual,
                    gate_threshold=args.gate_threshold, scale_by_gate=args.scale_by_gate,
                    apply_residual=True, inject_fn=inj)
                res_stats = rollout(res_wrap, env, args.horizon, sd, det_seed)

                row = {
                    "policy": pol, "mode": mode, "state": si,
                    "baseline_return": base_stats["return"], "residual_return": res_stats["return"],
                    "baseline_success": base_stats["success"], "residual_success": res_stats["success"],
                    "residual_interventions": res_wrap.interventions,
                    "residual_mean_gate": res_wrap.diagnostics_summary()["mean_gate"],
                }
                if mode != "none":
                    det = window_detection(res_wrap, args.inject_start, args.inject_len)
                    row.update({"detected": int(det["detected"]),
                                "detect_latency": det["latency"],
                                "max_gate_in_window": det["max_gate_in_window"]})
                rows.append(row); all_rows.append(row)
            stats = {
                "policy": pol, "mode": mode,
                "return": paired_stats(rows, "return"),
                "success": paired_stats(rows, "success"),
            }
            if mode != "none":
                det_rows = [r for r in rows if r.get("detected") is not None]
                stats["detection_rate"] = float(np.mean([r["detected"] for r in det_rows])) if det_rows else 0.0
                lats = [r["detect_latency"] for r in det_rows if r["detect_latency"] is not None]
                stats["median_detect_latency"] = float(np.median(lats)) if lats else None
            summaries.append(stats)
            sr = stats["success"]
            print(f"  [{pol}/{mode}] success base={sr['baseline_mean']:.2f} res={sr['residual_mean']:.2f} "
                  f"delta={sr['mean_delta']:+.3f} CI={tuple(round(x,3) for x in sr['ci95_delta'])} "
                  f"p={sr['sign_test_p']:.3f}"
                  + (f" det_rate={stats.get('detection_rate'):.2f}" if mode != 'none' else ""), flush=True)

    payload = {"args": vars(args), "summaries": summaries}
    (out_dir / f"recovery_{args.mode}_summary.json").write_text(json.dumps(payload, indent=2))
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(out_dir / f"recovery_{args.mode}_rollouts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    print(f"wrote {out_dir}/recovery_{args.mode}_summary.json")
    return payload


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["natural", "induced"], default="induced")
    p.add_argument("--policies", default="bc,diffusion")
    p.add_argument("--n_initial_states", type=int, default=30)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--residual_checkpoint", default=str(A2L_PR_ROOT / "notebooks" / "gated_residual_recovery_policy.pth"))
    p.add_argument("--gate_threshold", type=float, default=0.5)
    p.add_argument("--residual_weight", type=float, default=0.5)
    p.add_argument("--clamp_residual", type=float, default=0.5)
    p.add_argument("--scale_by_gate", action="store_true")
    p.add_argument("--inject_start", type=int, default=25)
    p.add_argument("--inject_len", type=int, default=20)
    p.add_argument("--drift", type=float, default=0.4, help="lateral drift action bias per step")
    p.add_argument("--output_dir", default=str(A2L_PR_ROOT / "output" / "recovery_eval"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
