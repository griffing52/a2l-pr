#!/usr/bin/env python3
"""Detector -> FSM recovery: closed-loop scripted recovery triggered by the gate.

Motivation (see results.md, 2026-06-25): the learned action-residual detects well
but corrects poorly (neutral/harmful), partly because it fires every step and adds a
mismatched delta. This module keeps ONLY the detector (gate + failure-type) and, when
it trips, hands control to a short, one-shot SCRIPTED recovery in OSC-pose action
space, then re-engages the base policy. A refractory window prevents continuous
re-firing (the suspected cause of the residual's harm).

Recovery programs (OSC action = [dx,dy,dz,drx,dry,drz, gripper], +1 close / -1 open):
  premature_close -> reopen gripper (let the policy re-approach + re-grasp)
  premature_open  -> re-close gripper, small lift
  underreach_idle -> move toward the object (descend onto it), then resume
  lateral_drift   -> re-center over the object in xy, then resume

Geometry: obs['object'][0:3] is the (eef - nut) vector, so action[:3] = -gain*rel
drives the gripper toward the target. Verified empirically (square/ph).

Evaluation mirrors recovery_eval.py (paired, identical injection, per-state seed) so
the FSM numbers are directly comparable to the learned-residual numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
sys.path.append(str(A2L_PR_ROOT / "src"))
sys.path.append(str(A2L_PR_ROOT.parent / "robomimic" / "robomimic"))

from a2l_pr.eval.residual_runtime import (  # noqa: E402
    ResidualWrapper, extract_state, load_gated_residual, predict_window,
)
from a2l_pr.eval import recovery_eval as RE  # noqa: E402
from a2l_pr.utils.failure_labels import merge_failure_id_to_type, select_failure_label  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


def _clip_action(a):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    a[:6] = np.clip(a[:6], -1.0, 1.0)
    a[6] = np.clip(a[6], -1.0, 1.0)
    return a


# A recovery program is a list of (primitive, params, n_steps).
def recovery_program(failure_name, severity=0.5):
    name = failure_name.lower()
    # severity scales the number of steps / magnitude where it makes sense.
    extra = int(round(4 * severity))
    if "premature_gripper_close" in name or name == "premature_close":
        return [("set_gripper", {"val": -1.0}, 6 + extra)]
    if "premature_gripper_open" in name or name == "premature_open":
        return [("set_gripper", {"val": 1.0}, 6 + extra),
                ("move_z", {"dz": 0.3, "grip": 1.0}, 3)]
    if "underreach" in name:
        # DISENGAGE: an approach-phase freeze self-recovers once the base policy resumes;
        # any scripted motion fights it and hurts (move_to_object measured -0.23). Detecting
        # it but NOT acting is the correct policy. Empty program -> no intervention.
        return []
    if "lateral_drift" in name or "drift" in name:
        return [("move_to_object", {"gain": 4.0, "use_z": False, "grip": None, "clip": 0.3}, 10 + extra)]
    return []  # no_failure / unknown -> no-op (disengage)


class FSMRecoveryWrapper:
    """Base policy + detector-gated scripted recovery FSM."""

    def __init__(self, base_policy, detector, device, gate_threshold=0.85,
                 refractory=18, move_gain=8.0, inject_fn=None, use_severity=False,
                 trigger_persist=3, max_recoveries=2,
                 trigger_mode="detector", oracle_step=None, oracle_mode=None,
                 recover_modes=None):
        self.base_policy = base_policy
        self.detector = detector
        self.device = device
        self.gate_threshold = gate_threshold
        self.refractory = refractory
        self.move_gain = move_gain
        self.inject_fn = inject_fn
        self.use_severity = use_severity
        self.trigger_persist = trigger_persist
        self.max_recoveries = max_recoveries
        self.trigger_mode = trigger_mode  # "detector" | "oracle"
        self.oracle_step = oracle_step
        self.oracle_mode = oracle_mode
        # restrict which DETECTED failure types are allowed to trigger a recovery
        # (None = all). Substring match against the type label.
        self.recover_modes = recover_modes
        self.history_length = int(getattr(detector, "history_length", 12))
        meta = getattr(detector, "checkpoint_metadata", {})
        n = int(meta.get("num_failure_types", 5))
        mapping = meta.get("failure_type_to_id", {}) or {}
        self.failure_id_to_type = merge_failure_id_to_type(
            {int(v): str(k) for k, v in mapping.items()}, num_failure_types=n)
        self.reset_diagnostics()

    def start_episode(self):
        if hasattr(self.base_policy, "start_episode"):
            self.base_policy.start_episode()
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.state_history = []
        self.action_history = []
        self.step = 0
        self.mode = "NORMAL"       # NORMAL | RECOVERING | REFRACTORY
        self.program = []
        self.prog_idx = 0
        self.prog_step_left = 0
        self.refractory_left = 0
        self.interventions = 0
        self.gate_probs = []
        self.applied_interventions = []
        self.recovered_types = []
        self.persist_count = 0
        self.n_recoveries = 0

    def _detect(self, ob, provisional_action):
        state = extract_state(ob)
        if state is None:
            return None
        self.state_history.append(state)
        self.action_history.append(np.asarray(provisional_action, dtype=np.float32))
        self.state_history = self.state_history[-self.history_length:]
        self.action_history = self.action_history[-self.history_length:]
        if len(self.state_history) < self.history_length:
            return None
        pred = predict_window(self.detector, np.stack(self.state_history),
                              np.stack(self.action_history), self.device)
        sel = select_failure_label(pred["failure_probs"], self.failure_id_to_type,
                                   prefer_non_no_failure=pred["gate_prob"] >= self.gate_threshold)
        sev = float(pred["severity"]) if (self.use_severity and "severity" in pred) else 0.5
        return {"gate": pred["gate_prob"], "type": str(sel["label"]), "severity": sev}

    def _scripted_action(self, ob, base):
        prim, params, _ = self.program[self.prog_idx]
        a = np.zeros(7, dtype=np.float32)
        if prim == "set_gripper":
            a[6] = params["val"]
        elif prim == "move_z":
            a[2] = params["dz"]; a[6] = params["grip"]
        elif prim == "move_to_object":
            obj = np.asarray(ob["object"]) if "object" in ob else np.zeros(3)
            if obj.ndim > 1:          # diffusion exposes stacked obs history; use latest frame
                obj = obj[-1]
            rel = obj[:3]
            cmd = -params["gain"] * rel
            if not params["use_z"]:
                cmd[2] = 0.0
            clip = params.get("clip", 1.0)
            a[:3] = np.clip(cmd, -clip, clip)
            a[6] = base[6] if params["grip"] is None else params["grip"]
        return _clip_action(a)

    def _advance_program(self):
        self.prog_step_left -= 1
        if self.prog_step_left <= 0:
            self.prog_idx += 1
            if self.prog_idx >= len(self.program):
                self.mode = "REFRACTORY"
                self.refractory_left = self.refractory
            else:
                self.prog_step_left = self.program[self.prog_idx][2]

    def __call__(self, ob, goal=None, batched_ob=False):
        base = np.asarray(self.base_policy(ob=ob, goal=goal, batched_ob=batched_ob)).reshape(-1).astype(np.float32)
        if base.shape[0] != self.detector.action_dim:
            base = base[: self.detector.action_dim].copy()
        corrupted = base.copy()
        if self.inject_fn is not None:
            corrupted = np.asarray(self.inject_fn(self.step, base), dtype=np.float32).reshape(-1)
        self.step += 1

        det = self._detect(ob, corrupted)  # detector sees the (failing) stream
        gate = det["gate"] if det else 0.0
        self.gate_probs.append(gate)

        # persistence counter for conservative detector triggering
        if gate >= self.gate_threshold:
            self.persist_count += 1
        else:
            self.persist_count = 0

        executed = corrupted
        applied = False
        if self.mode == "NORMAL":
            trigger, trig_type, trig_sev = False, None, 0.5
            if self.trigger_mode == "oracle":
                if self.oracle_step is not None and (self.step - 1) == self.oracle_step:
                    trigger, trig_type = True, self.oracle_mode
            elif det and self.persist_count >= self.trigger_persist and self.n_recoveries < self.max_recoveries:
                trigger, trig_type, trig_sev = True, det["type"], det["severity"]
            if trigger and self.recover_modes is not None:
                # only act if the DETECTED type is in the allowed set (else disengage)
                if not any(m in str(trig_type) for m in self.recover_modes):
                    trigger = False
            if trigger:
                prog = recovery_program(trig_type, trig_sev)
                if prog:
                    self.mode = "RECOVERING"
                    self.program = prog
                    self.prog_idx = 0
                    self.prog_step_left = prog[0][2]
                    self.recovered_types.append(trig_type)
                    self.interventions += 1
                    self.n_recoveries += 1
                    self.persist_count = 0
                    executed = self._scripted_action(ob, base)
                    applied = True
                    self._advance_program()
        elif self.mode == "RECOVERING":
            executed = self._scripted_action(ob, base)
            applied = True
            self._advance_program()
        elif self.mode == "REFRACTORY":
            self.refractory_left -= 1
            if self.refractory_left <= 0:
                self.mode = "NORMAL"

        self.applied_interventions.append(applied)
        if self.action_history:
            self.action_history[-1] = executed.astype(np.float32)  # history reflects truth
        return _clip_action(executed)

    def diagnostics_summary(self):
        return {"interventions": int(self.interventions),
                "mean_gate": float(np.mean(self.gate_probs)) if self.gate_probs else 0.0,
                "max_gate": float(np.max(self.gate_probs)) if self.gate_probs else 0.0}


def run(args):
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)
    detector = load_gated_residual(args.detector_checkpoint, device)
    policies = args.policies.split(",")
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    drift_xy = (args.drift, args.drift)

    all_rows, summaries = [], []
    for pol in policies:
        ckpt = RE.find_best_checkpoint(RE.ROBOMIMIC_ROOT / RE.MODEL_DIRS[pol])
        import robomimic.utils.file_utils as FileUtils
        base_policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=ckpt, device=device, verbose=False)
        env = RE.make_env(ckpt_dict)
        states = RE.capture_states(env, args.n_initial_states, args.seed)
        print(f"[{pol}] {len(states)} states", flush=True)
        for mode in RE.FAILURE_MODES:
            rows = []
            for si, sd in enumerate(states):
                det_seed = args.seed + si
                inj = RE.make_injector(mode, args.inject_start, args.inject_len, drift_xy)
                # baseline: injection, detector observes but never acts
                base_wrap = ResidualWrapper(base_policy, detector, device,
                                            gate_threshold=args.gate_threshold,
                                            apply_residual=False, inject_fn=inj)
                base_stats = RE.rollout(base_wrap, env, args.horizon, sd, det_seed)
                # fsm: injection + scripted recovery
                fsm = FSMRecoveryWrapper(base_policy, detector, device,
                                         gate_threshold=args.gate_threshold,
                                         refractory=args.refractory, inject_fn=inj,
                                         use_severity=args.use_severity,
                                         trigger_persist=args.trigger_persist,
                                         max_recoveries=args.max_recoveries,
                                         trigger_mode=args.trigger,
                                         oracle_step=args.inject_start + args.inject_len,
                                         oracle_mode=mode,
                                         recover_modes=(args.recover_modes.split(",") if args.recover_modes else None))
                fsm_stats = RE.rollout(fsm, env, args.horizon, sd, det_seed)
                row = {"policy": pol, "mode": mode, "state": si,
                       "baseline_return": base_stats["return"], "residual_return": fsm_stats["return"],
                       "baseline_success": base_stats["success"], "residual_success": fsm_stats["success"],
                       "fsm_interventions": fsm.interventions}
                rows.append(row); all_rows.append(row)
            stats = {"policy": pol, "mode": mode,
                     "return": RE.paired_stats(rows, "return"),
                     "success": RE.paired_stats(rows, "success"),
                     "mean_interventions": float(np.mean([r["fsm_interventions"] for r in rows]))}
            summaries.append(stats)
            sr = stats["success"]
            print(f"  [{pol}/{mode}] success base={sr['baseline_mean']:.2f} fsm={sr['residual_mean']:.2f} "
                  f"delta={sr['mean_delta']:+.3f} CI={tuple(round(x,3) for x in sr['ci95_delta'])} "
                  f"p={sr['sign_test_p']:.3f} interv={stats['mean_interventions']:.1f}", flush=True)

    (out_dir / "fsm_recovery_summary.json").write_text(json.dumps({"args": vars(args), "summaries": summaries}, indent=2))
    if all_rows:
        keys = sorted({k for r in all_rows for k in r})
        with open(out_dir / "fsm_recovery_rollouts.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(all_rows)
    print(f"wrote {out_dir}/fsm_recovery_summary.json", flush=True)
    return summaries


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--policies", default="bc,diffusion")
    p.add_argument("--n_initial_states", type=int, default=30)
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--detector_checkpoint", default=str(A2L_PR_ROOT / "notebooks" / "gated_residual_retrained_norm.pth"))
    p.add_argument("--gate_threshold", type=float, default=0.85)
    p.add_argument("--refractory", type=int, default=18)
    p.add_argument("--trigger", choices=["detector", "oracle"], default="detector")
    p.add_argument("--trigger_persist", type=int, default=3)
    p.add_argument("--max_recoveries", type=int, default=2)
    p.add_argument("--recover_modes", default=None,
                   help="comma list of detected-type substrings allowed to trigger recovery "
                        "(e.g. 'premature_close,lateral_drift'); default = all")
    p.add_argument("--inject_start", type=int, default=25)
    p.add_argument("--inject_len", type=int, default=20)
    p.add_argument("--drift", type=float, default=0.4)
    p.add_argument("--use_severity", action="store_true")
    p.add_argument("--output_dir", default=str(A2L_PR_ROOT / "output" / "fsm_recovery"))
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
