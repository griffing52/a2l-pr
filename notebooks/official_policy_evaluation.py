#!/usr/bin/env python3
"""Official four-way policy evaluation for BC / diffusion with optional gated residuals.

Run from the a2l-pr directory, usually with:
    conda run -n a2l python notebooks/official_policy_evaluation.py --n_rollouts 50 --horizon 400 --seed 0 --render_video
"""

import argparse
import csv
import json
import os
import re
import sys
from copy import deepcopy
from pathlib import Path

import h5py
import imageio.v2 as imageio
import numpy as np
import torch

# Keep imports relative to this repository layout.
A2L_PR_ROOT = Path(__file__).resolve().parents[1]
A2L_ROOT = A2L_PR_ROOT.parent
ROBOMIMIC_ROOT = A2L_ROOT / "robomimic" / "robomimic"
sys.path.append(str(A2L_PR_ROOT / "src"))
sys.path.append(str(ROBOMIMIC_ROOT))

from a2l_pr.models import GatedResidualRecoveryPolicy  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402
from robomimic.algo import RolloutPolicy  # noqa: E402


STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
]


def find_best_checkpoint(model_dir):
    model_dir = Path(model_dir)
    candidates = list(model_dir.glob("**/*.pth"))
    if not candidates:
        raise FileNotFoundError(f"No .pth checkpoints found under {model_dir}")

    success_candidates = []
    for path in candidates:
        match = re.search(r"success_([0-9.]+)", path.name)
        if match:
            try:
                success_candidates.append((float(match.group(1).rstrip('.')), path))
            except ValueError:
                pass
    if success_candidates:
        return str(max(success_candidates, key=lambda x: x[0])[1])

    epoch_candidates = []
    for path in candidates:
        match = re.search(r"epoch_(\d+)", path.name)
        if match:
            epoch_candidates.append((int(match.group(1)), path))
    if epoch_candidates:
        return str(max(epoch_candidates, key=lambda x: x[0])[1])

    return str(max(candidates, key=lambda p: p.stat().st_mtime))


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
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    model.checkpoint_metadata = ckpt
    return model


class GatedResidualWrapper:
    def __init__(self, base_policy, residual_policy, device, residual_weight=0.1, clamp_residual=0.1, gate_threshold=0.7):
        self.base_policy = base_policy
        self.residual_policy = residual_policy
        self.device = device
        self.residual_weight = residual_weight
        self.clamp_residual = clamp_residual
        self.gate_threshold = gate_threshold
        self.history_length = int(getattr(residual_policy, "history_length", 12))
        metadata = getattr(residual_policy, "checkpoint_metadata", {})
        mapping = metadata.get("failure_type_to_id", {}) or {}
        self.failure_id_to_type = {int(v): str(k) for k, v in mapping.items()} or {0: "no_failure"}
        self.reset_diagnostics()

    def start_episode(self):
        if hasattr(self.base_policy, "start_episode"):
            self.base_policy.start_episode()
        self.reset_diagnostics()

    def reset_diagnostics(self):
        self.state_history = []
        self.action_history = []
        self.action_step = 0
        self.interventions = 0
        self.gate_probs = []
        self.residual_norms = []
        self.failure_type_names = []
        self.failure_confidences = []
        self.applied_interventions = []

    def _extract_state(self, obs):
        parts = []
        for key in STATE_KEYS:
            if key in obs:
                value = np.asarray(obs[key])
                # Diffusion checkpoints may expose stacked observation history; use the latest frame.
                if value.ndim > 1:
                    value = value[-1]
                parts.append(value.reshape(-1).astype(np.float32))
        if parts:
            return np.concatenate(parts)
        return None

    def __call__(self, ob, goal=None, batched_ob=False):
        base_action = self.base_policy(ob=ob, goal=goal, batched_ob=batched_ob)
        base_action = np.asarray(base_action).reshape(-1).astype(np.float32)
        if base_action.shape[0] != self.residual_policy.action_dim:
            # Diffusion policies can expose an action chunk; the environment executes one 7D action.
            base_action = base_action[:self.residual_policy.action_dim].copy()
        current_step = self.action_step
        self.action_step += 1

        state = self._extract_state(ob)
        if state is None:
            return base_action
        self.state_history.append(state)
        self.action_history.append(base_action.copy())
        self.state_history = self.state_history[-self.history_length:]
        self.action_history = self.action_history[-self.history_length:]
        if len(self.state_history) < self.history_length:
            return base_action

        with torch.no_grad():
            ps = torch.tensor(np.stack(self.state_history), dtype=torch.float32, device=self.device).unsqueeze(0)
            pa = torch.tensor(np.stack(self.action_history), dtype=torch.float32, device=self.device).unsqueeze(0)
            out = self.residual_policy.predict_first_step(ps, pa)
        gate_prob = float(out["gate_probs"][0].detach().cpu().item())
        residual = out["residuals"][0].detach().cpu().numpy()
        residual = np.clip(residual, -self.clamp_residual, self.clamp_residual)
        residual_norm = float(np.linalg.norm(residual))
        failure_probs = torch.softmax(out["failure_logits"][0], dim=-1).detach().cpu().numpy()
        failure_id = int(np.argmax(failure_probs))
        failure_name = self.failure_id_to_type.get(failure_id, f"failure_{failure_id}")
        failure_conf = float(failure_probs[failure_id])
        applied = gate_prob >= self.gate_threshold

        self.gate_probs.append(gate_prob)
        self.residual_norms.append(residual_norm)
        self.failure_type_names.append(failure_name)
        self.failure_confidences.append(failure_conf)
        self.applied_interventions.append(applied)

        if applied:
            self.interventions += 1
            return base_action + self.residual_weight * gate_prob * residual
        return base_action

    def diagnostics_summary(self):
        if not self.gate_probs:
            return {
                "interventions": 0,
                "mean_gate": 0.0,
                "max_gate": 0.0,
                "mean_residual_norm": 0.0,
                "top_failure_at_max_gate": "none",
            }
        top_idx = int(np.argmax(self.gate_probs))
        return {
            "interventions": int(self.interventions),
            "mean_gate": float(np.mean(self.gate_probs)),
            "max_gate": float(np.max(self.gate_probs)),
            "mean_residual_norm": float(np.mean(self.residual_norms)),
            "top_failure_at_max_gate": self.failure_type_names[top_idx],
            "top_failure_confidence": float(self.failure_confidences[top_idx]),
        }


def make_env_from_checkpoint(ckpt_dict, video=False):
    obs_spec = ckpt_dict.get("obs_spec")
    if obs_spec:
        ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)
    env, _ = FileUtils.env_from_checkpoint(
        ckpt_dict=ckpt_dict,
        render=False,
        render_offscreen=video,
        verbose=False,
    )
    return env


def render_frame(env, camera_names, height=512, width=512):
    frames = []
    for camera_name in camera_names:
        frames.append(env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name))
    return np.concatenate(frames, axis=1)


def rollout(policy, env, horizon, video_writer=None, video_skip=5, camera_names=None):
    if camera_names is None:
        camera_names = ["agentview"]
    policy.start_episode()
    obs = env.reset()
    state_dict = env.get_state()
    obs = env.reset_to(state_dict)

    total_reward = 0.0
    success = False
    steps = 0
    try:
        for step_i in range(horizon):
            action = policy(ob=obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            success = bool(env.is_success().get("task", False))
            if video_writer is not None and step_i % video_skip == 0:
                video_writer.append_data(render_frame(env, camera_names))
            steps = step_i + 1
            if done or success:
                break
            obs = deepcopy(next_obs)
            state_dict = env.get_state()
    except env.rollout_exceptions as e:
        print(f"WARNING: rollout exception: {e}")
    return {"return": float(total_reward), "horizon": int(steps), "success": float(success)}


def evaluate_policy(name, checkpoint_path, residual_policy, args, device, output_dir):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device, verbose=False)
    if residual_policy is not None:
        policy = GatedResidualWrapper(
            policy,
            residual_policy,
            device=device,
            residual_weight=args.residual_weight,
            clamp_residual=args.clamp_residual,
            gate_threshold=args.gate_threshold,
        )
    env = make_env_from_checkpoint(ckpt_dict, video=args.render_video)
    video_writer = None
    video_path = None
    if args.render_video:
        video_path = output_dir / f"{name}.mp4"
        video_writer = imageio.get_writer(str(video_path), fps=20)

    rollout_rows = []
    try:
        for i in range(args.n_rollouts):
            stats = rollout(policy, env, args.horizon, video_writer=video_writer, video_skip=args.video_skip, camera_names=args.camera_names)
            row = {"policy": name, "rollout": i, **stats}
            if isinstance(policy, GatedResidualWrapper):
                row.update(policy.diagnostics_summary())
            rollout_rows.append(row)
            print(
                f"{name} rollout {i + 1}/{args.n_rollouts}: "
                f"return={stats['return']:.1f} success={stats['success']:.0f} horizon={stats['horizon']}"
            )
    finally:
        if video_writer is not None:
            video_writer.close()

    returns = np.array([r["return"] for r in rollout_rows], dtype=float)
    successes = np.array([r["success"] for r in rollout_rows], dtype=float)
    summary = {
        "policy": name,
        "checkpoint": str(checkpoint_path),
        "n_rollouts": int(len(rollout_rows)),
        "mean_return": float(np.mean(returns)) if len(returns) else 0.0,
        "std_return": float(np.std(returns)) if len(returns) else 0.0,
        "min_return": float(np.min(returns)) if len(returns) else 0.0,
        "max_return": float(np.max(returns)) if len(returns) else 0.0,
        "success_rate": float(np.mean(successes)) if len(successes) else 0.0,
        "video_path": str(video_path) if video_path is not None else None,
    }
    if residual_policy is not None:
        summary.update({
            "mean_interventions": float(np.mean([r.get("interventions", 0) for r in rollout_rows])),
            "mean_gate": float(np.mean([r.get("mean_gate", 0.0) for r in rollout_rows])),
            "max_gate": float(np.max([r.get("max_gate", 0.0) for r in rollout_rows])),
        })
    return summary, rollout_rows


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--camera_names", nargs="+", default=["agentview", "robot0_eye_in_hand"])
    parser.add_argument("--output_dir", type=str, default=str(A2L_PR_ROOT / "notebooks" / "official_eval_results"))
    parser.add_argument("--bc_checkpoint", type=str, default=None)
    parser.add_argument("--diffusion_checkpoint", type=str, default=None)
    parser.add_argument("--residual_checkpoint", type=str, default=str(A2L_PR_ROOT / "notebooks" / "gated_residual_recovery_policy.pth"))
    parser.add_argument("--gate_threshold", type=float, default=0.7)
    parser.add_argument("--residual_weight", type=float, default=0.1)
    parser.add_argument("--clamp_residual", type=float, default=0.1)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    bc_checkpoint = args.bc_checkpoint or find_best_checkpoint(ROBOMIMIC_ROOT / "bc_trained_models" / "test")
    diffusion_checkpoint = args.diffusion_checkpoint or find_best_checkpoint(ROBOMIMIC_ROOT / "diffusion_policy_trained_models" / "test")
    residual_policy = load_gated_residual(args.residual_checkpoint, device=device)

    print("Residual policy input: low-dimensional state/action history only; videos are evaluation renderings, not model inputs.")
    print(f"BC checkpoint: {bc_checkpoint}")
    print(f"Diffusion checkpoint: {diffusion_checkpoint}")
    print(f"Residual checkpoint: {args.residual_checkpoint}")
    print(f"Output dir: {output_dir}")

    all_summaries = []
    all_rollouts = []
    specs = [
        ("bc", bc_checkpoint, None),
        ("bc_gated_residual", bc_checkpoint, residual_policy),
        ("diffusion", diffusion_checkpoint, None),
        ("diffusion_gated_residual", diffusion_checkpoint, residual_policy),
    ]
    for name, checkpoint, maybe_residual in specs:
        print("\n" + "=" * 80)
        print(f"Evaluating {name}")
        print("=" * 80)
        summary, rows = evaluate_policy(name, checkpoint, maybe_residual, args, device, output_dir)
        all_summaries.append(summary)
        all_rollouts.extend(rows)

    summary_path = output_dir / "official_eval_summary.json"
    rollout_path = output_dir / "official_eval_rollouts.csv"
    summary_path.write_text(json.dumps({"args": vars(args), "summaries": all_summaries}, indent=2))
    write_csv(rollout_path, all_rollouts)

    print("\nOfficial evaluation summary:")
    for summary in all_summaries:
        extra = ""
        if "mean_interventions" in summary:
            extra = f" mean_interventions={summary['mean_interventions']:.2f} mean_gate={summary['mean_gate']:.3f} max_gate={summary['max_gate']:.3f}"
        print(
            f"{summary['policy']}: mean_return={summary['mean_return']:.2f} "
            f"std={summary['std_return']:.2f} success={summary['success_rate']:.2%}{extra}"
        )
    print(f"Wrote {summary_path}")
    print(f"Wrote {rollout_path}")


if __name__ == "__main__":
    main()
