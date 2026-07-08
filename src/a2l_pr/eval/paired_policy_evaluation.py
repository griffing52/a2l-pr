#!/usr/bin/env python3
"""Paired evaluation for BC / diffusion with and without gated residuals.

The official eval script compares each policy in its own rollout stream, which
means the environment can diverge across policies before the residual gate even
has a chance to act. This harness captures a fixed set of initial states once,
then replays the exact same starts for every policy so the return deltas become
paired and interpretable.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
import textwrap

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
# A2L_PR_ROOT = Path(__file__).resolve().parents[1]
A2L_ROOT = A2L_PR_ROOT.parent
ROBOMIMIC_ROOT = A2L_ROOT / "robomimic" / "robomimic"
sys.path.append(str(A2L_PR_ROOT / "src"))
sys.path.append(str(ROBOMIMIC_ROOT))

from a2l_pr.models import GatedResidualRecoveryPolicy  # noqa: E402
from a2l_pr.utils.failure_labels import merge_failure_id_to_type, select_failure_label  # noqa: E402
import robomimic.utils.file_utils as FileUtils  # noqa: E402
import robomimic.utils.obs_utils as ObsUtils  # noqa: E402
import robomimic.utils.torch_utils as TorchUtils  # noqa: E402


STATE_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
]


MODEL_REGISTRY = {
    "bc": {
        "dir": "bc_trained_models/test",
        "arg_name": "--bc_checkpoint",
        "help": "Path to BC policy checkpoint (defaults to best under robomimic/bc_trained_models/test)",
    },
    "bc_transformer": {
        "dir": "bc_transformer_trained_models/test",
        "arg_name": "--bc_transformer_checkpoint",
        "help": "Path to BC Transformer policy checkpoint (defaults to best under robomimic/bc_transformer_trained_models/test)",
    },
    "diffusion": {
        "dir": "diffusion_policy_trained_models/test",
        "arg_name": "--diffusion_checkpoint",
        "help": "Path to Diffusion policy checkpoint (defaults to best under robomimic/diffusion_policy_trained_models/test)",
    },
}



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
        num_failure_types = int(metadata.get("num_failure_types", getattr(residual_policy, "num_failure_types", 5)))
        mapping = metadata.get("failure_type_to_id", {}) or {}
        id_to_type = {int(v): str(k) for k, v in mapping.items()}
        self.failure_id_to_type = merge_failure_id_to_type(id_to_type, num_failure_types=num_failure_types)
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
            base_action = base_action[:self.residual_policy.action_dim].copy()
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
        applied = gate_prob >= self.gate_threshold

        selected_failure = select_failure_label(
            failure_probs,
            self.failure_id_to_type,
            prefer_non_no_failure=applied,
        )
        failure_name = str(selected_failure["label"])
        failure_conf = float(selected_failure["confidence"])

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
            return {"interventions": 0, "mean_gate": 0.0, "max_gate": 0.0, "mean_residual_norm": 0.0, "top_failure_at_max_gate": "none"}
        top_idx = int(np.argmax(self.gate_probs))
        return {
            "interventions": int(self.interventions),
            "mean_gate": float(np.mean(self.gate_probs)),
            "max_gate": float(np.max(self.gate_probs)),
            "mean_residual_norm": float(np.mean(self.residual_norms)),
            "top_failure_at_max_gate": self.failure_type_names[top_idx],
            "top_failure_confidence": float(self.failure_confidences[top_idx]),
        }

    def intervention_trace(self):
        return {
            "gate_probs": list(self.gate_probs),
            "failure_type_names": list(self.failure_type_names),
            "failure_confidences": list(self.failure_confidences),
            "applied_interventions": list(self.applied_interventions),
            "residual_norms": list(self.residual_norms),
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


def overlay_bottom_left_text(frame, text, max_line_width=34):
    if not text:
        return frame

    wrapped = textwrap.fill(text, width=max_line_width)
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=font, spacing=3)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    margin = 10
    padding = 6
    x0 = margin
    y0 = max(margin, image.height - text_height - padding * 2 - margin)
    x1 = x0 + text_width + padding * 2
    y1 = y0 + text_height + padding * 2

    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    draw.multiline_text((x0 + padding, y0 + padding), wrapped, fill=(255, 255, 255), font=font, spacing=3)
    return np.asarray(image)


def overlay_top_left_label(frame, label):
    if not label:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    # font = ImageFont.truetype("arial.ttf", 32)
    bbox = draw.textbbox((0, 0), label, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    padding = 6
    x0 = 10
    y0 = 10
    x1 = x0 + text_width + padding * 2
    y1 = y0 + text_height + padding * 2

    draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
    draw.text((x0 + padding, y0 + padding), label, fill=(255, 255, 255), font=font)
    return np.asarray(image)


def stack_side_by_side(left_frame, right_frame):
    left = np.asarray(left_frame)
    right = np.asarray(right_frame)
    height = min(left.shape[0], right.shape[0])
    if left.shape[0] != height:
        left = left[:height]
    if right.shape[0] != height:
        right = right[:height]
    return np.concatenate([left, right], axis=1)


def rollout_from_state(policy, env, horizon, state_dict, capture_frames=False, video_skip=5, camera_names=None, video_label=None):
    if camera_names is None:
        camera_names = ["agentview"]
    policy.start_episode()
    obs = env.reset()
    obs = env.reset_to(deepcopy(state_dict))

    total_reward = 0.0
    success = False
    steps = 0
    frames = []
    try:
        for step_i in range(horizon):
            action = policy(ob=obs)
            next_obs, reward, done, _ = env.step(action)
            total_reward += reward
            success = bool(env.is_success().get("task", False))
            if capture_frames and step_i % video_skip == 0:
                frame = render_frame(env, camera_names)
                if video_label:
                    frame = overlay_top_left_label(frame, video_label)
                if isinstance(policy, GatedResidualWrapper) and policy.applied_interventions:
                    last_idx = len(policy.applied_interventions) - 1
                    if policy.applied_interventions[last_idx]:
                        failure_name = policy.failure_type_names[last_idx]
                        frame = overlay_bottom_left_text(frame, failure_name)
                frames.append(frame)
            steps = step_i + 1
            if done or success:
                break
            obs = deepcopy(next_obs)
    except env.rollout_exceptions as e:
        print(f"WARNING: rollout exception: {e}")
    return {"return": float(total_reward), "horizon": int(steps), "success": float(success), "frames": frames}


def capture_initial_states(env, n_initial_states, seed):
    states = []
    np.random.seed(seed)
    torch.manual_seed(seed)
    for _ in range(n_initial_states):
        env.reset()
        states.append(deepcopy(env.get_state()))
    return states


def write_csv(path, rows):
    if not rows:
        return
    keys = sorted({key for row in rows for key in row.keys()})
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_video(path, frames, fps=20):
    if not frames:
        return
    with imageio.get_writer(str(path), fps=fps) as writer:
        for frame in frames:
            writer.append_data(np.asarray(frame))


def paired_summary(rows, policy_a, policy_b):
    policy_a_returns = np.array([row[f"{policy_a}_return"] for row in rows], dtype=float)
    policy_b_returns = np.array([row[f"{policy_b}_return"] for row in rows], dtype=float)
    delta = policy_b_returns - policy_a_returns
    better = float(np.mean(delta > 0)) if len(delta) else 0.0
    equal = float(np.mean(np.isclose(delta, 0.0))) if len(delta) else 0.0
    return {
        "policy_a": policy_a,
        "policy_b": policy_b,
        "mean_return_a": float(np.mean(policy_a_returns)) if len(policy_a_returns) else 0.0,
        "mean_return_b": float(np.mean(policy_b_returns)) if len(policy_b_returns) else 0.0,
        "mean_delta": float(np.mean(delta)) if len(delta) else 0.0,
        "std_delta": float(np.std(delta)) if len(delta) else 0.0,
        "fraction_b_better": better,
        "fraction_tied": equal,
    }


def evaluate_policy_on_states(name, checkpoint_path, residual_policy, state_dicts, args, device, output_dir):
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device, verbose=False)
    env = make_env_from_checkpoint(ckpt_dict, video=args.render_video)
    if residual_policy is not None:
        policy = GatedResidualWrapper(
            policy,
            residual_policy,
            device=device,
            residual_weight=args.residual_weight,
            clamp_residual=args.clamp_residual,
            gate_threshold=args.gate_threshold,
        )

    video_writer = None
    video_path = None
    if args.render_video:
        video_path = output_dir / f"{name}.mp4"
        video_writer = imageio.get_writer(str(video_path), fps=20)

    rows = []
    try:
        video_label = "Residual" if residual_policy is not None else "Baseline"
        for i, state_dict in enumerate(state_dicts):
            stats = rollout_from_state(
                policy,
                env,
                args.horizon,
                state_dict,
                capture_frames=args.render_video,
                video_skip=args.video_skip,
                camera_names=args.camera_names,
                video_label=video_label if args.render_video else None,
            )
            frames = stats.pop("frames", [])
            row = {"policy": name, "initial_state": i, **stats}
            if isinstance(policy, GatedResidualWrapper):
                row.update(policy.diagnostics_summary())
                row["had_intervention"] = bool(row.get("interventions", 0) > 0)
            rows.append(row)
            if video_writer is not None and frames:
                for frame in frames:
                    video_writer.append_data(frame)
            print(f"{name} start {i + 1}/{len(state_dicts)}: return={stats['return']:.1f} success={stats['success']:.0f} horizon={stats['horizon']}")
    finally:
        if video_writer is not None:
            video_writer.close()

    returns = np.array([row["return"] for row in rows], dtype=float)
    successes = np.array([row["success"] for row in rows], dtype=float)
    summary = {
        "policy": name,
        "checkpoint": str(checkpoint_path),
        "n_rollouts": int(len(rows)),
        "mean_return": float(np.mean(returns)) if len(returns) else 0.0,
        "std_return": float(np.std(returns)) if len(returns) else 0.0,
        "success_rate": float(np.mean(successes)) if len(successes) else 0.0,
        "video_path": str(video_path) if video_path is not None else None,
    }
    if residual_policy is not None:
        summary.update({
            "mean_interventions": float(np.mean([r.get("interventions", 0) for r in rows])),
            "mean_gate": float(np.mean([r.get("mean_gate", 0.0) for r in rows])),
            "max_gate": float(np.max([r.get("max_gate", 0.0) for r in rows])),
        })
    return summary, rows


def save_twin_comparison_videos(name, checkpoint_path, residual_policy, state_dicts, rows, args, device, output_dir):
    if residual_policy is None:
        return
    if not args.save_intervention_videos and not args.save_twin_videos:
        return

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device, verbose=False)
    twin_dir = output_dir / "videos"
    twin_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    for row in rows:
        if saved >= args.video_state_limit:
            break
        if not row.get("had_intervention", False):
            continue

        state_index = int(row["initial_state"])
        state_dict = state_dicts[state_index]

        base_policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=checkpoint_path, device=device, verbose=False)
        base_env = make_env_from_checkpoint(ckpt_dict, video=True)
        base_result = rollout_from_state(
            base_policy,
            base_env,
            args.horizon,
            state_dict,
            capture_frames=True,
            video_skip=args.video_skip,
            camera_names=args.camera_names,
            video_label="Baseline",
        )

        residual_policy_clone = load_gated_residual(args.residual_checkpoint, device=device)
        residual_wrapper = GatedResidualWrapper(
            policy,
            residual_policy_clone,
            device=device,
            residual_weight=args.residual_weight,
            clamp_residual=args.clamp_residual,
            gate_threshold=args.gate_threshold,
        )
        residual_env = make_env_from_checkpoint(ckpt_dict, video=True)
        residual_result = rollout_from_state(
            residual_wrapper,
            residual_env,
            args.horizon,
            state_dict,
            capture_frames=True,
            video_skip=args.video_skip,
            camera_names=args.camera_names,
            video_label="Residual",
        )

        base_frames = base_result.get("frames", [])
        residual_frames = residual_result.get("frames", [])
        if not base_frames or not residual_frames:
            continue

        compare_frames = []
        max_steps = min(len(base_frames), len(residual_frames))
        for step_idx in range(max_steps):
            left = overlay_top_left_label(base_frames[step_idx], "Baseline")
            right = overlay_top_left_label(residual_frames[step_idx], "Residual")
            compare_frames.append(stack_side_by_side(left, right))

        state_tag = f"state{state_index:03d}"
        base_path = twin_dir / f"{name}_{state_tag}_baseline.mp4"
        residual_path = twin_dir / f"{name}_{state_tag}_residual.mp4"
        compare_path = twin_dir / f"{name}_{state_tag}_compare.mp4"

        write_video(base_path, base_frames)
        write_video(residual_path, residual_frames)
        write_video(compare_path, compare_frames)

        if args.save_failure_videos:
            failure_dir = twin_dir / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_path = failure_dir / f"{name}_{state_tag}_failure.mp4"
            write_video(failure_path, residual_frames)
        saved += 1



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_initial_states", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--render_video", action="store_true")
    parser.add_argument("--video_skip", type=int, default=5)
    parser.add_argument("--camera_names", nargs="+", default=["agentview"])
    parser.add_argument("--output_dir", type=str, default=str(A2L_PR_ROOT / "output" / "paired_eval_results"))
    
    # Dynamically add registered model options
    for name, config in MODEL_REGISTRY.items():
        parser.add_argument(config["arg_name"], type=str, default=None, help=config["help"])

    parser.add_argument("--residual_checkpoint", type=str, default=str(A2L_PR_ROOT / "notebooks" / "gated_residual_recovery_policy.pth"))
    parser.add_argument("--gate_threshold", type=float, default=0.7)
    parser.add_argument("--residual_weight", type=float, default=0.1)
    parser.add_argument("--clamp_residual", type=float, default=0.1)
    parser.add_argument("--save_twin_videos", action="store_true")
    parser.add_argument("--save_intervention_videos", action="store_true")
    parser.add_argument("--save_failure_videos", action="store_true")
    parser.add_argument("--video_state_limit", type=int, default=4)
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = TorchUtils.get_torch_device(try_to_use_cuda=True)

    # Resolve checkpoints dynamically
    checkpoints = {}
    for name, config in MODEL_REGISTRY.items():
        arg_attr = config["arg_name"].lstrip('-').replace('-', '_')
        val = getattr(args, arg_attr, None)
        if val is None:
            val = find_best_checkpoint(ROBOMIMIC_ROOT / config["dir"])
        checkpoints[name] = val

    residual_policy = load_gated_residual(args.residual_checkpoint, device=device)

    print("Paired evaluation uses identical initial states across all policies.")
    for name in MODEL_REGISTRY:
        print(f"{name.upper()} checkpoint: {checkpoints[name]}")
    print(f"Residual checkpoint: {args.residual_checkpoint}")

    # Use first available checkpoint to initialize the environment and capture initial states
    first_model_name = list(MODEL_REGISTRY.keys())[0]
    first_ckpt = checkpoints[first_model_name]
    _, first_ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=first_ckpt, device=device, verbose=False)
    base_env = make_env_from_checkpoint(first_ckpt_dict, video=args.render_video)
    state_dicts = capture_initial_states(base_env, args.n_initial_states, args.seed)

    all_summaries = []
    all_rows = []
    specs = []
    for name in MODEL_REGISTRY:
        specs.append((name, checkpoints[name], None))
        specs.append((f"{name}_gated_residual", checkpoints[name], residual_policy))

    for name, checkpoint, maybe_residual in specs:
        print("\n" + "=" * 80)
        print(f"Evaluating {name}")
        print("=" * 80)
        summary, rows = evaluate_policy_on_states(name, checkpoint, maybe_residual, state_dicts, args, device, output_dir)
        all_summaries.append(summary)
        all_rows.extend(rows)
        if maybe_residual is not None and (args.save_twin_videos or args.save_intervention_videos):
            save_twin_comparison_videos(name, checkpoint, maybe_residual, state_dicts, rows, args, device, output_dir)

    paired_rows = []
    by_state = {}
    for row in all_rows:
        by_state.setdefault(row["initial_state"], {})[row["policy"]] = row

    paired_pairs = [(name, f"{name}_gated_residual") for name in MODEL_REGISTRY]

    paired_summaries = []
    for a, b in paired_pairs:
        rows = []
        for state_idx, row_map in sorted(by_state.items()):
            if a in row_map and b in row_map:
                rows.append({
                    "initial_state": state_idx,
                    f"{a}_return": row_map[a]["return"],
                    f"{b}_return": row_map[b]["return"],
                    f"{a}_success": row_map[a]["success"],
                    f"{b}_success": row_map[b]["success"],
                })
        paired_rows.extend(rows)
        paired_summaries.append(paired_summary(rows, a, b))

    summary_path = output_dir / "paired_eval_summary.json"
    rollout_path = output_dir / "paired_eval_rollouts.csv"
    summary_path.write_text(json.dumps({"args": vars(args), "summaries": all_summaries, "paired": paired_summaries}, indent=2))
    write_csv(rollout_path, all_rows)

    paired_path = output_dir / "paired_deltas.csv"
    write_csv(paired_path, paired_rows)

    print("\nPaired summary:")
    for item in paired_summaries:
        print(
            f"{item['policy_b']} vs {item['policy_a']}: mean_delta={item['mean_delta']:.3f} "
            f"fraction_b_better={item['fraction_b_better']:.2%}"
        )
    print(f"Wrote {summary_path}")
    print(f"Wrote {rollout_path}")
    print(f"Wrote {paired_path}")


if __name__ == "__main__":
    main()