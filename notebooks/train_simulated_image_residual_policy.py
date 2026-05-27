#!/usr/bin/env python3
"""Train an image-conditioned gated residual policy from perturbed Robomimic rollouts.

The script builds a dataset of short rollout windows from simulated trajectories,
applies synthetic perturbations, and learns to predict the residual action that
brings the perturbed rollout back toward the original demonstration.

Inputs:
- low-dimensional history: end-effector pose, gripper state, joint positions,
  joint velocities, and previous actions
- image history: real Robomimic RGB frames when available, otherwise a small
  synthetic rendering derived from the state history

Outputs:
- a checkpoint containing the trained policy and dataset metadata
- a JSON training summary with dataset statistics and validation metrics
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = REPO_ROOT.parent
sys.path.append(str(REPO_ROOT / "src"))

from a2l_pr.adapters.robomimic import RobomimicAdapter
from a2l_pr.perturbations.generator import PerturbationGenerator, PerturbationType


STATE_KEY_PRIORITY = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "robot0_joint_pos",
    "robot0_joint_vel",
    "object-state",
]

IMAGE_KEY_PRIORITY = [
    "agentview_image",
    "robot0_eye_in_hand_image",
    "eye_in_hand_image",
    "wrist_image",
]


@dataclass(frozen=True)
class ResidualSample:
    state_history: np.ndarray
    action_history: np.ndarray
    image_history: np.ndarray
    target_residual: np.ndarray
    gate_label: int
    perturbation_type: str
    demo_key: str
    timestep: int


class ResidualWindowDataset(Dataset):
    def __init__(self, samples: Sequence[ResidualSample]):
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "state_history": torch.tensor(sample.state_history, dtype=torch.float32),
            "action_history": torch.tensor(sample.action_history, dtype=torch.float32),
            "image_history": torch.tensor(sample.image_history, dtype=torch.float32),
            "target_residual": torch.tensor(sample.target_residual, dtype=torch.float32),
            "gate_label": torch.tensor(sample.gate_label, dtype=torch.float32),
        }


class TinyImageEncoder(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.net(images)


class ImageConditionedGatedResidualPolicy(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        history_length: int,
        image_embed_dim: int = 128,
        history_embed_dim: int = 256,
        fusion_hidden_dim: int = 256,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.history_length = history_length

        self.history_encoder = nn.GRU(
            input_size=state_dim + action_dim,
            hidden_size=history_embed_dim,
            batch_first=True,
        )
        self.image_encoder = TinyImageEncoder(in_channels=3, embed_dim=image_embed_dim)
        self.fusion = nn.Sequential(
            nn.Linear(history_embed_dim + image_embed_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim),
            nn.LayerNorm(fusion_hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.residual_head = nn.Linear(fusion_hidden_dim, action_dim)
        self.gate_head = nn.Linear(fusion_hidden_dim, 1)

    def forward(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        image_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        history_input = torch.cat([state_history, action_history], dim=-1)
        _, hidden = self.history_encoder(history_input)
        history_feat = hidden[-1]

        batch_size, time_steps, channels, height, width = image_history.shape
        image_feat = self.image_encoder(image_history.reshape(batch_size * time_steps, channels, height, width))
        image_feat = image_feat.reshape(batch_size, time_steps, -1).mean(dim=1)

        fused = self.fusion(torch.cat([history_feat, image_feat], dim=-1))
        return {
            "residuals": self.residual_head(fused),
            "gate_logits": self.gate_head(fused).squeeze(-1),
        }

    def predict_first_step(
        self,
        state_history: torch.Tensor,
        action_history: torch.Tensor,
        image_history: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        outputs = self.forward(state_history, action_history, image_history)
        outputs["gate_probs"] = torch.sigmoid(outputs["gate_logits"])
        return outputs


def load_robomimic_demo(hdf5_path: Path, demo_key: str) -> Dict:
    with h5py.File(hdf5_path, "r") as handle:
        demo_grp = handle["data"][demo_key]
        obs_grp = demo_grp["obs"]
        observations = {key: obs_grp[key][:] for key in obs_grp.keys()}
        return {
            "actions": demo_grp["actions"][:],
            "observations": observations,
            "metadata": {"dataset_type": "robomimic"},
            "demo_key": demo_key,
        }


def list_demo_keys(hdf5_path: Path, max_demos: int) -> List[str]:
    with h5py.File(hdf5_path, "r") as handle:
        demo_keys = sorted(handle["data"].keys())
    return demo_keys[:max_demos]


def infer_image_key(observation_keys: Iterable[str]) -> Optional[str]:
    keys = list(observation_keys)
    for candidate in IMAGE_KEY_PRIORITY:
        if candidate in keys:
            return candidate
    for key in keys:
        if "image" in key.lower():
            return key
    return None


def infer_trajectory_length(trajectory: Dict) -> int:
    actions = np.asarray(trajectory["actions"])
    lengths = [len(actions)]
    observations = trajectory.get("observations", {})
    for value in observations.values():
        arr = np.asarray(value)
        if arr.ndim >= 1:
            lengths.append(int(arr.shape[0]))
    return int(min(lengths)) if lengths else 0


def extract_state_vector(observations: Dict[str, np.ndarray], step: int, state_keys: Sequence[str]) -> np.ndarray:
    parts: List[np.ndarray] = []
    for key in state_keys:
        if key not in observations:
            continue
        value = np.asarray(observations[key])
        if step >= len(value):
            continue
        item = np.asarray(value[step]).reshape(-1).astype(np.float32)
        if item.size > 0:
            parts.append(item)
    if not parts:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(parts, axis=0).astype(np.float32)


def build_state_matrix(observations: Dict[str, np.ndarray], length: int) -> np.ndarray:
    rows = [extract_state_vector(observations, step, STATE_KEY_PRIORITY) for step in range(length)]
    return np.stack(rows, axis=0).astype(np.float32)


def normalize_image(frame: np.ndarray) -> np.ndarray:
    array = np.asarray(frame)
    if array.ndim != 3:
        raise ValueError(f"Expected an RGB frame with 3 dimensions, got shape {array.shape}")
    if array.shape[0] in {1, 3} and array.shape[-1] not in {1, 3}:
        array = np.transpose(array, (1, 2, 0))
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] != 3:
        raise ValueError(f"Expected 3-channel RGB data, got shape {array.shape}")
    return array.astype(np.float32) / 255.0


def select_window_steps(length: int, perturb_window: Tuple[int, int], history_length: int, max_samples: int, rng: np.random.Generator) -> List[int]:
    start, end = perturb_window
    left = max(history_length - 1, start - 6)
    right = min(length - 1, end + 10)
    local_steps = list(range(left, right + 1))

    if len(local_steps) <= max_samples:
        return local_steps

    keep = sorted(rng.choice(local_steps, size=max_samples, replace=False).tolist())
    return keep


def render_state_to_image(state_vector: np.ndarray, image_size: int = 64) -> np.ndarray:
    """Create a deterministic pseudo-image from a low-dimensional state vector."""
    vec = np.tanh(np.asarray(state_vector, dtype=np.float32).reshape(-1))
    canvas = np.zeros((3, image_size, image_size), dtype=np.float32)
    if vec.size == 0:
        return canvas

    bins = min(image_size, max(12, vec.size))
    col_width = max(1, image_size // bins)
    for index, value in enumerate(vec[:bins]):
        height = int((float(value) + 1.0) * 0.5 * (image_size - 1))
        x0 = index * col_width
        x1 = min(image_size, x0 + col_width)
        color = index % 3
        canvas[color, image_size - height - 1 :, x0:x1] = 0.25 + 0.75 * abs(float(value))

    return canvas


def extract_image_history(
    observations: Dict[str, np.ndarray],
    image_key: Optional[str],
    step: int,
    history_length: int,
    synthetic_state_history: Optional[np.ndarray] = None,
) -> np.ndarray:
    frames: List[np.ndarray] = []
    start = step - history_length + 1
    for cursor in range(start, step + 1):
        if image_key is not None and image_key in observations and cursor < len(observations[image_key]):
            frame = normalize_image(observations[image_key][cursor])
            frame = np.transpose(frame, (2, 0, 1))
        else:
            if synthetic_state_history is None:
                raise ValueError("Synthetic image mode requested without state history")
            frame = render_state_to_image(synthetic_state_history[cursor - start])
        frames.append(frame.astype(np.float32))
    return np.stack(frames, axis=0)


def build_clean_samples(
    demo_key: str,
    trajectory: Dict,
    history_length: int,
    clean_samples_per_demo: int,
    rng: np.random.Generator,
) -> List[ResidualSample]:
    length = infer_trajectory_length(trajectory)
    if length < history_length + 1:
        return []

    actions = np.asarray(trajectory["actions"], dtype=np.float32)[:length]
    observations = trajectory["observations"]
    state_matrix = build_state_matrix(observations, length)
    image_key = infer_image_key(observations.keys())

    candidate_steps = list(range(history_length - 1, length))
    if len(candidate_steps) > clean_samples_per_demo:
        candidate_steps = sorted(rng.choice(candidate_steps, size=clean_samples_per_demo, replace=False).tolist())

    samples: List[ResidualSample] = []
    for step in candidate_steps:
        state_history = state_matrix[step - history_length + 1 : step + 1]
        action_history = actions[step - history_length + 1 : step + 1]
        image_history = extract_image_history(observations, image_key, step, history_length, synthetic_state_history=state_history)
        target = np.zeros_like(actions[step], dtype=np.float32)
        samples.append(
            ResidualSample(
                state_history=state_history,
                action_history=action_history,
                image_history=image_history,
                target_residual=target,
                gate_label=0,
                perturbation_type="clean",
                demo_key=demo_key,
                timestep=int(step),
            )
        )
    return samples


def build_perturbed_samples(
    demo_key: str,
    trajectory: Dict,
    perturbation_generator: PerturbationGenerator,
    history_length: int,
    perturbations_per_demo: int,
    samples_per_perturbation: int,
    gate_residual_threshold: float,
    rng: np.random.Generator,
) -> List[ResidualSample]:
    samples: List[ResidualSample] = []
    perturbation_types = [
        PerturbationType.UNDERREACH_IDLE,
        PerturbationType.PREMATURE_CLOSE,
        PerturbationType.PREMATURE_OPEN,
        PerturbationType.LATERAL_DRIFT,
    ]

    for _ in range(perturbations_per_demo):
        perturbation_type = perturbation_types[int(rng.integers(len(perturbation_types)))]
        severity = float(rng.uniform(0.15, 0.95))
        seed = int(rng.integers(0, 2**31 - 1))
        result = perturbation_generator.apply_perturbation(trajectory, perturbation_type, severity=severity, seed=seed)
        if result is None:
            continue

        perturbed = result.perturbed_trajectory
        observations = perturbed["observations"]
        original_actions = np.asarray(trajectory["actions"], dtype=np.float32)
        perturbed_actions = np.asarray(perturbed["actions"], dtype=np.float32)
        length = min(infer_trajectory_length(perturbed), len(original_actions), len(perturbed_actions))
        if length < history_length + 1:
            continue

        original_actions = original_actions[:length]
        perturbed_actions = perturbed_actions[:length]
        state_matrix = build_state_matrix(observations, length)
        image_key = infer_image_key(observations.keys())

        steps = select_window_steps(length, result.perturbation_window, history_length, samples_per_perturbation, rng)
        perturb_start, perturb_end = result.perturbation_window
        recovery_tail = max(2, history_length // 3)

        for step in steps:
            if step < history_length - 1:
                continue

            state_history = state_matrix[step - history_length + 1 : step + 1]
            action_history = perturbed_actions[step - history_length + 1 : step + 1]
            image_history = extract_image_history(observations, image_key, step, history_length, synthetic_state_history=state_history)
            target_residual = original_actions[step] - perturbed_actions[step]
            gate_label = int(
                np.linalg.norm(target_residual) > gate_residual_threshold
                or (perturb_start <= step <= min(length - 1, perturb_end + recovery_tail))
            )

            samples.append(
                ResidualSample(
                    state_history=state_history,
                    action_history=action_history,
                    image_history=image_history,
                    target_residual=target_residual.astype(np.float32),
                    gate_label=gate_label,
                    perturbation_type=perturbation_type.value,
                    demo_key=demo_key,
                    timestep=int(step),
                )
            )

    return samples


def build_dataset(
    hdf5_path: Path,
    max_demos: int,
    history_length: int,
    clean_samples_per_demo: int,
    perturbations_per_demo: int,
    samples_per_perturbation: int,
    gate_residual_threshold: float,
    seed: int,
) -> Tuple[List[ResidualSample], Dict[str, int]]:
    rng = np.random.default_rng(seed)
    adapter = RobomimicAdapter()
    generator = PerturbationGenerator(rng=rng)

    demo_keys = list_demo_keys(hdf5_path, max_demos=max_demos)
    samples: List[ResidualSample] = []
    stats = {
        "demos": 0,
        "clean_samples": 0,
        "perturbed_samples": 0,
        "perturbed_rollouts": 0,
    }

    for demo_key in demo_keys:
        raw_traj = load_robomimic_demo(hdf5_path, demo_key)
        trajectory = adapter.load(raw_traj)
        clean = build_clean_samples(demo_key, trajectory, history_length, clean_samples_per_demo, rng)
        perturbed = build_perturbed_samples(
            demo_key,
            trajectory,
            generator,
            history_length,
            perturbations_per_demo,
            samples_per_perturbation,
            gate_residual_threshold,
            rng,
        )

        if not clean and not perturbed:
            continue

        stats["demos"] += 1
        stats["clean_samples"] += len(clean)
        stats["perturbed_samples"] += len(perturbed)
        stats["perturbed_rollouts"] += perturbations_per_demo
        samples.extend(clean)
        samples.extend(perturbed)

    return samples, stats


def tensor_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def weighted_gate_loss(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float) -> torch.Tensor:
    if pos_weight <= 0:
        pos_weight = 1.0
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=torch.tensor(pos_weight, device=logits.device))


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    gate_loss_weight: float,
    pos_weight: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    residual_losses: List[float] = []
    gate_losses: List[float] = []
    total_losses: List[float] = []
    gate_predictions: List[float] = []
    gate_targets: List[float] = []

    for batch in loader:
        batch = tensor_batch_to_device(batch, device)
        outputs = model(batch["state_history"], batch["action_history"], batch["image_history"])
        residual_loss = F.mse_loss(outputs["residuals"], batch["target_residual"])
        gate_loss = weighted_gate_loss(outputs["gate_logits"], batch["gate_label"], pos_weight=pos_weight)
        loss = residual_loss + gate_loss_weight * gate_loss

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        residual_losses.append(float(residual_loss.detach().cpu().item()))
        gate_losses.append(float(gate_loss.detach().cpu().item()))
        total_losses.append(float(loss.detach().cpu().item()))
        gate_predictions.extend(torch.sigmoid(outputs["gate_logits"]).detach().cpu().numpy().tolist())
        gate_targets.extend(batch["gate_label"].detach().cpu().numpy().tolist())

    preds = np.asarray(gate_predictions, dtype=np.float32) >= 0.5
    targets = np.asarray(gate_targets, dtype=np.float32) >= 0.5
    accuracy = float((preds == targets).mean()) if len(targets) else 0.0
    return {
        "loss": float(np.mean(total_losses)) if total_losses else 0.0,
        "residual_loss": float(np.mean(residual_losses)) if residual_losses else 0.0,
        "gate_loss": float(np.mean(gate_losses)) if gate_losses else 0.0,
        "gate_accuracy": accuracy,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an image-conditioned gated residual policy on perturbed Robomimic rollouts.")
    parser.add_argument(
        "--hdf5_path",
        type=str,
        default=str(PROJECT_ROOT / "robomimic" / "datasets" / "square" / "ph" / "image_v15.hdf5"),
    )
    parser.add_argument("--max_demos", type=int, default=40)
    parser.add_argument("--history_length", type=int, default=12)
    parser.add_argument("--clean_samples_per_demo", type=int, default=6)
    parser.add_argument("--perturbations_per_demo", type=int, default=2)
    parser.add_argument("--samples_per_perturbation", type=int, default=16)
    parser.add_argument("--gate_residual_threshold", type=float, default=0.03)
    parser.add_argument("--gate_loss_weight", type=float, default=0.5)
    parser.add_argument("--train_split", type=float, default=0.85)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default=str(REPO_ROOT / "notebooks" / "output" / "image_residual_policy"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hdf5_path = Path(args.hdf5_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"Robomimic image dataset not found: {hdf5_path}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {hdf5_path}")
    samples, stats = build_dataset(
        hdf5_path=hdf5_path,
        max_demos=args.max_demos,
        history_length=args.history_length,
        clean_samples_per_demo=args.clean_samples_per_demo,
        perturbations_per_demo=args.perturbations_per_demo,
        samples_per_perturbation=args.samples_per_perturbation,
        gate_residual_threshold=args.gate_residual_threshold,
        seed=args.seed,
    )
    if not samples:
        raise RuntimeError("No training samples were generated.")

    state_dim = int(samples[0].state_history.shape[-1])
    action_dim = int(samples[0].action_history.shape[-1])
    image_shape = tuple(samples[0].image_history.shape)

    print(f"Generated {len(samples)} samples from {stats['demos']} demos")
    print(f"State dim: {state_dim} | Action dim: {action_dim} | Image shape: {image_shape}")

    dataset = ResidualWindowDataset(samples)
    train_size = max(1, int(len(dataset) * float(args.train_split)))
    val_size = max(1, len(dataset) - train_size)
    if train_size + val_size > len(dataset):
        train_size = len(dataset) - val_size

    split_generator = torch.Generator().manual_seed(args.seed)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=split_generator)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ImageConditionedGatedResidualPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        history_length=args.history_length,
    ).to(device)

    gate_targets = np.asarray([sample.gate_label for sample in samples], dtype=np.float32)
    positive_count = float(gate_targets.sum())
    negative_count = float(len(gate_targets) - positive_count)
    pos_weight = negative_count / max(1.0, positive_count)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    best_val_loss = float("inf")
    best_epoch = -1
    history: List[Dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            gate_loss_weight=args.gate_loss_weight,
            pos_weight=pos_weight,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=None,
            device=device,
            gate_loss_weight=args.gate_loss_weight,
            pos_weight=pos_weight,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_residual_loss": train_metrics["residual_loss"],
            "train_gate_loss": train_metrics["gate_loss"],
            "train_gate_accuracy": train_metrics["gate_accuracy"],
            "val_loss": val_metrics["loss"],
            "val_residual_loss": val_metrics["residual_loss"],
            "val_gate_loss": val_metrics["gate_loss"],
            "val_gate_accuracy": val_metrics["gate_accuracy"],
        }
        history.append(row)

        print(
            f"epoch {epoch:02d} | train loss {train_metrics['loss']:.4f} "
            f"val loss {val_metrics['loss']:.4f} | val gate acc {val_metrics['gate_accuracy']:.3f}"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            checkpoint = {
                "model_class": "ImageConditionedGatedResidualPolicy",
                "state_dim": state_dim,
                "action_dim": action_dim,
                "history_length": args.history_length,
                "image_shape": list(image_shape),
                "state_keys": STATE_KEY_PRIORITY,
                "image_key_priority": IMAGE_KEY_PRIORITY,
                "train_args": vars(args),
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint, output_dir / "best_image_residual_policy.pth")

    summary = {
        "dataset_stats": stats,
        "sample_count": len(samples),
        "state_dim": state_dim,
        "action_dim": action_dim,
        "image_shape": list(image_shape),
        "positive_gate_fraction": float(positive_count / max(1.0, len(gate_targets))),
        "pos_weight": float(pos_weight),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "history": history,
    }

    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved checkpoint to {output_dir / 'best_image_residual_policy.pth'}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()