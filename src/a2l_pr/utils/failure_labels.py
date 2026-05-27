"""Failure label helpers for residual-policy diagnostics and overlays."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from a2l_pr.perturbations.generator import PerturbationType


def _is_no_failure_label(label: str) -> bool:
    text = str(label).strip().lower()
    return text in {"no_failure", "none", "normal", "no failure"}


def default_failure_id_to_type(num_failure_types: int = 5) -> Dict[int, str]:
    # Keep a canonical ordering aligned with the existing perturbation enum.
    labels = [
        "no_failure",
        PerturbationType.UNDERREACH_IDLE.value,
        PerturbationType.PREMATURE_CLOSE.value,
        PerturbationType.PREMATURE_OPEN.value,
        PerturbationType.LATERAL_DRIFT.value,
    ]
    out: Dict[int, str] = {}
    for idx in range(max(1, int(num_failure_types))):
        if idx < len(labels):
            out[idx] = labels[idx]
        else:
            out[idx] = f"failure_{idx}"
    return out


def merge_failure_id_to_type(
    user_mapping: Optional[Dict[int, str]],
    num_failure_types: int,
) -> Dict[int, str]:
    merged = default_failure_id_to_type(num_failure_types=num_failure_types)
    if not user_mapping:
        return merged
    for key, value in user_mapping.items():
        merged[int(key)] = str(value)
    return merged


def select_failure_label(
    failure_probs: np.ndarray,
    failure_id_to_type: Dict[int, str],
    prefer_non_no_failure: bool = False,
) -> Dict[str, object]:
    probs = np.asarray(failure_probs, dtype=float).reshape(-1)
    if probs.size == 0:
        return {"id": 0, "label": "no_failure", "confidence": 0.0}

    top_id = int(np.argmax(probs))
    chosen_id = top_id

    if prefer_non_no_failure:
        non_no_failure_ids = []
        for idx in range(probs.size):
            label = failure_id_to_type.get(idx, f"failure_{idx}")
            if not _is_no_failure_label(label):
                non_no_failure_ids.append(idx)

        top_label = failure_id_to_type.get(top_id, f"failure_{top_id}")
        if _is_no_failure_label(top_label) and non_no_failure_ids:
            chosen_id = int(max(non_no_failure_ids, key=lambda i: probs[i]))

    chosen_label = failure_id_to_type.get(chosen_id, f"failure_{chosen_id}")
    chosen_conf = float(probs[chosen_id])
    return {"id": chosen_id, "label": chosen_label, "confidence": chosen_conf}
