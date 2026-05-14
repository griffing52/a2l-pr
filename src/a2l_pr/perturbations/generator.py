import numpy as np
from typing import Dict, Tuple, Optional, List
import copy
from dataclasses import dataclass
from enum import Enum

from a2l_pr.utils.trajectory import TrajectoryAnalyzer, TrajectoryModifier

class PerturbationType(Enum):
    """Enumeration of perturbation types."""
    UNDERREACH_IDLE = "underreach_idle_before_max_reach"
    PREMATURE_CLOSE = "premature_gripper_close"
    PREMATURE_OPEN = "premature_gripper_open"
    LATERAL_DRIFT = "lateral_drift"

@dataclass
class PerturbationResult:
    """Result of applying a perturbation to a trajectory."""
    perturbed_trajectory: Dict
    perturbation_type: PerturbationType
    perturbation_window: Tuple[int, int]
    severity: float  # 0.0 to 1.0
    parameters: Dict  # parameters used for this perturbation
    recovery_text: str
    theoretical_failure_mode: str

class PerturbationGenerator:
    """Generate synthetic perturbations for trajectories."""
    
    def __init__(self, rng: Optional[np.random.Generator] = None):
        """
        Args:
            rng: numpy random generator; if None, creates new one
        """
        self.rng = rng or np.random.default_rng()
        self.modifier = TrajectoryModifier()

    @staticmethod
    def _clamp_window(start: int, end: int, traj_len: int, min_width: int = 3) -> Tuple[int, int]:
        start = max(0, int(start))
        end = min(traj_len - 1, int(end))
        if end - start + 1 < min_width:
            end = min(traj_len - 1, start + min_width - 1)
        return start, max(start, end)

    @staticmethod
    def _phase_landmarks(analyzer: TrajectoryAnalyzer) -> Dict[str, Optional[int]]:
        transitions = analyzer.find_gripper_transitions()
        close_steps = [t for t, direction in transitions if direction == 'close']
        open_steps = [t for t, direction in transitions if direction == 'open']
        contact_start, contact_end = analyzer.estimate_contact_phase()

        return {
            'first_close': close_steps[0] if close_steps else None,
            'last_open': open_steps[-1] if open_steps else None,
            'contact_start': contact_start,
            'contact_end': contact_end,
        }
    def _overwrite_with_pause(self, trajectory: Dict, start: int, end: int, gripper_val=None, gripper_idx=-1) -> Dict:
        traj_copy = copy.deepcopy(trajectory)
        start = max(0, start)
        end = min(len(traj_copy.get('actions', [])) - 1, end)
        if start >= end:
            return traj_copy
        
        if isinstance(traj_copy['actions'], np.ndarray):
            traj_copy['actions'][start:end+1] = 0.0
            if gripper_val is not None and gripper_idx >= 0:
                traj_copy['actions'][start:end+1, gripper_idx] = gripper_val
        elif isinstance(traj_copy['actions'], list):
            for i in range(start, end+1):
                traj_copy['actions'][i] = np.zeros_like(traj_copy['actions'][i])
                if gripper_val is not None and gripper_idx >= 0:
                    traj_copy['actions'][i][gripper_idx] = gripper_val
                
        obs = traj_copy.get('observations', {})
        if isinstance(obs, dict):
            for k, v in obs.items():
                if isinstance(v, np.ndarray) and len(v) > start:
                    freeze_val = v[start].copy()
                    traj_copy['observations'][k][start:end+1] = freeze_val
        elif isinstance(obs, list):
            for i in range(start, end+1):
                traj_copy['observations'][i] = copy.deepcopy(traj_copy['observations'][start])
                
        return traj_copy

    def apply_perturbation(
        self,
        trajectory: Dict,
        perturbation_type: PerturbationType,
        severity: float = 0.5,
        seed: Optional[int] = None
    ) -> Optional[PerturbationResult]:
        """Apply a perturbation to a trajectory.
        
        Args:
            trajectory: trajectory dict
            perturbation_type: which perturbation to apply
            severity: 0.0-1.0 severity scaling
            seed: optional seed for reproducibility
        
        Returns:
            PerturbationResult or None if perturbation not applicable
        """
        if seed is not None:
            rng = np.random.default_rng(seed)
        else:
            rng = self.rng
        
        analyzer = TrajectoryAnalyzer(trajectory)
        traj_len = analyzer.get_trajectory_length()
        
        if traj_len < 10:
            return None  # Trajectory too short
        
        if perturbation_type == PerturbationType.UNDERREACH_IDLE:
            return self._apply_underreach_idle(trajectory, analyzer, severity, rng)
        elif perturbation_type == PerturbationType.PREMATURE_CLOSE:
            return self._apply_premature_close(trajectory, analyzer, severity, rng)
        elif perturbation_type == PerturbationType.PREMATURE_OPEN:
            return self._apply_premature_open(trajectory, analyzer, severity, rng)
        elif perturbation_type == PerturbationType.LATERAL_DRIFT:
            return self._apply_lateral_drift(trajectory, analyzer, severity, rng)
        
        return None
    
    def _apply_underreach_idle(
        self, trajectory: Dict, analyzer: TrajectoryAnalyzer, severity: float, rng
    ) -> Optional[PerturbationResult]:
        """A. Under-reach idle before max reach."""
        max_reach_step = analyzer.find_max_reach()
        traj_len = analyzer.get_trajectory_length()
        landmarks = self._phase_landmarks(analyzer)

        first_close = landmarks['first_close']
        last_open = landmarks['last_open']

        anchor_candidates: List[Tuple[str, int]] = []
        if first_close is not None and first_close > 3:
            anchor_candidates.append(('pre_grasp', int(first_close)))

        if last_open is not None and last_open > 3:
            late_release = last_open >= int(0.60 * traj_len)
            after_grasp = first_close is not None and (last_open - first_close) >= 4
            if late_release or after_grasp:
                anchor_candidates.append(('pre_release', int(last_open)))
        if max_reach_step is not None and max_reach_step > 3:
            anchor_candidates.append(('max_reach', int(max_reach_step)))

        if not anchor_candidates:
            return None

        preference = str(analyzer.metadata.get('underreach_anchor_preference', 'auto')).lower()
        anchor_name, anchor_step = anchor_candidates[0]

        if preference in {'pre_grasp', 'pre_release', 'max_reach'}:
            selected = [(name, step) for name, step in anchor_candidates if name == preference]
            if selected:
                anchor_name, anchor_step = selected[0]
        elif preference == 'auto':
            weighted = []
            for name, step in anchor_candidates:
                weight = 1.0
                if name == 'pre_grasp':
                    weight = 2.5
                elif name == 'pre_release':
                    weight = 2.0
                elif name == 'max_reach':
                    weight = 1.2
                weighted.append((name, step, weight))
            weights = np.array([w for _, _, w in weighted], dtype=float)
            weights = weights / np.sum(weights)
            pick = int(rng.choice(len(weighted), p=weights))
            anchor_name, anchor_step = weighted[pick][0], weighted[pick][1]

        back_span = max(6, int(traj_len * (0.08 + 0.10 * severity)))
        approach_start = max(0, int(anchor_step) - back_span)
        approach_end = max(approach_start + 2, int(anchor_step) - 1)
        approach_start, approach_end = self._clamp_window(approach_start, approach_end, traj_len)
        
        if approach_end - approach_start < 3:
            return None
        
        stop_short_distance_m = 0.03 + severity * 0.12
        idle_base_steps = int(analyzer.metadata.get('underreach_idle_base_steps', 14))
        idle_extra_steps = int(analyzer.metadata.get('underreach_idle_extra_steps', 40))
        idle_steps = int(idle_base_steps + severity * idle_extra_steps)
        max_idle_ratio = float(analyzer.metadata.get('underreach_idle_max_ratio', 0.45))
        idle_steps = max(idle_steps, int(traj_len * (0.12 + 0.15 * severity)))
        idle_steps = min(idle_steps, max(8, int(traj_len * max_idle_ratio)))
        forward_nudge_distance_m = 0.02 + severity * 0.06
        speed_scale = 0.2 + (1 - severity) * 0.3
        
        anchor_gap_steps = int(analyzer.metadata.get('underreach_anchor_gap_steps', max(2, traj_len // 40)))
        perturb_step = int(anchor_step) - anchor_gap_steps
        perturb_step = max(approach_start, min(perturb_step, approach_end - 1))
        
        # We want to pause from perturb_step until after it would have grasped
        first_close = landmarks['first_close']
        if first_close is None:
            first_close = int(anchor_step) + 5
            
        # skip grasping and go straight to lifting
        resume_step = min(traj_len - 1, int(first_close) + 8) 
        
        perturbed = self._overwrite_with_pause(trajectory, perturb_step, resume_step)

        failure_mode = {
            'pre_grasp': "Stopping short before grasp so the gripper cannot properly acquire the object.",
            'pre_release': "Stopping short before release so placement alignment is missed.",
            'max_reach': "Stalling at max reach near placement, causing handoff / placement failure.",
        }.get(anchor_name, "Hesitation or controller stall near a key manipulation phase.")
        
        params = {
            "perturbation_type": PerturbationType.UNDERREACH_IDLE.value,
            "anchor_type": anchor_name,
            "anchor_step": int(anchor_step),
            "stop_short_distance_m": stop_short_distance_m,
            "idle_steps": idle_steps,
            "forward_nudge_distance_m": forward_nudge_distance_m,
            "speed_scale": speed_scale,
        }
        
        return PerturbationResult(
            perturbed_trajectory=perturbed,
            perturbation_type=PerturbationType.UNDERREACH_IDLE,
            perturbation_window=(perturb_step, perturb_step + idle_steps),
            severity=severity,
            parameters=params,
            recovery_text="Just a little further, then continue the original motion.",
            theoretical_failure_mode=failure_mode
        )
    
    def _apply_premature_close(
        self, trajectory: Dict, analyzer: TrajectoryAnalyzer, severity: float, rng
    ) -> Optional[PerturbationResult]:
        """B. Premature gripper close."""
        landmarks = self._phase_landmarks(analyzer)
        close_step = analyzer.infer_gripper_event_step('close')
        if close_step is None:
            return None

        traj_len = analyzer.get_trajectory_length()

        if close_step <= 2:
            if landmarks['contact_start'] is not None:
                close_step = int(landmarks['contact_start'])
            else:
                close_step = max(3, int(traj_len * 0.30))
        
        shift_steps = -max(12, int((traj_len * 0.18) + severity * traj_len * 0.35))
        new_close_step = max(0, close_step + shift_steps)
        
        if new_close_step == close_step:
            return None
        
        xy_tolerance_m = 0.03 + severity * 0.06
        max_align_steps = 18 + int(severity * 14)
        
        params = {
            "perturbation_type": PerturbationType.PREMATURE_CLOSE.value,
            "shift_steps": shift_steps,
            "inferred_close_step": int(close_step),
            "xy_tolerance_m": xy_tolerance_m,
            "max_align_steps": max_align_steps,
        }
        
        gripper_state = analyzer.get_gripper_state()
        closed_val = 1.0
        if gripper_state is not None and len(gripper_state) > close_step:
             closed_val = float(np.max(gripper_state))
             
        # Find the gripper index
        actions = trajectory.get('actions', [])
        gripper_idx = self.modifier._gripper_action_index(trajectory, len(actions[0]) if len(actions)>0 else None)
        
        # Overwrite the downward reach with a pause but keep the gripper closed!
        # It resumes shortly after the original close step, so it lifts empty handed.
        resume_step = min(traj_len - 1, close_step + 5)
        perturbed = self._overwrite_with_pause(
            trajectory, 
            new_close_step, 
            resume_step, 
            gripper_val=closed_val, 
            gripper_idx=gripper_idx
        )
        
        return PerturbationResult(
            perturbed_trajectory=perturbed,
            perturbation_type=PerturbationType.PREMATURE_CLOSE,
            perturbation_window=(new_close_step, close_step),
            severity=severity,
            parameters=params,
            recovery_text="Open the gripper, align, then close again at the grasp point.",
            theoretical_failure_mode="Closing before object alignment, causing failed grasp."
        )
    
    def _apply_premature_open(
        self, trajectory: Dict, analyzer: TrajectoryAnalyzer, severity: float, rng
    ) -> Optional[PerturbationResult]:
        """C. Premature gripper open."""
        landmarks = self._phase_landmarks(analyzer)
        open_step = analyzer.infer_gripper_event_step('open')
        if open_step is None:
            return None
        
        traj_len = analyzer.get_trajectory_length()
        shift_steps = -max(3, int((traj_len * 0.06) + severity * traj_len * 0.14))
        new_open_step = max(0, open_step + shift_steps)
        
        if new_open_step == open_step:
            return None
        
        lift_distance_m = 0.02 + severity * 0.05
        speed_scale = 0.2 + (1 - severity) * 0.3
        
        params = {
            "perturbation_type": PerturbationType.PREMATURE_OPEN.value,
            "shift_steps": shift_steps,
            "inferred_open_step": int(open_step),
            "lift_distance_m": lift_distance_m,
            "speed_scale": speed_scale,
        }
        
        perturbed = self.modifier.shift_gripper_event(trajectory, open_step, shift_steps)
        open_hold = max(3, int(4 + severity * 5))
        
        gripper_state = analyzer.get_gripper_state()
        open_val = 0.0
        if gripper_state is not None and len(gripper_state) > open_step:
             open_val = float(np.min(gripper_state))

        perturbed = self.modifier.perturb_action_window(
            perturbed,
            (new_open_step, min(traj_len - 1, new_open_step + open_hold)),
            gripper_value=open_val,
        )
        
        return PerturbationResult(
            perturbed_trajectory=perturbed,
            perturbation_type=PerturbationType.PREMATURE_OPEN,
            perturbation_window=(new_open_step, open_step),
            severity=severity,
            parameters=params,
            recovery_text="Re-close gripper, lift slightly, and continue to placement.",
            theoretical_failure_mode="Dropping/releasing too early."
        )
    
    def _apply_lateral_drift(
        self, trajectory: Dict, analyzer: TrajectoryAnalyzer, severity: float, rng
    ) -> Optional[PerturbationResult]:
        """D. Lateral drift."""
        ee_pos = analyzer.get_ee_position()
        if len(ee_pos) < 10:
            return None
        
        traj_len = analyzer.get_trajectory_length()
        landmarks = self._phase_landmarks(analyzer)
        
        # Determine the "major event" (grasp/contact)
        major_event_step = landmarks['first_close'] or landmarks['contact_start']
        if major_event_step is None:
            major_event_step = int(traj_len * 0.5)
            
        # Randomize the start of the drift to happen somewhere before the major event
        # (give it at least 5% into the trajectory before drifting)
        min_start = max(0, int(traj_len * 0.05))
        max_start = max(min_start + 1, int(major_event_step) - 5)
        
        perturb_start = int(rng.integers(min_start, max_start + 1))
        
        # The drift needs to persist through the major event so the grasp fails.
        # We can end it slightly after the major event or let it last a random duration 
        # as long as it covers the major event.
        min_end = int(major_event_step) + int(traj_len * 0.05)
        max_end = traj_len - 1
        perturb_end = min(max_end, max(min_end, perturb_start + int(traj_len * 0.30)))
        
        perturb_start, perturb_end = self._clamp_window(perturb_start, perturb_end, traj_len)
        
        drift_base = float(analyzer.metadata.get('lateral_drift_base_m', 0.050))
        drift_extra = float(analyzer.metadata.get('lateral_drift_extra_m', 0.080))
        max_drift = drift_base + severity * drift_extra
        lateral_offset = np.array([
            rng.uniform(-max_drift, max_drift),
            rng.uniform(-max_drift, max_drift),
            0.0
        ])
        
        perturbed = copy.deepcopy(trajectory)
        obs = perturbed.get('observations', {})
        ee_key = None
        for k in ['object-state', 'robot0_eef_pos', 'eef_pos', 'ee_pos', 'slave_ee_pos']:
            if k in obs:
                ee_key = k
                break
        
        if ee_key and isinstance(obs[ee_key], np.ndarray):
            ee_array = obs[ee_key].copy()
            drift_duration = perturb_end - perturb_start + 1
            if drift_duration > 0:
                ramp = np.linspace(0, 1, drift_duration)[:, np.newaxis] * lateral_offset[:3]
                ee_array[perturb_start:perturb_end+1, :3] += ramp
                if perturb_end + 1 < traj_len:
                    ee_array[perturb_end+1:, :3] += lateral_offset[:3]
            perturbed['observations'][ee_key] = ee_array
        perturbed = self.modifier.perturb_action_window(
            perturbed,
            (max(0, perturb_start - 2), min(traj_len - 1, perturb_end + 2)),
            delta=lateral_offset[:3] * float(analyzer.metadata.get('lateral_drift_action_gain', (1.0 + 0.8 * severity))),
            action_indices=slice(0, 3)
        )
        
        xy_tolerance_m = 0.02 + severity * 0.05
        speed_scale = 0.25 + (1 - severity) * 0.25
        
        params = {
            "perturbation_type": PerturbationType.LATERAL_DRIFT.value,
            "lateral_offset_xy": lateral_offset[:2].tolist(),
            "xy_tolerance_m": xy_tolerance_m,
            "speed_scale": speed_scale,
        }
        
        return PerturbationResult(
            perturbed_trajectory=perturbed,
            perturbation_type=PerturbationType.LATERAL_DRIFT,
            perturbation_window=(perturb_start, perturb_end),
            severity=severity,
            parameters=params,
            recovery_text="Re-center over the target and continue.",
            theoretical_failure_mode="Calibration drift or lateral disturbance that misses alignment."
        )
