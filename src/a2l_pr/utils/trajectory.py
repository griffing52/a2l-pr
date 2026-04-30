import numpy as np
from typing import Dict, List, Tuple, Optional
import copy

class TrajectoryAnalyzer:
    """Analyze trajectories to detect key events and phases.
    
    Expects trajectory to be a dict with keys like 'observations' and 'actions'.
    """
    
    def __init__(self, trajectory: Dict, metadata: Optional[Dict] = None):
        self.trajectory = trajectory
        trajectory_metadata = trajectory.get('metadata', {}) if isinstance(trajectory, dict) else {}
        self.metadata = {**trajectory_metadata, **(metadata or {})}
        self._cache = {}

    def _schema_keys(self, name: str, defaults: List[str]) -> List[str]:
        value = self.metadata.get(name)
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return defaults

    def _schema_index(self, name: str, default: int = -1) -> int:
        value = self.metadata.get(name, default)
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_1d_signal(array: np.ndarray) -> np.ndarray:
        signal = np.asarray(array)
        if signal.ndim == 0:
            return signal.reshape(1)
        if signal.ndim == 1:
            return signal
        if signal.ndim >= 2 and signal.shape[-1] == 2:
            return signal[..., 0]
        return np.mean(signal, axis=-1)

    def _get_observation_count(self) -> int:
        obs = self.trajectory.get('observations', None)
        if isinstance(obs, list):
            return len(obs)
        if isinstance(obs, dict) and obs:
            for value in obs.values():
                array = np.asarray(value)
                if array.ndim >= 1:
                    return int(array.shape[0])
        actions = self.get_actions()
        if actions is not None and actions.size > 0:
            return int(actions.shape[0])
        return 0

    def _get_observation_value(self, key: str) -> Optional[np.ndarray]:
        obs = self.trajectory.get('observations', None)
        if isinstance(obs, list):
            if not obs or not isinstance(obs[0], dict) or key not in obs[0]:
                return None
            return np.asarray([o[key] for o in obs])
        if isinstance(obs, dict) and key in obs:
            return np.asarray(obs[key])
        return None
    
    def get_ee_position(self) -> np.ndarray:
        if 'ee_position' in self._cache:
            return self._cache['ee_position']
        
        for key in self._schema_keys('ee_keys', ['object-state', 'robot0_eef_pos', 'eef_pos', 'ee_pos', 'slave_ee_pos']):
            value = self._get_observation_value(key)
            if value is not None and value.size > 0:
                ee_pos = np.asarray(value)
                if ee_pos.ndim == 1:
                    ee_pos = ee_pos.reshape(-1, 1)
                ee_pos = ee_pos[:, :3]
                self._cache['ee_position'] = ee_pos
                return ee_pos
        
        return np.array([])
    
    def get_gripper_state(self) -> Optional[np.ndarray]:
        if 'gripper_state' in self._cache:
            return self._cache['gripper_state']
        
        for key in self._schema_keys('gripper_keys', ['gripper', 'robot0_gripper_qpos', 'gripper_qpos', 'gripper_pos', 'slave_gripper_angle']):
            value = self._get_observation_value(key)
            if value is not None and value.size > 0:
                gripper = np.asarray(value)
                if gripper.ndim == 1:
                    gripper = gripper.reshape(-1, 1)
                self._cache['gripper_state'] = gripper
                return gripper

        actions = self.get_actions()
        action_gripper_index = self._schema_index('gripper_action_index', -1)
        if actions is not None and actions.size > 0 and actions.ndim >= 2 and actions.shape[1] >= 1:
            if action_gripper_index < 0:
                action_gripper_index = actions.shape[1] + action_gripper_index
            if 0 <= action_gripper_index < actions.shape[1]:
                gripper = np.asarray(actions[:, action_gripper_index]).reshape(-1, 1)
                self._cache['gripper_state'] = gripper
                return gripper
        
        return None
    
    def get_actions(self) -> np.ndarray:
        if 'actions' in self._cache:
            return self._cache['actions']
        
        raw_actions = self.trajectory.get('actions', None)
        if raw_actions is None:
            actions = np.empty((0,))
        else:
            actions = np.asarray(raw_actions)

        if actions.ndim == 0:
            actions = np.empty((0,))

        self._cache['actions'] = actions
        return actions
    
    def find_max_reach(self) -> int:
        ee_pos = self.get_ee_position()
        if len(ee_pos) == 0:
            return self._get_observation_count() // 2
        reach = np.linalg.norm(ee_pos, axis=1)
        return int(np.argmax(reach))
    
    def find_gripper_transitions(self) -> List[Tuple[int, str]]:
        gripper = self.get_gripper_state()
        if gripper is None or len(gripper) < 2:
            return []
        
        gripper_norm = self._to_1d_signal(gripper).copy()
        if gripper_norm.min() != gripper_norm.max():
            gripper_norm = (gripper_norm - gripper_norm.min()) / (gripper_norm.max() - gripper_norm.min())

        gripper_1d = np.asarray(gripper_norm).reshape(-1)
        if len(gripper_1d) >= 5:
            kernel = np.ones(5, dtype=float) / 5.0
            gripper_1d = np.convolve(gripper_1d, kernel, mode='same')
        
        transitions = []
        open_threshold = float(self.metadata.get('gripper_open_threshold', 0.7))
        close_threshold = float(self.metadata.get('gripper_close_threshold', 0.3))
        
        # Determine initial state based on thresholds to be robust
        current_state = 'open' if gripper_1d[0] >= 0.5 else 'close'
        
        for t in range(1, len(gripper_1d)):
            prev_val = gripper_1d[t - 1]
            curr_val = gripper_1d[t]
            if current_state != 'open' and prev_val < open_threshold and curr_val >= open_threshold:
                transitions.append((t, 'open'))
                current_state = 'open'
            elif current_state != 'close' and prev_val > close_threshold and curr_val <= close_threshold:
                transitions.append((t, 'close'))
                current_state = 'close'
        
        return transitions

    def infer_gripper_event_step(self, event_type: str) -> Optional[int]:
        transitions = self.find_gripper_transitions()
        if event_type == 'close':
            close_steps = [t for t, direction in transitions if direction == 'close']
            if close_steps:
                return int(close_steps[0])
        elif event_type == 'open':
            open_steps = [t for t, direction in transitions if direction == 'open']
            if open_steps:
                return int(open_steps[-1])

        gripper = self.get_gripper_state()
        if gripper is not None and len(gripper) >= 2:
            gripper_1d = self._to_1d_signal(gripper)
            if gripper_1d.min() != gripper_1d.max():
                gripper_1d = (gripper_1d - gripper_1d.min()) / (gripper_1d.max() - gripper_1d.min())
            if len(gripper_1d) >= 5:
                kernel = np.ones(5, dtype=float) / 5.0
                gripper_1d = np.convolve(gripper_1d, kernel, mode='same')

            if event_type == 'close':
                candidate = np.where(gripper_1d <= 0.35)[0]
                if len(candidate) > 0:
                    return int(candidate[0])
            elif event_type == 'open':
                candidate = np.where(gripper_1d >= 0.65)[0]
                if len(candidate) > 0:
                    return int(candidate[-1])

        contact_start, contact_end = self.estimate_contact_phase()
        if event_type == 'close' and contact_start is not None:
            return int(contact_start)
        if event_type == 'open' and contact_end is not None:
            return int(contact_end)

        max_reach = self.find_max_reach()
        return int(max_reach)
    
    def estimate_contact_phase(self) -> Tuple[Optional[int], Optional[int]]:
        actions = self.get_actions()
        if actions is None or actions.size == 0:
            return None, None

        if actions.ndim == 1:
            actions = actions[:, None]
        
        window = max(5, len(actions) // 20)
        action_variance = np.array([
            np.mean(np.var(actions[max(0, t-window):t+window], axis=0))
            for t in range(len(actions))
        ])
        
        low_var_threshold = np.percentile(action_variance, 25)
        contact_steps = np.where(action_variance < low_var_threshold)[0]
        
        if len(contact_steps) > 0:
            start = int(contact_steps[0])
            end = int(contact_steps[-1])
            return start, end
        
        return None, None
    
    def get_trajectory_length(self) -> int:
        return self._get_observation_count()


class TrajectoryModifier:
    """Modify trajectories by applying perturbations and adjustments."""

    @staticmethod
    def _metadata(trajectory: Dict) -> Dict:
        metadata = trajectory.get('metadata', {}) if isinstance(trajectory, dict) else {}
        return metadata if isinstance(metadata, dict) else {}

    @staticmethod
    def _gripper_obs_keys(trajectory: Dict) -> List[str]:
        metadata = TrajectoryModifier._metadata(trajectory)
        value = metadata.get('gripper_keys')
        if isinstance(value, str):
            return [value]
        if isinstance(value, (list, tuple)):
            return [str(item) for item in value if item]
        return ['gripper', 'robot0_gripper_qpos', 'gripper_qpos', 'gripper_pos', 'slave_gripper_angle']

    @staticmethod
    def _gripper_action_index(trajectory: Dict, action_dim: Optional[int] = None) -> int:
        metadata = TrajectoryModifier._metadata(trajectory)
        value = metadata.get('gripper_action_index', -1)
        try:
            index = int(value)
        except (TypeError, ValueError):
            index = -1

        if action_dim is not None and index < 0:
            index = action_dim + index
        return index

    @staticmethod
    def _observation_length(obs) -> int:
        if isinstance(obs, list):
            return len(obs)
        if isinstance(obs, dict):
            for value in obs.values():
                array = np.asarray(value)
                if array.ndim >= 1:
                    return int(array.shape[0])
        return 0
    
    @staticmethod
    def perturb_ee_position(
        trajectory: Dict,
        perturbation_window: Tuple[int, int],
        offset: np.ndarray,
        mode: str = 'additive'
    ) -> Dict:
        traj_copy = copy.deepcopy(trajectory)
        obs = traj_copy.get('observations', [])
        if not obs:
            return traj_copy

        ee_key = None
        metadata = TrajectoryModifier._metadata(traj_copy)
        ee_keys = metadata.get('ee_keys')
        if isinstance(ee_keys, str):
            ee_key_candidates = [ee_keys]
        elif isinstance(ee_keys, (list, tuple)):
            ee_key_candidates = [str(item) for item in ee_keys if item]
        else:
            ee_key_candidates = ['object-state', 'robot0_eef_pos', 'eef_pos', 'ee_pos', 'slave_ee_pos']

        start, end = perturbation_window

        if isinstance(obs, dict):
            for key in ee_key_candidates:
                if key in obs:
                    ee_key = key
                    break

            if ee_key is None:
                return traj_copy

            array = np.asarray(obs[ee_key]).copy()
            if array.ndim < 2:
                return traj_copy

            start = max(0, int(start))
            end = min(array.shape[0] - 1, int(end))
            if end < start:
                return traj_copy

            if mode == 'additive':
                array[start:end + 1, :3] = array[start:end + 1, :3] + offset
            elif mode == 'replace':
                array[start:end + 1, :3] = offset

            obs[ee_key] = array
            return traj_copy

        if not isinstance(obs, list) or not obs:
            return traj_copy

        for key in ee_key_candidates:
            if isinstance(obs[0], dict) and key in obs[0]:
                ee_key = key
                break

        if ee_key is None:
            return traj_copy

        for t in range(start, min(end + 1, len(obs))):
            if ee_key in obs[t]:
                state = obs[t][ee_key]
                if mode == 'additive':
                    state[:3] = state[:3] + offset
                elif mode == 'replace':
                    state[:3] = offset
        
        return traj_copy

    @staticmethod
    def perturb_action_window(
        trajectory: Dict,
        perturbation_window: Tuple[int, int],
        delta: Optional[np.ndarray] = None,
        gripper_value: Optional[float] = None,
        action_indices: Optional[slice] = None,
    ) -> Dict:
        traj_copy = copy.deepcopy(trajectory)
        actions = traj_copy.get('actions', None)
        if actions is None:
            return traj_copy

        start, end = perturbation_window
        obs = traj_copy.get('observations', None)
        obs_length = TrajectoryModifier._observation_length(obs)
        if isinstance(actions, np.ndarray):
            if actions.size == 0:
                return traj_copy
            action_array = actions
            gripper_index = TrajectoryModifier._gripper_action_index(traj_copy, action_array.shape[1])
            start = max(0, int(start))
            end = min(len(action_array) - 1, int(end))
            if end < start:
                return traj_copy

            if delta is not None and action_array.ndim >= 2:
                indices = action_indices if action_indices is not None else slice(0, min(3, action_array.shape[1]))
                action_array[start:end + 1, indices] = action_array[start:end + 1, indices] + delta

            if gripper_value is not None and action_array.ndim >= 2 and 0 <= gripper_index < action_array.shape[1]:
                action_array[start:end + 1, gripper_index] = gripper_value

            traj_copy['actions'] = action_array
            return traj_copy

        if not isinstance(actions, list) or not actions:
            return traj_copy

        start = max(0, int(start))
        end = min(len(actions) - 1, int(end))
        if end < start:
            return traj_copy

        sample_action = np.asarray(actions[start])
        gripper_index = TrajectoryModifier._gripper_action_index(traj_copy, sample_action.shape[0] if sample_action.ndim >= 1 else None)

        for t in range(start, end + 1):
            action = np.asarray(actions[t]).copy()
            if delta is not None and action.ndim >= 1:
                indices = action_indices if action_indices is not None else slice(0, min(3, action.shape[0]))
                action[indices] = action[indices] + delta
            if gripper_value is not None and action.ndim >= 1 and 0 <= gripper_index < action.shape[0]:
                action[gripper_index] = gripper_value
            actions[t] = action

        traj_copy['actions'] = actions
        return traj_copy

    @staticmethod
    def shift_gripper_event(
        trajectory: Dict,
        current_step: int,
        shift_steps: int,
    ) -> Dict:
        traj_copy = copy.deepcopy(trajectory)
        obs = traj_copy.get('observations', [])
        actions = traj_copy.get('actions', [])

        if not obs:
            return traj_copy
        
        gripper_key = None
        gripper_keys = TrajectoryModifier._gripper_obs_keys(traj_copy)

        if isinstance(obs, dict):
            for key in gripper_keys:
                if key in obs:
                    gripper_key = key
                    break
        else:
            for key in gripper_keys:
                if isinstance(obs[0], dict) and key in obs[0]:
                    gripper_key = key
                    break

        if gripper_key is None:
            gripper_seq = None
        else:
            if isinstance(obs, dict):
                gripper_array = np.asarray(obs[gripper_key])
                gripper_seq = gripper_array.mean(axis=1) if gripper_array.ndim > 1 else np.asarray(gripper_array)
            else:
                gripper_seq = np.array([o.get(gripper_key, [0])[0] for o in obs])

        action_seq = None
        if isinstance(actions, np.ndarray) and actions.ndim >= 2 and actions.shape[1] >= 1:
            action_seq = np.array(actions[:, -1])
        elif isinstance(actions, list) and actions and np.asarray(actions[0]).ndim >= 1:
            action_seq = np.array([np.asarray(a)[-1] for a in actions])

        if gripper_seq is None and action_seq is None:
            return traj_copy
        
        if gripper_seq is not None:
            new_gripper_seq = np.roll(gripper_seq, shift_steps)
            if isinstance(obs, dict):
                gripper_array = np.asarray(obs[gripper_key]).copy()
                if gripper_array.ndim == 1:
                    gripper_array[:] = new_gripper_seq
                else:
                    gripper_array[:] = np.expand_dims(new_gripper_seq, axis=-1)
                obs[gripper_key] = gripper_array
            else:
                for t, val in enumerate(new_gripper_seq):
                    obs[t][gripper_key] = np.array([val])

        if action_seq is not None:
            new_action_seq = np.roll(action_seq, shift_steps)
            if isinstance(actions, np.ndarray):
                actions[:, -1] = new_action_seq
            else:
                for t, val in enumerate(new_action_seq):
                    action = np.asarray(actions[t]).copy()
                    action[-1] = val
                    actions[t] = action

        traj_copy['actions'] = actions
        traj_copy['observations'] = obs
        
        return traj_copy

    @staticmethod
    def insert_pause(
        trajectory: Dict,
        pause_start: int,
        pause_duration: int,
    ) -> Dict:
        traj_copy = copy.deepcopy(trajectory)
        obs = traj_copy.get('observations', [])
        actions = traj_copy.get('actions', [])

        if isinstance(obs, dict):
            obs_length = TrajectoryModifier._observation_length(obs)
            if obs_length == 0 or pause_start >= obs_length:
                return traj_copy

            pause_start = max(0, int(pause_start))
            pause_duration = max(0, int(pause_duration))

            for key, value in list(obs.items()):
                array = np.asarray(value)
                if array.ndim < 1 or array.shape[0] != obs_length:
                    continue
                repeated = np.repeat(array[pause_start:pause_start + 1], pause_duration, axis=0)
                obs[key] = np.concatenate([array[:pause_start], repeated, array[pause_start:]], axis=0)

            if isinstance(actions, np.ndarray) and actions.ndim >= 1 and len(actions) >= obs_length:
                zero_action = np.zeros_like(actions[pause_start])
                actions = np.concatenate([
                    actions[:pause_start],
                    np.repeat(zero_action[None, ...], pause_duration, axis=0),
                    actions[pause_start:]
                ], axis=0)
            elif isinstance(actions, list) and actions and len(actions) >= obs_length:
                pause_action = np.zeros_like(np.asarray(actions[pause_start]))
                for _ in range(pause_duration):
                    actions.insert(pause_start, pause_action.copy())

            traj_copy['observations'] = obs
            traj_copy['actions'] = actions
            return traj_copy

        if not isinstance(obs, list) or not obs:
            return traj_copy

        if pause_start >= len(obs):
            return traj_copy

        pause_obs = copy.deepcopy(obs[pause_start])
        if pause_start < len(actions):
            pause_action = np.zeros_like(actions[pause_start])
        else:
            pause_action = np.zeros(actions[0].shape if len(actions) > 0 else 1)

        for _ in range(pause_duration):
            obs.insert(pause_start, copy.deepcopy(pause_obs))
            if pause_start < len(actions):
                actions.insert(pause_start, pause_action.copy())

        traj_copy['observations'] = obs
        traj_copy['actions'] = actions

        return traj_copy
