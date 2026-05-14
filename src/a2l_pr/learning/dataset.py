import torch
from torch.utils.data import Dataset
import numpy as np

class FailureRecoveryDataset(Dataset):
    """
    Dataset for learning to recognize failures and predict recoveries.
    
    Expects data elements that contain:
    - frames: A sequence of images (e.g., from the last 3 time steps).
    - actions: History of actions (e.g., last 2 actions).
    - failure_type_id: Integer label (0 for 'no failure', 1 for 'drop', etc.).
    - fsm_id: Integer label for the recovery template.
    - recovery_params: Float vector for the recovery parameters.
    """
    def __init__(self, data_records, transform=None):
        """
        Args:
            data_records: List of dictionaries, each containing:
                - 'frames': numpy array (N, C, H, W) where N is number of frames (e.g. 3)
                - 'actions': numpy array (M, action_dim)
                - 'failure_type_id': int
                - 'fsm_id': int
                - 'recovery_params': numpy array (size of recovery params)
            transform: Optional torchvision transforms.
        """
        self.data_records = data_records
        self.transform = transform

    def __len__(self):
        return len(self.data_records)

    def __getitem__(self, idx):
        record = self.data_records[idx]
        
        # 1. Frames or temporal feature vector
        frames = record['frames']
        if isinstance(frames, np.ndarray):
            frames = torch.from_numpy(frames).float()
            
        if self.transform:
            frames = self.transform(frames)
        
        if frames.ndim == 4:
            N, C, H, W = frames.shape
            frames_out = frames.reshape(N * C, H, W)
        else:
            frames_out = frames.reshape(-1)
        
        # 2. Actions
        actions = record.get('actions', None)
        if actions is not None:
            if isinstance(actions, np.ndarray):
                actions = torch.from_numpy(actions).float()
                
        # 3. Targets
        failure_type = torch.tensor(record['failure_type_id'], dtype=torch.long)
        fsm_id = torch.tensor(record.get('fsm_id', 0), dtype=torch.long)
        
        # if 'no failure', recovery_params might be empty or zero
        rec_params = record.get('recovery_params', np.zeros(7))
        recovery_params = torch.tensor(rec_params, dtype=torch.float32)
        
        sample = {
            'frames': frames_out,
            'failure_type': failure_type,
            'fsm_id': fsm_id,
            'recovery_params': recovery_params
        }
        
        if actions is not None:
            sample['actions'] = actions
            
        return sample
