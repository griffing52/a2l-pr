import pandas as pd
import numpy as np
from typing import Dict, Any
from a2l_pr.adapters.base import DataAdapter

class PiperAdapter(DataAdapter):
    """Adapter for AgileX Piper trajectory CSV files."""
    
    def load(self, source: str) -> Dict:
        """Load a Piper actions.csv file into standard format.
        
        Args:
            source: Path to the actions.csv file
            
        Returns:
            Standard trajectory dictionary
        """
        df = pd.read_csv(source)
        
        # Extract relevant columns
        # Piper Cartesian coordinates are typically in micrometers.
        # We scale them to standard meters (1e-6) so that perturbation thresholds work properly.
        ee_pos = df[['slave_ee_x', 'slave_ee_y', 'slave_ee_z']].values.astype(np.float32) * 1e-6
        gripper = df[['slave_gripper_angle']].values.astype(np.float32)
        
        # Actions might be the joints or we can just treat the state as the action for demonstration
        actions = df[['slave_j1', 'slave_j2', 'slave_j3', 'slave_j4', 'slave_j5', 'slave_j6', 'slave_gripper_angle']].values.astype(np.float32)
        
        trajectory = {
            'observations': {
                'slave_ee_pos': ee_pos,
                'slave_gripper_angle': gripper,
                'original_df': df.to_dict('list') # Store original data to reconstruct it later
            },
            'actions': actions,
            'metadata': {
                'dataset_type': 'piper_csv',
                'ee_keys': ['slave_ee_pos'],
                'gripper_keys': ['slave_gripper_angle'],
                'gripper_action_index': -1,
                # Adjust thresholds for piper if needed (piper gripper might not be 0 to 1)
                # Assuming piper gripper is around 0 to some max. We will need to normalize in the analyzer.
            }
        }
        return trajectory
        
    def save(self, trajectory: Dict, destination: str) -> None:
        """Save standard trajectory dictionary back to a Piper CSV format.
        
        Args:
            trajectory: Standard trajectory dictionary
            destination: Path to save the new CSV
        """
        # Reconstruct DataFrame from original_df to keep all other columns intact
        original_data = trajectory['observations'].get('original_df')
        if original_data is None:
            raise ValueError("Trajectory does not contain original Piper dataframe information for saving.")
            
        df = pd.DataFrame(original_data)
        
        # Overwrite with potentially modified values
        ee_pos = trajectory['observations'].get('slave_ee_pos')
        gripper = trajectory['observations'].get('slave_gripper_angle')
        actions = trajectory.get('actions')
        
        length = min(len(df), len(ee_pos))
        
        if ee_pos is not None:
            # Scale back from meters to micrometers
            df.loc[:length-1, ['slave_ee_x', 'slave_ee_y', 'slave_ee_z']] = ee_pos[:length] * 1e6
            
        if gripper is not None:
            df.loc[:length-1, ['slave_gripper_angle']] = gripper[:length]
            
        if actions is not None:
            # Assuming actions match the joint/gripper columns we extracted
            df.loc[:length-1, ['slave_j1', 'slave_j2', 'slave_j3', 'slave_j4', 'slave_j5', 'slave_j6', 'slave_gripper_angle']] = actions[:length]
            
        # If the trajectory was shortened or lengthened, truncate/pad the dataframe
        df = df.iloc[:length]
            
        df.to_csv(destination, index=False)
