# %% [markdown]
# # Piper Perturbation Demo
# This script demonstrates how to load an AgileX Piper trajectory, apply perturbations
# using the `a2l-pr` package, and visualize the results.

# %%
import os
import sys

# Ensure the a2l-pr package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from a2l_pr.adapters.piper import PiperAdapter
from a2l_pr.perturbations.generator import PerturbationGenerator, PerturbationType
from a2l_pr.recovery.generator import RecoveryCodeGenerator, recovery_to_json
from a2l_pr.utils.visualization import visualize_trajectory_comparison, visualize_trajectory_comparison_3d
from a2l_pr.config.config import ConfigManager

# %% [markdown]
# ## Configuration and Loading

# %%
# Load configuration
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "perturbation_config.yaml"))
config_mgr = ConfigManager.from_yaml(config_path)

# Path to the Piper actions.csv file (Update this path if necessary)
piper_csv_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "agilex_data_collection", "pick_bag_joe", "episode_000", "actions.csv"
))

if not os.path.exists(piper_csv_path):
    print(f"File not found: {piper_csv_path}")
    print("Please update the path to a valid Piper actions.csv file.")
else:
    # Initialize the Piper adapter and load the trajectory
    adapter = PiperAdapter()
    trajectory = adapter.load(piper_csv_path)

    # Inject dataset defaults from config
    dataset_defaults = config_mgr.get_dataset_defaults(trajectory['metadata']['dataset_type'])
    trajectory['metadata'].update(dataset_defaults)
    trajectory['metadata'].update(config_mgr.get_perturbation_config())

    print(f"Loaded Piper trajectory with {len(trajectory['actions'])} steps.")

# %% [markdown]
# ## Apply Perturbation

# %%
if os.path.exists(piper_csv_path):
    # Initialize generator
    generator = PerturbationGenerator()

    # Apply a Premature Close perturbation
    print("Applying Premature Close Perturbation...")
    result = generator.apply_perturbation(
        trajectory, 
        PerturbationType.PREMATURE_CLOSE, 
        severity=0.6,
        seed=42
    )

    if result is not None:
        print(f"Success! Perturbation applied: {result.perturbation_type.name}")
        print(f"Failure Mode: {result.theoretical_failure_mode}")
        
        # Visualize the comparison
        visualize_trajectory_comparison(
            trajectory, 
            result.perturbed_trajectory, 
            title="Piper Trajectory: Original vs Premature Close"
        )
        visualize_trajectory_comparison_3d(
            trajectory, 
            result.perturbed_trajectory, 
            title="3D Piper Trajectory: Original vs Premature Close"
        )
    else:
        print("Perturbation not applicable for this trajectory.")

# %% [markdown]
# ## Generate Recovery FSM

# %%
if os.path.exists(piper_csv_path) and result is not None:
    print("\nGenerating Recovery FSM DSL...")
    recovery_code = RecoveryCodeGenerator.generate_recovery_code(result)
    
    # Pretty print the JSON
    print(recovery_to_json(recovery_code))

# %% [markdown]
# ## Apply Lateral Drift Perturbation

# %%
if os.path.exists(piper_csv_path):
    print("\nApplying Lateral Drift Perturbation...")
    result_drift = generator.apply_perturbation(
        trajectory, 
        PerturbationType.LATERAL_DRIFT, 
        severity=0.8,
        seed=100
    )

    if result_drift is not None:
        visualize_trajectory_comparison(
            trajectory, 
            result_drift.perturbed_trajectory, 
            title="Piper Trajectory: Original vs Lateral Drift"
        )
        visualize_trajectory_comparison_3d(
            trajectory, 
            result_drift.perturbed_trajectory, 
            title="3D Piper Trajectory: Original vs Lateral Drift"
        )

# %% [markdown]
# ## Generate Video of Original Trajectory
# We can use `cv2` to stitch the `realsense_color` frames into a video.
# We will encode it at 60 FPS to play it back quickly.

# %%
import cv2
import glob

# Path to realsense_color images
realsense_dir = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "agilex_data_collection", "pick_bag_joe", "episode_000", "realsense_color"
))

if os.path.exists(realsense_dir):
    print("\nGenerating video from realsense images...")
    image_files = sorted(glob.glob(os.path.join(realsense_dir, "*.png")))
    
    if len(image_files) > 0:
        # Read first image to get dimensions
        first_img = cv2.imread(image_files[0])
        height, width, layers = first_img.shape
        
        video_name = 'piper_episode_000_fast.mp4'
        # Define the codec and create VideoWriter object
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        # 60 fps to speed up the video playback
        fps = 60
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
        
        for img_path in image_files:
            video.write(cv2.imread(img_path))
            
        cv2.destroyAllWindows()
        video.release()
        print(f"Video saved successfully as '{video_name}' in the current working directory.")
        
        # If running in a Jupyter Notebook, you can display it using:
        # from IPython.display import Video
        # display(Video(video_name))
    else:
        print("No png images found in the directory.")
else:
    print(f"Directory not found: {realsense_dir}")
