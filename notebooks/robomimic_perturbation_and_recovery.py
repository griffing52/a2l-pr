# %% [markdown]
# # Robomimic Perturbation and Recovery (Refactored)
# This script demonstrates using the `a2l-pr` package to apply synthetic perturbations
# to a robomimic HDF5 trajectory, replacing the old monolithic notebook.

# %%
import os
import sys
import h5py

# Ensure the a2l-pr package is in the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from a2l_pr.adapters.robomimic import RobomimicAdapter
from a2l_pr.perturbations.generator import PerturbationGenerator, PerturbationType
from a2l_pr.recovery.generator import RecoveryCodeGenerator, recovery_to_json
from a2l_pr.utils.visualization import visualize_trajectory_comparison
from a2l_pr.config.config import ConfigManager

# %% [markdown]
# ## Utility function to load a single robomimic trajectory from HDF5

# %%
def load_robomimic_trajectory(hdf5_path, demo_key="demo_0"):
    """Loads a single trajectory dictionary from a standard robomimic HDF5 file."""
    if not os.path.exists(hdf5_path):
        print(f"HDF5 file not found: {hdf5_path}")
        return None
        
    f = h5py.File(hdf5_path, 'r')
    if "data" not in f or demo_key not in f["data"]:
        print(f"Invalid HDF5 structure or missing {demo_key}")
        f.close()
        return None
        
    demo_grp = f["data"][demo_key]
    obs_grp = demo_grp["obs"]
    
    # Reconstruct the dictionary
    trajectory = {
        'actions': demo_grp["actions"][:],
        'observations': {}
    }
    
    for key in obs_grp.keys():
        trajectory['observations'][key] = obs_grp[key][:]
        
    f.close()
    return trajectory

# %% [markdown]
# ## Configuration and Loading

# %%
# Load configuration
config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "perturbation_config.yaml"))
config_mgr = ConfigManager.from_yaml(config_path)

# Path to Robomimic HDF5 (Update this to point to a valid dataset)
# Using a dummy path for demonstration. The user should point this to their actual dataset.
robomimic_hdf5_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "..", "..", "robomimic", "datasets", "square", "ph", "image.hdf5"
))

# We'll use a mocked trajectory if the file doesn't exist just to show the API structure.
import numpy as np
raw_trajectory = load_robomimic_trajectory(robomimic_hdf5_path)

if raw_trajectory is None:
    print("Loading a synthetic dummy trajectory for demonstration purposes.")
    traj_len = 100
    t = np.linspace(0, 1, traj_len)
    raw_trajectory = {
        'observations': {
            'robot0_eef_pos': np.column_stack((t, t, t)),
            'robot0_gripper_qpos': np.concatenate((np.zeros((50, 1)), np.ones((50, 1)))) # close at step 50
        },
        'actions': np.random.randn(traj_len, 7)
    }

# Apply Adapter
adapter = RobomimicAdapter()
trajectory = adapter.load(raw_trajectory)

# Inject dataset defaults from config
dataset_defaults = config_mgr.get_dataset_defaults(trajectory['metadata']['dataset_type'])
trajectory['metadata'].update(dataset_defaults)
trajectory['metadata'].update(config_mgr.get_perturbation_config())

print(f"Loaded Robomimic trajectory with {len(trajectory['actions'])} steps.")

# %% [markdown]
# ## Apply Perturbation

# %%
# Initialize generator
generator = PerturbationGenerator()

# Apply an Underreach Idle perturbation
print("Applying Underreach Idle Perturbation...")
result = generator.apply_perturbation(
    trajectory, 
    PerturbationType.UNDERREACH_IDLE, 
    severity=0.5,
    seed=42
)

if result is not None:
    print(f"Success! Perturbation applied: {result.perturbation_type.name}")
    print(f"Failure Mode: {result.theoretical_failure_mode}")
    
    # Visualize the comparison
    visualize_trajectory_comparison(
        trajectory, 
        result.perturbed_trajectory, 
        title="Robomimic Trajectory: Original vs Underreach Idle"
    )
else:
    print("Perturbation not applicable for this trajectory.")

# %% [markdown]
# ## Generate Recovery FSM

# %%
if result is not None:
    print("\nGenerating Recovery FSM DSL...")
    recovery_code = RecoveryCodeGenerator.generate_recovery_code(result)
    
    # Pretty print the JSON
    print(recovery_to_json(recovery_code))

# %% [markdown]
# ## 10. Simulator Video Playback

# %%
import os
import h5py
import imageio
import numpy as np
from pathlib import Path
from IPython.display import HTML, display
from base64 import b64encode

try:
    import robomimic.utils.obs_utils as ObsUtils
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
except ImportError:
    print("robomimic is not installed. Video playback may fail.")

def render_trajectory_to_video(traj_dict, dataset_path, ep_name, output_path):
    if not os.path.exists(dataset_path):
        print(f"Dataset {dataset_path} not found.")
        return False

    # Initialize headless env
    dummy_spec = dict(obs=dict(low_dim=["robot0_eef_pos"], rgb=[]))
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=dummy_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=dataset_path)
    env = EnvUtils.create_env_from_metadata(env_meta=env_meta, render=False, render_offscreen=True)
    is_robosuite_env = EnvUtils.is_robosuite_env(env_meta)

    with h5py.File(dataset_path, 'r') as f:
        states = f[ep_name]['states'][()]
        initial_state = dict(states=states[0])
        if is_robosuite_env:
            initial_state["model"] = f[ep_name].attrs["model_file"]
            initial_state["ep_meta"] = f[ep_name].attrs.get("ep_meta", None)

    env.reset_to(initial_state)
    actions = traj_dict.get('actions', [])

    if len(actions) == 0:
        print("No actions found for playback.")
        return False

    writer = imageio.get_writer(output_path, fps=20)
    print(f"Rendering {len(actions)} frames to {output_path}...")
    for i in range(len(actions)):
        env.step(actions[i])
        if i % 5 == 0:
            frame = env.render(mode="rgb_array", height=256, width=256, camera_name="agentview")
            writer.append_data(frame)
    writer.close()
    return True

# Map script variables to the snippet's expected variables
DATASET_PATH = robomimic_hdf5_path
OUTPUT_DIR = Path(os.path.dirname(__file__)) / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# NOTE: Do not rely on global `trajectory` / `result` here, since later debug cells may overwrite them.
if DATASET_PATH and 'load_robomimic_trajectory' in globals():
    # Pick episode name from the loaded dataset demo selection.
    if 'first_demo_name' in locals():
        ep_name = first_demo_name
    elif 'trajectory' in locals() and isinstance(trajectory, dict) and trajectory.get('name'):
        ep_name = trajectory['name']
    else:
        ep_name = 'data/demo_0'
        
    demo_key = ep_name.replace("data/", "") if ep_name.startswith("data/") else ep_name

    # Pick perturbation result from the main perturbation section.
    selected_result = None
    if 'premature_close_result' in locals() and premature_close_result is not None:
        selected_result = premature_close_result
    elif 'perturbation_results' in locals() and perturbation_results:
        selected_result = perturbation_results[0]
    elif 'result' in locals() and result is not None:
        selected_result = result

    original_traj = load_robomimic_trajectory(DATASET_PATH, demo_key=demo_key)

    if original_traj is None or selected_result is None:
        print("Missing original trajectory or perturbation result for playback.")
    else:
        orig_out = str(OUTPUT_DIR / 'original.mp4')
        pert_out = str(OUTPUT_DIR / 'perturbed.mp4')

        print(f"Playback episode: {ep_name}")
        print(f"Perturbation shown: {selected_result.perturbation_type.value}")

        print("Rendering original trajectory (this may take a minute)...")
        ok_orig = render_trajectory_to_video(original_traj, DATASET_PATH, ep_name, orig_out)
        print("Rendering perturbed trajectory...")
        ok_pert = render_trajectory_to_video(selected_result.perturbed_trajectory, DATASET_PATH, ep_name, pert_out)

        # Display side-by-side only when both fresh renders succeeded.
        if ok_orig and ok_pert and os.path.exists(orig_out) and os.path.exists(pert_out):
            orig_b64 = b64encode(open(orig_out, "rb").read()).decode('ascii')
            pert_b64 = b64encode(open(pert_out, "rb").read()).decode('ascii')

            html = f"""
            <div style='display: flex; flex-direction: row; justify-content: space-around;'>
                <div style='text-align: center;'>
                    <h3>Original Trajectory ({ep_name})</h3>
                    <video width='400' controls autoplay loop>
                        <source src='data:video/mp4;base64,{orig_b64}' type='video/mp4'>
                    </video>
                </div>
                <div style='text-align: center;'>
                    <h3>Perturbed Trajectory ({selected_result.perturbation_type.value})</h3>
                    <video width='400' controls autoplay loop>
                        <source src='data:video/mp4;base64,{pert_b64}' type='video/mp4'>
                    </video>
                </div>
            </div>
            """
            display(HTML(html))
        else:
            print("Skipping display because at least one video failed to render.")
