# a2l-pr

A framework for synthetic perturbations and recovery generation for robotic trajectories.

## Installation

```bash
pip install -e .
```

## Rendering perturbations photoreally (one-to-many bridge)

`a2l_pr.rendering.O2MPreviewRenderer` renders a `PerturbationResult` with the
sibling [one-to-many](../one-to-many) world-model renderer: the **pre-grasp
segment** (static scene — until the perturbed gripper actually closes on the
object) is re-rendered in both views, side by side with the real recording:

- **third-person (ZED)**: URDF arm at IK-solved perturbed joints over the real
  clean plate (depth-ordered against the metric scene depth) — also yields an
  IK reachability label per perturbation;
- **wrist (RealSense)**: the real frame depth-warped by the perturbation's EE
  offset (Video-Depth-Anything).

Whatever a perturbation did to actions, its renderable effect is taken as the
per-frame EE offset `perturbed_obs − original_obs` (for Piper the perturbed
*actions* are absolute joints and not directly renderable).

```bash
# in the o2m conda env; needs one-to-many stage-10 assets for the episode
PYTHONPATH=src:../one-to-many/src MUJOCO_GL=egl \
    python scripts/render_perturbation_previews.py \
    --types lateral_drift --seed 34 --severity 0.7
# -> output/o2m_previews/<type>/{zed_preview.mp4, wrist_preview.mp4, stills, preview.json}

# CURRENT BEST STACK (2026-07): SAM-3 gripper masks + metric-scene fill with
# LaMa residuals (the "drift_backleft_sam3" reference render):
PYTHONPATH=src:../one-to-many/src MUJOCO_GL=egl HF_HOME=~/hf_cache \
    python scripts/render_perturbation_previews.py \
    --types lateral_drift --seed 56 --severity 0.9 --no-zed \
    --fill-method scene_lama --gripper-mask sam3 --name drift_backleft_sam3

# GenWarp diffusion NVS arms: full-frame (--wrist-renderer genwarp) or
# genwarp_holes = forward-warped REAL pixels everywhere, exposure-matched
# GenWarp only inside the disocclusion holes (confines per-frame flicker):
PYTHONPATH=src:../one-to-many/src MUJOCO_GL=egl HF_HOME=~/hf_cache \
    python scripts/render_perturbation_previews.py \
    --types lateral_drift --seed 56 --severity 0.9 --no-zed \
    --wrist-renderer genwarp_holes --gripper-mask sam3 --name drift_backleft_gwholes

# Push past severity 1.0 with metadata overrides (~12cm achieved offset):
#   --severity 1.0 --meta lateral_drift_extra_m=0.10
```

Wrist previews are three panels — `real | raw un-inpainted warp | filled`
(`--no-raw-panel` to drop the middle). `--fill-method scene_lama` fills the
disocclusions from o2m's stage-10 **metric ZED reconstruction** rendered at the
perturbed wrist pose (temporally consistent by construction), with big-LaMa for
the residual holes the ZED never saw. `--gripper-mask` picks the per-frame
gripper masker: `sam3` (SAM-3 video tracker seeded by the bank — tightest
boundaries; needs the 3.3 GB `facebook/sam3` weights in `$HF_HOME`), `bank`
(cross-episode state-indexed masks, ~5 ms/frame, auto-built and cached per
dataset; the default), `temporal` (single-episode rigidity; the automatic
fallback below 4 episodes), `trapezoid` (static baseline). Warp spray/ghosting
on thin structures is suppressed by default (`warp.reject_floaters` in o2m's
`configs/worldmodel.yaml` — bag handles stay single up to ≥12 cm offsets).

Caveats: valid only while the scene is static (the bridge auto-stops at the
perturbed grasp); frames where the wrist is close to clutter at large offset
still degrade (see one-to-many `docs/synthetic_data.md` → Known limitations);
underreach pauses need `--underreach-gap` ≳ 40 on episode_000 to freeze during
visible motion (the default anchor lands on the grasp itself).
