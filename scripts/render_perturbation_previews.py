#!/usr/bin/env python3
"""Render a2l-pr perturbations photoreally with the one-to-many renderer.

For each requested perturbation type: apply it to a recorded Piper episode,
then render the PRE-GRASP segment (static scene — before the bag is picked up)
side by side with the real recording, in both views (third-person ZED composite
+ wrist depth-warp). See ``a2l_pr.rendering.o2m_bridge``.

    cd a2l-pr
    PYTHONPATH=src:../one-to-many/src MUJOCO_GL=egl \
        python scripts/render_perturbation_previews.py \
        --types lateral_drift underreach_idle_before_max_reach --severity 0.6

Outputs -> output/o2m_previews/<type>/ (zed_preview.mp4, wrist_preview.mp4,
stills, preview.json).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from a2l_pr.adapters.piper import PiperAdapter  # noqa: E402
from a2l_pr.perturbations.generator import (PerturbationGenerator,  # noqa: E402
                                            PerturbationType)
from a2l_pr.rendering import O2MPreviewRenderer  # noqa: E402

DEFAULT_EPISODE = ("/home/griffing52/vail/bot2bot/bot2bot/a2l/"
                   "agilex_data_collection/pick_bag_joe/episode_000")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episode-dir", default=DEFAULT_EPISODE)
    ap.add_argument("--types", nargs="+",
                    default=["lateral_drift", "underreach_idle_before_max_reach"],
                    choices=[t.value for t in PerturbationType])
    ap.add_argument("--severity", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--until", type=int, default=None,
                    help="Render [0, until); default = the PERTURBED grasp frame.")
    ap.add_argument("--underreach-gap", type=int, default=25,
                    help="underreach_anchor_gap_steps: how many steps BEFORE the "
                         "detected close the idle starts. The default anchor gap "
                         "(~9) lands the pause ON the physical grasp for this "
                         "episode (a2l-pr's first_close estimate lags the closing "
                         "edge); a larger gap freezes the arm mid-approach, which "
                         "is the renderable pre-grasp case.")
    ap.add_argument("--out", default=str(_ROOT / "output" / "o2m_previews"))
    ap.add_argument("--wrist-renderer", default="depthwarp",
                    choices=["depthwarp", "genwarp", "genwarp_holes"],
                    help="genwarp = diffusion NVS on the same VDA depth "
                         "(clean disocclusions at large offsets, ~10-18s/frame)."
                         " genwarp_holes = forward-warped REAL pixels with "
                         "GenWarp only inside the disocclusion holes (keeps "
                         "its look, confines its per-frame flicker).")
    ap.add_argument("--genwarp-depth-scale", type=float, default=None)
    ap.add_argument("--gripper-mask", default="bank",
                    choices=["bank", "sam3", "temporal", "trapezoid"],
                    help="bank = cross-episode state-indexed masks (portable, "
                         "no priors; needs >=4 sibling episodes, else falls "
                         "back to temporal). sam3 = SAM-3 video tracker "
                         "seeded by the bank masks (tightest boundaries; "
                         "3.3GB facebook/sam3 weights). temporal = single-"
                         "episode rigidity. trapezoid = static baseline.")
    ap.add_argument("--no-raw-panel", action="store_true",
                    help="Drop the middle un-inpainted pure-warp panel.")
    ap.add_argument("--no-zed", action="store_true", help="Wrist view only.")
    ap.add_argument("--fill-method", default=None,
                    choices=["nearest", "edge_aware", "inpaint", "lama",
                             "scene", "scene_lama"],
                    help="scene = fill disocclusions from the stage-10 metric "
                         "ZED reconstruction (one model per episode -> "
                         "temporally consistent fill). lama = big-LaMa Fourier-"
                         "conv inpainting (structure-preserving, deterministic)"
                         "; scene_lama = scene fill with LaMa for the leftover "
                         "holes the ZED never saw (instead of TELEA).")
    ap.add_argument("--name", default=None,
                    help="Output subdir name (default: the perturbation type).")
    ap.add_argument("--meta", action="append", default=[],
                    help="Trajectory-metadata override KEY=VALUE (repeatable), "
                         "e.g. lateral_drift_extra_m=0.10 for >13cm drifts.")
    args = ap.parse_args()

    episode_dir = Path(args.episode_dir)
    original = PiperAdapter().load(str(episode_dir / "actions.csv"))
    original["metadata"]["underreach_anchor_gap_steps"] = args.underreach_gap
    for kv in args.meta:
        k, v = kv.split("=", 1)
        try:
            original["metadata"][k] = float(v)
        except ValueError:
            original["metadata"][k] = v
    gen = PerturbationGenerator()
    renderer = O2MPreviewRenderer(episode_dir)

    for tname in args.types:
        ptype = PerturbationType(tname)
        result = gen.apply_perturbation(original, ptype,
                                        severity=args.severity, seed=args.seed)
        if result is None:
            print(f"[skip] {tname}: not applicable to this trajectory")
            continue
        out_dir = Path(args.out) / (args.name or tname)
        r = renderer.render_preview(original, result, out_dir,
                                    until=args.until, stride=args.stride,
                                    views=("wrist",) if args.no_zed else ("zed", "wrist"),
                                    wrist_renderer=args.wrist_renderer,
                                    genwarp_depth_scale=args.genwarp_depth_scale,
                                    gripper_mask=args.gripper_mask,
                                    raw_panel=not args.no_raw_panel,
                                    fill_method=args.fill_method)
        print(f"[done] {tname}: window={result.perturbation_window} "
              f"severity={result.severity:.2f} -> {r.rendered_frames} frames "
              f"(grasp @ {r.grasp_frame}), max offset {r.max_offset_m*100:.1f}cm, "
              f"IK {'success' if r.ik_success else f'FAIL x{r.n_unreachable}'}")
        print(f"       {r.out_dir}")


if __name__ == "__main__":
    main()
