"""Render a2l-pr perturbations with the one-to-many (o2m) world-model renderer.

Turns a :class:`a2l_pr.perturbations.generator.PerturbationResult` into
photoreal(ish) dual-view previews of the PRE-GRASP segment — the part of the
episode where the scene is still static (nothing has been picked up), so o2m's
static-scene renderers are valid:

  - **third-person (ZED)**: URDF robot at IK joints over the real clean plate,
    depth-ordered against the metric scene depth (o2m stages 07/10);
  - **wrist (RealSense)**: the real recorded frame depth-warped by the
    perturbation's EE offset (Video-Depth-Anything + forward splat).

The mapping is deliberately narrow: whatever the perturbation did to actions,
its *renderable* effect is the per-frame base-frame EE offset
``perturbed_obs - original_obs`` (a2l-pr freezes/ramps observations; for Piper
the perturbed *actions* are absolute joints and NOT usable directly). o2m then
IK-solves FK(measured)+offset per frame — zero offset reproduces the measured
arm exactly — and labels reachability, mirroring its own pipeline.

Requires the one-to-many checkout (default sibling ``a2l/one-to-many``) with
its stage-10 assets built for the episode (clean plate, ZED extrinsic; scene
depth npz optional). Run inside the ``o2m`` conda env with ``MUJOCO_GL=egl``.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_O2M_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/one-to-many")


def _ensure_o2m(root: Path) -> None:
    p = str(root / "src")
    if p not in sys.path:
        sys.path.insert(0, p)


@dataclass
class PreviewResult:
    """Paths + labels produced by :meth:`O2MPreviewRenderer.render_preview`."""
    out_dir: Path
    zed_mp4: Optional[Path]
    wrist_mp4: Optional[Path]
    stills: List[Path]
    grasp_frame: int
    rendered_frames: int
    ik_success: bool
    n_unreachable: int
    max_offset_m: float


class O2MPreviewRenderer:
    """Reusable renderer: one instance per episode, many perturbations."""

    def __init__(self, episode_dir: str | Path,
                 o2m_root: str | Path = _O2M_ROOT,
                 o2m_config: str = "configs/worldmodel.yaml",
                 depth_encoder: str = "vits"):
        self.o2m_root = Path(o2m_root)
        _ensure_o2m(self.o2m_root)
        from o2m.config import Config

        self.cfg = Config.from_yaml(str(self.o2m_root / o2m_config))
        self.wm = self.cfg.section("worldmodel")
        self.episode_dir = Path(episode_dir)
        self.depth_encoder = depth_encoder
        self._model = None       # PiperModel (FK/IK)
        self._tp = None          # ThirdPersonRenderer
        self._warper = None      # WristWarper
        self._vde = None         # VideoDepthEstimator
        self._gripper_max = None

    # -- lazy o2m components -------------------------------------------------
    def _abs(self, p) -> Path:
        q = Path(p)
        return q if q.is_absolute() else (self.o2m_root / q).resolve()

    @property
    def model(self):
        if self._model is None:
            from o2m.robot import PiperModel
            self._model = PiperModel(
                self.cfg.require("robot.urdf"), self.cfg.require("robot.urdf_dir"),
                base_frame=self.cfg.get("robot.base_frame", "base_link"),
                ee_frame=self.cfg.get("robot.ee_frame"),
                camera_frame=self.cfg.get("robot.camera_frame") or "hand_cam")
        return self._model

    @property
    def thirdperson(self):
        if self._tp is None:
            from PIL import Image
            from o2m.robot import RobotRenderer
            from o2m.worldmodel.thirdperson import (ThirdPersonRenderer,
                                                    load_zed_camera)
            from o2m.worldmodel.scene_cloud import load_scene_depth
            rr = RobotRenderer(
                str(self.cfg.get("robot.render_urdf") or self.cfg.require("robot.urdf")),
                self.cfg.require("robot.urdf_dir"))
            cam = load_zed_camera(self._abs(self.wm["zed_extrinsic_npz"]))
            plate = np.asarray(Image.open(self._abs(self.wm["clean_plate"])).convert("RGB"))
            tp_cfg = self.wm.get("thirdperson", {})
            depth = load_scene_depth(self._abs(tp_cfg["scene_depth_npz"])) \
                if tp_cfg.get("scene_depth_npz") else None
            self._tp = ThirdPersonRenderer(rr, cam, plate, scene_depth=depth,
                                           depth_margin=float(tp_cfg.get("depth_margin", 0.12)))
        return self._tp

    @property
    def warper(self):
        if self._warper is None:
            from o2m.worldmodel.wrist_warp import (GripperMask, WristIntrinsics,
                                                   WristWarper)
            warp = self.wm.get("warp", {})
            self._warper = WristWarper(
                WristIntrinsics(**self.wm["wrist_intrinsics"]),
                GripperMask(**self.wm["gripper_mask"]),
                kernel_splat=bool(warp.get("kernel_splat", True)),
                inpaint_holes=bool(warp.get("inpaint_holes", True)),
                fill_method=warp.get("fill_method", "inpaint"),
                reject_floaters=bool(warp.get("reject_floaters", True)))
        return self._warper

    def _bank_masks(self, wrist_paths, traj, idx, reals,
                    sam3: bool = False) -> Optional[List[np.ndarray]]:
        """Per-frame masks from the cross-episode mask bank (built + cached
        once per dataset), optionally boundary-snapped by the SAM-3 video
        tracker seeded with the bank masks. Returns None when there are too
        few sibling episodes to build a scene-diverse bank."""
        from o2m.worldmodel.gripper_mask import MaskBankGripperMasker
        siblings = sorted(p for p in self.episode_dir.parent.glob("episode_*")
                          if p.is_dir())
        if len(siblings) < 4:
            print(f"[o2m_bridge] only {len(siblings)} episodes — mask bank "
                  "needs scene diversity, falling back to temporal masker")
            return None
        bank_path = (self.o2m_root / "outputs" / "mask_banks" /
                     f"{self.episode_dir.parent.name}.npz")
        if bank_path.exists():
            bank = MaskBankGripperMasker.load(bank_path)
        else:
            bank = MaskBankGripperMasker.build(siblings)
            bank.save(bank_path)
            print(f"[o2m_bridge] built gripper mask bank -> {bank_path}")
        masker = bank
        if sam3:
            from o2m.worldmodel.gripper_mask import Sam3GripperMasker
            masker = Sam3GripperMasker(bank)
        return masker.attach(wrist_paths, traj.gripper,
                             ee_positions=traj.positions).masks(idx, rgbs=reals)

    def _wrist_depths(self, reals: Sequence[np.ndarray]) -> List[np.ndarray]:
        from o2m.worldmodel.wrist_warp import disparities_to_depths
        from o2m.depth import VideoDepthEstimator
        if self._vde is None:
            self._vde = VideoDepthEstimator(encoder=self.depth_encoder)
        try:
            disps = self._vde.estimate_sequence(reals)
        except Exception as e:  # CUDA OOM (GPU shared with other jobs) -> CPU
            if "out of memory" not in str(e).lower():
                raise
            print("[o2m_bridge] GPU OOM for video depth — retrying on CPU "
                  "(slow but fine for previews)")
            self._vde = VideoDepthEstimator(encoder=self.depth_encoder, device="cpu")
            disps = self._vde.estimate_sequence(reals)
        return list(disparities_to_depths(disps))

    # -- the bridge -----------------------------------------------------------
    def render_preview(self, original: Dict, result, out_dir: str | Path,
                       until: Optional[int] = None, stride: int = 1,
                       views: Sequence[str] = ("zed", "wrist"),
                       stills_at: int = 3, fps: int = 30,
                       wrist_renderer: str = "depthwarp",
                       genwarp_depth_scale: Optional[float] = None,
                       gripper_mask: str = "temporal",
                       raw_panel: bool = True,
                       fill_method: Optional[str] = None) -> PreviewResult:
        """Render the pre-grasp segment of one perturbation, side by side.

        Args:
            original: a2l-pr trajectory dict (PiperAdapter format, unperturbed).
            result: a2l-pr ``PerturbationResult`` for the same trajectory.
            out_dir: output directory (created).
            until: last frame to render (exclusive). None -> the PERTURBED
                trajectory's grasp frame (objects only move once the perturbed
                gripper actually closes on the bag — an idling/underreaching
                arm leaves the scene static well past the original grasp).
            stride: temporal stride.
            views: any of ("zed", "wrist").
            stills_at: save this many evenly spaced side-by-side stills.
            wrist_renderer: "depthwarp" (fast, real pixels, sprays past
                ~5-8 cm) or "genwarp" (Sony GenWarp diffusion NVS on the SAME
                VDA depth + camera offset — clean disocclusions at the large
                offsets failures produce; ~10-18 s/frame on GPU).
            genwarp_depth_scale: overrides the o2m config value. <1 exaggerates
                the parallax (VDA pseudo-depth pins the median at 0.5 m, which
                under-shifts vs metric — see one-to-many notes 2026-07-02).
            gripper_mask: "bank" (cross-episode state-indexed masks — portable,
                no appearance/position priors; auto-falls-back to temporal when
                <4 sibling episodes), "temporal" (single-episode rigidity,
                Piper-tuned gates) or "trapezoid" (static baseline).
            raw_panel: insert the UN-inpainted pure forward-warp as a middle
                panel in the wrist preview (real | raw | filled) so hole-fill /
                diffusion quality can be judged against the raw reprojection.
        """
        from PIL import Image
        from o2m.data import load_ee_trajectory, load_joint_trajectory
        from o2m.data.episode import Episode
        from o2m.render.video import save_mp4
        from o2m.worldmodel.perturb import (PerturbationSpec, PerturbedTrajectory,
                                            check_feasibility, detect_grasp_frame)
        from o2m.worldmodel.wrist_warp import base_offset_to_camera

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ep = Episode(self.episode_dir)

        # Original trajectory in o2m units (from the adapter's preserved CSV).
        df = pd.DataFrame(original["observations"]["original_df"])
        traj = load_ee_trajectory(df)
        joints = load_joint_trajectory(df)
        grasp = detect_grasp_frame(traj.gripper)

        # The renderable signal: per-frame base-frame EE offset (metres).
        pert_obs = result.perturbed_trajectory["observations"]
        offsets = np.zeros_like(traj.positions)
        n = min(len(offsets), len(pert_obs["slave_ee_pos"]))
        offsets[:n] = (pert_obs["slave_ee_pos"][:n]
                       - original["observations"]["slave_ee_pos"][:n])

        if until is not None:
            end = int(until)
        else:
            # The scene is static until the PERTURBED gripper closes on the
            # object. (Frozen/idling obs delay this past the original grasp;
            # a premature close pulls it earlier — conservative either way.)
            g_stream = np.abs(np.asarray(pert_obs["slave_gripper_angle"],
                                         float).reshape(-1)) * 1e-6
            end = int(detect_grasp_frame(g_stream))
        end = min(end, len(traj))
        idx = list(range(0, end, stride))

        # IK for the shifted arm (o2m convention: target = FK(measured)+offset).
        sub = traj.copy()
        sub.timestamps, sub.positions = sub.timestamps[idx], sub.positions[idx]
        sub.rotvecs, sub.gripper = sub.rotvecs[idx], sub.gripper[idx]
        pert_sub = PerturbedTrajectory(
            traj=sub, offsets=offsets[idx], weights=np.ones(len(idx)),
            grasp_frame=grasp,
            spec=PerturbationSpec(name=result.perturbation_type.value))
        feas = check_feasibility(self.model, pert_sub, joints[idx])

        # Perturbed gripper -> finger joints (premature close/open visible).
        g_pert = np.abs(np.asarray(pert_obs["slave_gripper_angle"]).reshape(-1)) * 1e-6
        g_max = max(float(np.abs(traj.gripper).max()), 1e-6)
        max_w = float(self.cfg.get("robot.gripper.max_width_m", 0.07))
        fingers = np.clip(g_pert / g_max, 0, 1) * (max_w / 2.0)

        stills: List[Path] = []
        still_ids = set(np.linspace(0, len(idx) - 1, stills_at, dtype=int).tolist())
        zed_mp4 = wrist_mp4 = None

        if "zed" in views:
            zed_paths = ep.zed_frames()
            frames = []
            for k, i in enumerate(idx):
                q8 = np.concatenate([feas.joints[k], [fingers[i], fingers[i]]])
                synth = self.thirdperson.render(q8)
                real = np.asarray(Image.open(zed_paths[i]).convert("RGB"))
                frames.append(np.concatenate([real, synth], axis=1))
            zed_mp4 = save_mp4(frames, out_dir / "zed_preview.mp4", fps=max(1, fps // stride))
            for k in still_ids:
                p = out_dir / f"zed_f{idx[k]:03d}.png"
                Image.fromarray(frames[k]).save(p)
                stills.append(p)

        if "wrist" in views:
            from o2m.worldmodel.wrist_warp import WristIntrinsics
            wrist_paths = ep.wrist_frames()
            reals = [np.asarray(Image.open(wrist_paths[i]).convert("RGB")) for i in idx]
            depths = self._wrist_depths(reals)
            cam_frame = self.model.camera_frame
            intr = WristIntrinsics(**self.wm["wrist_intrinsics"])

            if gripper_mask in ("bank", "sam3"):
                gmasks = self._bank_masks(wrist_paths, traj, idx, reals,
                                          sam3=gripper_mask == "sam3")
                if gmasks is None:          # too few sibling episodes
                    gripper_mask = "temporal"
            if gripper_mask == "temporal":
                from o2m.worldmodel.gripper_mask import TemporalGripperMasker
                masker = TemporalGripperMasker(wrist_paths, traj.gripper, traj.positions)
                gmasks = masker.masks(idx, rgbs=reals)
            elif gripper_mask == "trapezoid":
                from o2m.worldmodel.wrist_warp import GripperMask
                static = GripperMask(**self.wm["gripper_mask"]).mask(intr.height, intr.width)
                gmasks = [static] * len(idx)

            gw = None
            if wrist_renderer in ("genwarp", "genwarp_holes"):
                from o2m.worldmodel.genwarp_warp import GenWarpWrapper
                gwc = self.wm.get("genwarp", {})
                gw = GenWarpWrapper(
                    num_inference_steps=int(gwc.get("num_inference_steps", 20)),
                    guidance_scale=float(gwc.get("guidance_scale", 3.5)))
                gw_mode = gwc.get("mode", "pad")
                gw_scale = genwarp_depth_scale if genwarp_depth_scale is not None \
                    else float(gwc.get("depth_scale", 1.0))
            scene_filler = None
            if fill_method in ("scene", "scene_lama"):
                # Disocclusions filled from the stage-10 metric ZED cloud —
                # ONE reconstruction per episode => temporally consistent fill
                # (unlike per-frame generative fill). See o2m scene_fill.py.
                # scene_lama: leftover holes (never seen by the ZED) go to
                # LaMa instead of TELEA — better structure continuation.
                from o2m.worldmodel.fusion import optical_c2w
                from o2m.worldmodel.scene_fill import SceneFiller
                tp_cfg = self.wm.get("thirdperson", {})
                scene_filler = SceneFiller.from_stage10(
                    self._abs(tp_cfg["scene_depth_npz"]),
                    self._abs(self.wm["clean_plate"]), intr,
                    residual_fill="lama" if fill_method == "scene_lama"
                    else "inpaint")
            frames = []
            for k, i in enumerate(idx):
                T_cam = self.model.fk(joints[i], [cam_frame])[cam_frame]
                R = T_cam[:3, :3]
                dcam = base_offset_to_camera(offsets[i], R)
                gm = gmasks[k]
                zero = np.linalg.norm(dcam) < 1e-4
                raw = reals[k] if zero else \
                    self.warper.warp(reals[k], depths[k], dcam,
                                     fill_method="none", gmask=gm)
                if gw is not None:
                    if zero:
                        warped = reals[k]
                    else:
                        warped = gw.warp(reals[k], depths[k], dcam, intr.fy,
                                         mode=gw_mode, depth_scale=gw_scale)
                        if wrist_renderer == "genwarp_holes":
                            # Keep the forward-warped REAL pixels; take the
                            # diffusion output only inside the disocclusion
                            # holes — confines GenWarp's per-frame
                            # re-hallucination (flicker) to pixels nothing
                            # real covers. Same depth scaling on both paths
                            # so the two warps' parallax (and thus the holes)
                            # line up.
                            out_s, filled_s = self.warper.scatter(
                                reals[k], depths[k] * gw_scale, dcam, gmask=gm)
                            holes = ~(filled_s | gm)
                            # Exposure-match the fill (same trick as
                            # SceneFiller): the diffusion round-trip shifts
                            # brightness, which reads as bands at the hole
                            # seams. Estimate the per-channel bias where both
                            # sources cover the same pixels.
                            if filled_s.sum() > 500:
                                bias = np.clip(np.median(
                                    out_s[filled_s].astype(np.float32)
                                    - warped[filled_s].astype(np.float32),
                                    axis=0), -70, 70)
                                warped = np.clip(
                                    warped.astype(np.float32) + bias,
                                    0, 255).astype(np.uint8)
                            out_s[holes] = warped[holes]
                            warped = out_s
                        warped[gm] = reals[k][gm]   # gripper rigid to cam
                elif scene_filler is not None:
                    if zero:
                        warped = reals[k]
                    else:
                        from o2m.worldmodel.fusion import optical_c2w
                        out_s, filled_s = self.warper.scatter(
                            reals[k], depths[k], dcam, gmask=gm)
                        c2w_pert = optical_c2w(T_cam)
                        c2w_pert[:3, 3] += offsets[i]   # perturbed wrist pose
                        warped = scene_filler.fill(out_s, filled_s | gm, c2w_pert)
                        warped[gm] = reals[k][gm]
                else:
                    warped = reals[k] if zero else \
                        self.warper.warp(reals[k], depths[k], dcam, gmask=gm,
                                         fill_method=fill_method)
                panels = [reals[k]] + ([raw] if raw_panel else []) + [warped]
                frames.append(np.concatenate(panels, axis=1))
            wrist_mp4 = save_mp4(frames, out_dir / "wrist_preview.mp4", fps=max(1, fps // stride))
            for k in still_ids:
                p = out_dir / f"wrist_f{idx[k]:03d}.png"
                Image.fromarray(frames[k]).save(p)
                stills.append(p)

        info = {
            "perturbation_type": result.perturbation_type.value,
            "perturbation_window": list(result.perturbation_window),
            "severity": result.severity,
            "parameters": result.parameters,
            "recovery_text": result.recovery_text,
            "theoretical_failure_mode": result.theoretical_failure_mode,
            "grasp_frame": int(grasp),
            "wrist_renderer": wrist_renderer,
            "gripper_mask": gripper_mask,
            "wrist_panels": "real | raw warp | filled" if raw_panel else "real | filled",
            "rendered_frames": f"[0, {end}) stride {stride}",
            "ik_success": bool(feas.success),
            "n_unreachable": int(feas.n_unreachable),
            "max_offset_m": float(np.linalg.norm(offsets[:end], axis=1).max()),
        }
        with open(out_dir / "preview.json", "w") as f:
            json.dump(info, f, indent=2, default=str)

        return PreviewResult(out_dir=out_dir, zed_mp4=zed_mp4, wrist_mp4=wrist_mp4,
                             stills=stills, grasp_frame=int(grasp),
                             rendered_frames=len(idx), ik_success=bool(feas.success),
                             n_unreachable=int(feas.n_unreachable),
                             max_offset_m=info["max_offset_m"])
