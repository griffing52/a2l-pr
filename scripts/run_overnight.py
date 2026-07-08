#!/usr/bin/env python3
"""Overnight residual-recovery experiment suite (resumable, logged, incremental).

Phases (cheap -> definitive):
  1. detect_old      : offline detection metrics on the existing checkpoint
  2. train_norm      : retrain residual with input normalization (full split)
  3. train_nonorm    : retrain residual WITHOUT normalization (ablation)
  4. detect_norm     : offline detection on retrained-norm
  5. detect_nonorm   : offline detection on retrained-nonorm
  6. (pick BEST retrained checkpoint by AUROC + failure accuracy)
  7. induced_old     : closed-loop induced-failure recovery, old ckpt, corrected eval
  8. induced_best    : closed-loop induced-failure recovery, best ckpt, corrected eval
  9. natural_old_legacy   : natural paired eval reproducing the prior weak config
 10. natural_old_corrected: natural paired eval, old ckpt, corrected config
 11. natural_best_corrected: natural paired eval, best ckpt, corrected config
 12. sweep_induced_best   : residual_weight sweep on induced recovery (best ckpt)
 13. consolidate          : write consolidated_summary.json

Each phase is skipped if its output already exists (unless --force). Failures are
logged and do not abort the suite. Progress is written to STATUS.json after every
phase so the run can be inspected or resumed at any time.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

A2L_PR_ROOT = Path("/home/griffing52/vail/bot2bot/bot2bot/a2l/a2l-pr")
SRC = A2L_PR_ROOT / "src"
OLD_CKPT = A2L_PR_ROOT / "notebooks" / "gated_residual_recovery_policy.pth"
CKPT_NORM = A2L_PR_ROOT / "notebooks" / "gated_residual_retrained_norm.pth"
CKPT_NONORM = A2L_PR_ROOT / "notebooks" / "gated_residual_retrained_nonorm.pth"
PY = sys.executable


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class Suite:
    def __init__(self, out_dir: Path, force: bool, fast: bool):
        self.out = out_dir
        self.out.mkdir(parents=True, exist_ok=True)
        self.force = force
        self.fast = fast
        self.log_path = self.out / "run.log"
        self.status_path = self.out / "STATUS.json"
        self.status = {"started": now(), "phases": {}}
        self._save_status()

    def log(self, msg):
        line = f"[{now()}] {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def _save_status(self):
        self.status_path.write_text(json.dumps(self.status, indent=2))

    def run_cmd(self, name, cmd, output_marker: Path, phase_log: str):
        if output_marker.exists() and not self.force:
            self.log(f"SKIP {name} (exists: {output_marker.name})")
            self.status["phases"][name] = {"status": "skipped", "output": str(output_marker)}
            self._save_status()
            return True
        self.log(f"START {name}: {' '.join(str(c) for c in cmd)}")
        self.status["phases"][name] = {"status": "running", "started": now()}
        self._save_status()
        log_file = self.out / phase_log
        try:
            with open(log_file, "w") as lf:
                proc = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT,
                                      cwd=str(A2L_PR_ROOT), env=self._env())
            ok = proc.returncode == 0 and output_marker.exists()
            self.status["phases"][name] = {
                "status": "done" if ok else "failed",
                "finished": now(), "returncode": proc.returncode,
                "output": str(output_marker), "log": str(log_file),
            }
            self.log(f"{'DONE' if ok else 'FAILED'} {name} (rc={proc.returncode})")
        except Exception as e:  # noqa: BLE001
            self.status["phases"][name] = {"status": "error", "error": str(e),
                                           "traceback": traceback.format_exc()}
            self.log(f"ERROR {name}: {e}")
            ok = False
        self._save_status()
        return ok

    def _env(self):
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(SRC) + ":" + str(A2L_PR_ROOT.parent / "robomimic" / "robomimic") + ":" + env.get("PYTHONPATH", "")
        return env

    # ---- phase helpers ----
    def detect(self, name, ckpt, out_json, demo_offset=160, n_demos=40):
        cmd = [PY, str(SRC / "a2l_pr/eval/detection_eval.py"),
               "--checkpoint", str(ckpt), "--n_demos", str(n_demos),
               "--demo_offset", str(demo_offset), "--gate_threshold", "0.5",
               "--output", str(out_json)]
        return self.run_cmd(name, cmd, out_json, f"{name}.log")

    def train(self, name, out_ckpt, normalize):
        epochs = 4 if self.fast else 40
        n_train = 12 if self.fast else 160
        n_val = 6 if self.fast else 40
        cmd = [PY, str(SRC / "a2l_pr/learning/residual_training.py"),
               "--out_path", str(out_ckpt), "--normalize", "true" if normalize else "false",
               "--epochs", str(epochs), "--max_train_demos", str(n_train),
               "--max_val_demos", str(n_val), "--label", name]
        return self.run_cmd(name, cmd, out_ckpt, f"{name}.log")

    def recovery(self, name, mode, ckpt, out_dir, weight=1.0, clamp=1.0, thr=0.5,
                 scale_by_gate=False, n_states=None, policies="bc,diffusion"):
        if self.fast:
            policies = "bc"  # diffusion rollouts dominate runtime; validate chain on bc
        if n_states is None:
            n_states = 6 if self.fast else (30 if mode == "induced" else 50)
        out_dir = Path(out_dir)
        marker = out_dir / f"recovery_{mode}_summary.json"
        cmd = [PY, str(SRC / "a2l_pr/eval/recovery_eval.py"),
               "--mode", mode, "--policies", policies,
               "--residual_checkpoint", str(ckpt),
               "--n_initial_states", str(n_states),
               "--residual_weight", str(weight), "--clamp_residual", str(clamp),
               "--gate_threshold", str(thr), "--output_dir", str(out_dir)]
        if scale_by_gate:
            cmd.append("--scale_by_gate")
        return self.run_cmd(name, cmd, marker, f"{name}.log")


def pick_best(suite: Suite, detect_dir: Path):
    """Choose retrained checkpoint with highest (gate_auroc + failure_type_acc)."""
    candidates = {
        "norm": (CKPT_NORM, detect_dir / "norm.json"),
        "nonorm": (CKPT_NONORM, detect_dir / "nonorm.json"),
    }
    best, best_score, best_label = OLD_CKPT, -1.0, "old(fallback)"
    for label, (ckpt, js) in candidates.items():
        if not js.exists() or not Path(ckpt).exists():
            continue
        try:
            d = json.loads(js.read_text())
            score = (d.get("gate_auroc") or 0) + (d.get("failure_type_accuracy_positive") or 0)
            suite.log(f"candidate {label}: auroc={d.get('gate_auroc'):.3f} "
                      f"fail_acc={d.get('failure_type_accuracy_positive'):.3f} score={score:.3f}")
            if score > best_score:
                best, best_score, best_label = Path(ckpt), score, label
        except Exception as e:  # noqa: BLE001
            suite.log(f"could not read {js}: {e}")
    suite.log(f"BEST retrained = {best_label} ({best})")
    return best, best_label


def consolidate(suite: Suite, detect_dir, best_ckpt, best_label):
    def load(p):
        p = Path(p)
        return json.loads(p.read_text()) if p.exists() else None

    summary = {"generated": now(), "best_retrained": {"label": best_label, "path": str(best_ckpt)},
               "detection": {}, "recovery_induced": {}, "recovery_natural": {}, "sweeps": {}}
    for label, fn in [("old", "old.json"), ("norm", "norm.json"), ("nonorm", "nonorm.json")]:
        d = load(detect_dir / fn)
        if d:
            summary["detection"][label] = {
                "gate_auroc": d.get("gate_auroc"),
                "clean_false_positive_rate": d.get("clean_false_positive_rate"),
                "positive_recall": d.get("positive_recall"),
                "failure_type_accuracy_positive": d.get("failure_type_accuracy_positive"),
                "per_failure_type": d.get("per_failure_type"),
            }
    rec = suite.out / "recovery"
    for key, path in [
        ("induced_old", rec / "induced_old" / "recovery_induced_summary.json"),
        ("induced_best", rec / "induced_best" / "recovery_induced_summary.json"),
    ]:
        d = load(path)
        if d:
            summary["recovery_induced"][key] = d["summaries"]
    for key, path in [
        ("natural_old_legacy", rec / "natural_old_legacy" / "recovery_natural_summary.json"),
        ("natural_old_corrected", rec / "natural_old_corrected" / "recovery_natural_summary.json"),
        ("natural_best_corrected", rec / "natural_best_corrected" / "recovery_natural_summary.json"),
    ]:
        d = load(path)
        if d:
            summary["recovery_natural"][key] = d["summaries"]
    for w in ["0.25", "0.5", "1.0"]:
        d = load(rec / f"sweep_induced_w{w}" / "recovery_induced_summary.json")
        if d:
            summary["sweeps"][f"induced_w{w}"] = d["summaries"]

    (suite.out / "consolidated_summary.json").write_text(json.dumps(summary, indent=2))
    suite.log(f"wrote consolidated_summary.json")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--fast", action="store_true", help="tiny config for end-to-end validation")
    args = ap.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d")
    out_dir = Path(args.output_dir) if args.output_dir else (A2L_PR_ROOT / "output" / f"overnight_{stamp}")
    suite = Suite(out_dir, args.force, args.fast)
    detect_dir = out_dir / "detection"; detect_dir.mkdir(parents=True, exist_ok=True)
    rec = out_dir / "recovery"
    suite.log(f"=== overnight suite start (fast={args.fast}) out={out_dir} ===")

    # Phase 1: detection on old checkpoint
    suite.detect("detect_old", OLD_CKPT, detect_dir / "old.json")
    # Phase 2-3: retrain variants
    suite.train("train_norm", CKPT_NORM, normalize=True)
    suite.train("train_nonorm", CKPT_NONORM, normalize=False)
    # Phase 4-5: detection on retrained
    suite.detect("detect_norm", CKPT_NORM, detect_dir / "norm.json")
    suite.detect("detect_nonorm", CKPT_NONORM, detect_dir / "nonorm.json")
    # Phase 6: pick best
    best_ckpt, best_label = pick_best(suite, detect_dir)

    # Phase 7-8: induced recovery (corrected eval)
    suite.recovery("induced_old", "induced", OLD_CKPT, rec / "induced_old")
    suite.recovery("induced_best", "induced", best_ckpt, rec / "induced_best")

    # Phase 9: natural, old ckpt, LEGACY config (reproduce prior weak/null result)
    suite.recovery("natural_old_legacy", "natural", OLD_CKPT, rec / "natural_old_legacy",
                   weight=0.1, clamp=0.1, thr=0.7, scale_by_gate=True)
    # Phase 10-11: natural corrected
    suite.recovery("natural_old_corrected", "natural", OLD_CKPT, rec / "natural_old_corrected")
    suite.recovery("natural_best_corrected", "natural", best_ckpt, rec / "natural_best_corrected")

    # Phase 12: residual_weight sweep on induced recovery (best ckpt), bc-only for cost
    for w in ([0.5] if args.fast else [0.25, 0.5, 1.0]):
        suite.recovery(f"sweep_induced_w{w}", "induced", best_ckpt, rec / f"sweep_induced_w{w}",
                       weight=w, clamp=1.0, policies="bc")

    # Phase 13: consolidate
    consolidate(suite, detect_dir, best_ckpt, best_label)
    suite.status["finished"] = now()
    suite._save_status()
    suite.log("=== overnight suite complete ===")


if __name__ == "__main__":
    main()
