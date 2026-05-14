# A2L-Perturbation Experiments

This note tracks the main questions we need to answer in `a2l-pr`:

- Can the failure detector learn from a realistic temporal signal?
- Which model family works best on low-dimensional robotic logs?
- How much do perturbation duration and severity matter for detectability?
- What pieces transfer from robomimic simulation to real Agilex/Piper data?

## Current Situation

The robomimic `low_dim_v15.hdf5` dataset is low-dimensional, not image-based. That means the current baseline should be treated as a temporal state/action detector, not a vision classifier.

Key implication:
- Do not rely on random image fallbacks for this dataset.
- For robomimic low-dim logs and real robot logs, the first-pass model should operate on temporal windows of proprioception/state/action features.
- Vision can be added later only if the dataset actually contains camera observations at inference time.

## What We Are Trying To Learn

The target behavior is not just "classify failure". It is:

- recognize that the trajectory has entered a bad state,
- distinguish the failure type,
- suggest a primitive recovery action,
- do this early enough that the robot can still recover.

Examples:
- Underreach: the gripper is hovering just short of the object for long enough that the model can infer a missed grasp.
- Premature close: the gripper closes before alignment and stays closed long enough to be recognized.
- Lateral drift: the end effector drifts away from target alignment and remains offset through the critical phase.
- Premature open: the gripper opens too early and the drop/release is sustained.

## Experiments

### Experiment 1: Temporal MLP Baseline

Goal:
- Establish a clean baseline with the new low-dimensional temporal window representation.

Setup:
- Input: fixed window of low-dimensional state + action history.
- Model: small MLP over flattened temporal features.
- Training: class-weighted cross entropy, label smoothing, AdamW, gradient clipping.
- Perturbations: current synthetic failures, but with longer sustained windows.

Hypothesis:
- This should beat the old image-fallback baseline and tell us whether the task is learnable from state history alone.

What to watch:
- Train loss vs val loss gap.
- Per-class F1, not just overall accuracy.
- Early detection failure rate on the sustained windows.

### Experiment 2: GRU Sequence Model

Goal:
- Test whether explicit temporal recurrence improves detection of failure onset and sustained bad states.

Setup:
- Input: ordered sequence of low-dimensional state/action vectors.
- Model: GRU encoder over the window, then classification and recovery heads.
- Compare against Experiment 1 with the same data and same split.

Hypothesis:
- GRU should help because the task depends on temporal progression, not just a single snapshot.
- This is especially relevant for "been stuck there too long" failures.

What to watch:
- Validation loss and per-class F1.
- Detection latency: how soon after perturbation onset the model predicts failure.
- Stability across different window lengths.

### Experiment 3: Perturbation Duration Ablation

Goal:
- Find the minimum perturbation persistence required for detection.

Setup:
- Hold the model fixed.
- Vary synthetic failure duration and severity.
- Compare short vs medium vs long perturbation windows.

Suggested buckets:
- Short: brief failure, likely hard to detect.
- Medium: enough time for the model to observe abnormal behavior.
- Long/catastrophic: sustained error through the critical phase.

Hypothesis:
- Very short perturbations will be hard to label reliably.
- Longer perturbations should improve learning and produce more realistic detection behavior.

What to watch:
- Accuracy/F1 as a function of perturbation duration.
- Confusion between no-failure and failure classes.
- Whether the recovery head learns more stable outputs when the failure is persistent.

### Experiment 4: Window Length Ablation

Goal:
- Determine how much context the detector needs.

Setup:
- Train the same model family with different window sizes.
- Suggested windows: 6, 12, 24, 36 steps.

Hypothesis:
- Too little context will miss temporal cues.
- Too much context may add noise and make optimization harder.

What to watch:
- Validation accuracy and F1.
- Early-detection latency.
- Whether long windows overfit more easily.

## Recommended Order

1. Temporal MLP baseline with the current improved low-dim pipeline.
2. GRU model on the exact same data split and perturbation settings.
3. Perturbation duration ablation once the baseline is stable.
4. Window-length ablation after that.

## Metrics

Do not rely on accuracy alone.

Track:
- validation loss,
- per-class precision / recall / F1,
- balanced accuracy,
- confusion matrix,
- early failure detection latency,
- recovery parameter error for failure cases,
- overfitting gap between train and validation.

## Data Rules

Robomimic sim:
- Use low-dimensional state/action windows as the primary signal for this dataset.
- Only use images if the dataset actually provides camera observations.
- Keep perturbations sustained long enough for the model to observe the abnormal state.

Agilex / Piper real-world logs:
- Start with the same state-only temporal pipeline.
- Treat camera as optional, not required.
- If images are available, fuse them later as an extra modality.
- Real-world perturbations should be derived from actual failure logs, replayed demonstrations, or controlled induced failures, not synthetic image edits.

## Notes On Recovery

The recovery head is only useful if the detector actually learns a meaningful failure state.

Keep recovery targets simple at first:
- underreach -> small forward motion,
- premature close -> reopen and re-approach,
- lateral drift -> re-center,
- premature open -> re-close and continue.

If recovery predictions become noisy, consider separating the tasks:
- train failure detection first,
- then train recovery only on failure windows.

## Open Questions

- Should failure onset be labeled explicitly in addition to trajectory-level failure labels?
- Do we want a hierarchical model: no-failure vs failure first, then failure-type classification?
- Should recovery be a regression head, a text instruction head, or both?
- For Agilex/Piper, what sensor stream is most reliable and time-synced across demos?

## Practical Next Step

Implement the GRU experiment using the same low-dimensional windows and same train/val split as the current notebook, then compare it directly against the temporal MLP baseline.

## Initial Notebook Run

Quick comparison on the current notebook split:

- Temporal MLP: final val loss 3.2618, val acc 20.00%
- GRU temporal model: final val loss 2.9233, val acc 20.00%

The GRU is currently ahead on validation loss, which is the signal I care about most for this overfitting problem.
