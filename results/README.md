# Results

Measured evidence for the study, organised by pipeline stage. Each
folder holds the outputs of the measurement scripts and notebooks named
below, as run against the three evaluation clips. The dissertation
discusses a subset of these results in depth; the remainder are
reported here.

Nearly every figure was produced by a script or notebook in this
repository. Two files are transcriptions rather than regenerable
outputs and say so in a `source` column, and one recorded diagnostic is
noted below; everything else can be reproduced by running the named
instrument where its inputs exist (see each script's error message for
what it needs).

| Folder | Contents | Produced by |
| --- | --- | --- |
| `tracking/` | CLEAR MOT comparison across five tracker configurations (results, identity switches) | `scripts/run_evaluation.py` |
| `team_classification/` | Five-arm comparison summary; per-frame inference grid (61 files); abstention and margin measurements | `scripts/team_classification_sweep.ipynb`, `scripts/inference_grid.ipynb`, `scripts/measure_abstention_and_margins.ipynb` |
| `possession/` | Possession share per clip; threshold sweep, plain and gated | `scripts/possession_sweep.ipynb` |
| `events/` | Pass and interception scores against ground truth; gap sweep; tolerance sensitivity | `scripts/event_scoring.ipynb` |
| `keypoints/` | Reprojection and stability measurements; per-frame sufficiency; grouped test-split evaluation; the two-run training comparison; the discarded run's dropped-image record | `scripts/measure_court_keypoints.py`, `scripts/measure_test_split.py`, `scripts/measure_keypoint_runs.py` |
| `ball_detection/` | Detection-gate candidate evaluation; production-gate verification trace | `scripts/evaluate_gate_candidates.py` (evaluation); the verification trace is a recorded diagnostic (see note) |
| `homography/` | Court-mapping counts per clip | `scripts/measure_mapping_and_metrics.py` |
| `metrics/` | Speed and distance counts per clip | `scripts/measure_mapping_and_metrics.py` |
| `training/` | Per-epoch metrics, run configurations and validation plots for the two production training runs and the detection validation run | Ultralytics training and validation, run from the notebooks in `training/` |
| `camera_motion/` | Per-frame global camera motion of the three evaluation clips: horizontal displacement of the image centre and scale change, per frame pair | `scripts/measure_camera_motion.py` |

Notes an examiner may want:

- `keypoints/discarded_run_dropped_images.csv` records a training run
  discarded after a label-coordinate defect caused Ultralytics to drop
  a quarter of the training images silently. Two rows are transcribed
  from the discarded run's scan cache, which no longer exists; two rows
  were measured on 15 August 2026 from the rebuilt dataset's scan
  cache, and can be re-derived by rebuilding the grouped split with
  training/split_court_keypoints.py, though not by running anything in
  this repository as it stands.
- `training/detect_val/metrics.csv` is a transcription of the adopted
  ball detector's test-split metrics; the validation run wrote plots
  but no metrics table.
- `ball_detection/gate_verification/clip3_178_190_metadata.csv` is a
  recorded per-frame decision trace of the production gate on clip_3's
  occlusion window (frames 178 to 190). The rendering tool that wrote
  it is not part of the shipped pipeline, so the file is a recorded
  diagnostic rather than a regenerable output; the gate logic it traces
  lives in `basketball/detection/ball_interpolation.py`.
- In `training/court_kp_e500/`, the four `Pose*_curve.png` files are
  byte-identical to their `Box*` counterparts. Both curve families are
  flat at 1.0 for this single-instance task, so they render to the same
  image; the discriminating metric is mAP50-95, reported in
  `results.csv` and the run comparison.
- `keypoints/stage7_run_comparison.csv` reports two epochs per run:
  the epoch where pose mAP50-95 peaked, and the epoch whose weights
  Ultralytics saved as `best.pt`. These differ because Ultralytics
  selects on a combined box-and-pose fitness. The deployed
  `keypoints.pt` is the saved checkpoint, epoch 494.
