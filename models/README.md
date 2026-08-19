# Model weights

The pipeline uses two trained models. Neither is committed to the
repository, because both exceed GitHub's 100 MB file limit. Both are
attached to the repository's Release, together with a third checkpoint
used only in the tracking evaluation.

Download all three from [the latest Release](../../releases/latest)
and place them in this directory.

| File | Size | SHA-256 | Role |
| --- | --- | --- | --- |
| `ball.pt` | 137 MB | `5475b7cf…c60c5d` | Production detector. One seven-class YOLOv8x checkpoint serves both player and ball detection; each detector resolves its class index from the checkpoint's own names at run time. |
| `keypoints.pt` | 140 MB | `b428fa9d…9d3e80` | Court keypoint model (YOLOv8x-pose, 18 keypoints). The checkpoint Ultralytics saved at epoch 494 of the 500-epoch run. |
| `players.pt` | 137 MB | `e3b2f621…13681c` | Not used by the pipeline. Two-class detector retained as the comparison configuration in the tracking evaluation (`scripts/run_evaluation.py`). |

**Licence position on `keypoints.pt`.** It was trained on Roboflow
`fyp-3bwmg / reloc2-den7l` v1, whose licence field reads "Private" and
which grants no explicit reuse terms; the dataset is attributed in full
in the root `README.md`, and the checkpoint is published here for
non-commercial academic assessment of this dissertation. The three
checkpoints are fine-tuned YOLOv8 weights, which Ultralytics licenses
under AGPL-3.0.

Full hashes are in `checksums.txt` in this directory. Verify a download with:

    sha256sum -c checksums.txt

`main.py` expects `models/ball.pt` and reads the keypoint path from
`config/default.yaml`. With the two production weights in place, the
pipeline runs end to end; `players.pt` is only needed to reproduce the
tracking comparison.

Training records for both production models, including per-epoch
metrics and the run configurations, are in `results/training/`. The
training notebooks themselves are in `training/`.
