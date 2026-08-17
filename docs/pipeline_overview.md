# Pipeline overview

The pipeline takes a single broadcast video clip and produces an
annotated video plus per-frame analytics: player tracks, ball
trajectory, team assignments, possession, passes and interceptions,
court positions in metres, and per-player speed and distance.
`main.py` runs all nine stages, with one ordering note: the court
keypoint inference (stage 7) runs at the start of the run, grouped
with the other two detectors, and its output is consumed at stage 8.
Five stages cache their outputs (1, 2, 4, 5 and 7), so a re-run
recomputes only what changed; the remaining stages are cheap
derivations recomputed every run.

## Stages

**1. Player detection and tracking.** A YOLOv8x detector finds players
in every frame; ByteTrack links detections into per-player tracks.
Since the unified-detector adoption, player detection runs on the same
seven-class checkpoint as ball detection (`models/ball.pt`), with the
class index resolved from the checkpoint's own names at run time.

**2. Ball detection.** The same checkpoint, ball class. Raw detections
pass through a global-trajectory gate that rejects implausible jumps
before any interpolation happens; gating before interpolation matters,
because interpolating first manufactures ball positions the gate would
then trust.

**3. Ball interpolation.** Gaps left after gating are filled by
interpolation, then backward and forward fill, so every frame carries a
ball position.

**4. Team classification.** FashionCLIP assigns each tracked player to
a team by zero-shot classification of jersey crops, with per-track
temporal vote smoothing. Two comparators (K-means colour clustering
and an embedding-clustering variant) are implemented alongside for the
evaluation; each method uses its own measured-best crop fraction.

**5. Possession.** Frame-level possession is assigned from
ball-to-player proximity and bounding-box containment; a candidate
holder must persist for a consecutive-frame streak, backed by at least
one genuine (non-interpolated) ball detection, before possession is
confirmed.

**6. Events.** Possession changes are classified into passes
(same-team) and interceptions (opposing-team). A change is paired into
an event only when the new holder appears within a fixed maximum gap
of the last holder's final frame; beyond that gap the carried holder
is stale, and the transition is left unpaired rather than fabricated.

**7. Court keypoints.** A YOLOv8x-pose model (`models/keypoints.pt`)
predicts 18 court landmarks per frame, with a per-keypoint confidence
threshold selecting which are usable. The inference pass itself runs
at the start of the run, alongside the other detectors.

**8. Homography and court mapping.** Frames with at least four usable
keypoints yield a homography to a metric court template; player feet
positions map through it to court coordinates in metres. Frames
without a valid homography produce no positions: there is no fallback,
by design, so downstream consumers see an absence rather than a stale
or invented value.

**9. Speed and distance.** Per-player displacement between mapped
positions gives distance; a sliding window gives speed. A displacement
spanning a tracking or mapping gap longer than five frames is
suppressed rather than averaged, so a long occlusion cannot masquerade
as a sprint.

## Design rules that recur

Two rules shape most of the pipeline's edges. Measurements that cannot
be made are absent, never defaulted: unmapped frames yield no
positions, suppressed speeds yield no reading, and the annotators draw
nothing for them. And correctness gates run before any value is
manufactured: the ball gate precedes interpolation, the keypoint
threshold precedes homography.

## Caching

The five caching stages write their outputs under
`data/processed/<clip>/` with a fingerprint recording the code
revision and upstream inputs that produced them. A stage re-runs when
its fingerprint no longer matches. Three revision-constant names
(`INFERENCE_REVISION` on the detectors and the keypoint model,
`CLASSIFIER_REVISION` on the team classifiers, `LOGIC_REVISION` on the
possession tracker) invalidate caches when behaviour changes in ways
the fingerprint's inputs cannot see; bump the relevant constant when
changing a stage's behaviour in place.

## Configuration

`config/default.yaml` holds every tunable the pipeline reads, with a
comment per key stating its effect and constraints. Detector weights
are set in `main.py`; the keypoint model path is a config key.

## Running

    python main.py --input data/raw/clip_1.mp4

`--config` defaults to `config/default.yaml` and `--output` to
`data/outputs/`. A run expects the two production weights in `models/`
(see `models/README.md`) and writes the annotated video to
`data/outputs/`. The test suite (1,112 tests) runs without weights,
clips or caches: `python -m pytest`.
