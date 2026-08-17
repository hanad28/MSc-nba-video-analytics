# Court Keypoint Detection — Evaluation Specification (Stage 7)

Written 13 August 2026, before either training run completed and before any
result was read. Predictions K1–K5 and every scoring rule below are fixed at
the time of writing. Refuted predictions are recorded as refuted and never
rewritten, following `event_detection_spec.md`.

---

## 1. What this stage produces and why it is measured this way

`CourtKeypoints` locates 18 court landmarks in each frame. Its output is not an
end in itself: it supplies the image-to-template correspondences that Stage 8's
homography consumes. That dependency determines what "accurate" has to mean
here, and it is not one quantity but three.

**Localisation.** How close is a predicted keypoint to the true image position
of its landmark? This is what determines homography residual, and it is
continuous.

**Identification.** Is keypoint index *k* on the right landmark at all? A
basketball court is near mirror-symmetric, so the plausible failure is not a
displaced point but a correctly-placed point carrying the wrong index. This is
categorical.

**Sufficiency.** Does a given frame carry enough correct correspondences to
compute a homography at all? This is a per-frame binary and is the only one of
the three that Stage 8 consumes directly.

These come apart. A model can localise well while mislabelling indices, and
sufficiency can hold while both degrade. They are measured separately below.

**Why identification cannot be measured by the same method as localisation.**
The obvious label-free proxy for localisation is reprojection: fit a homography
from a subset of predicted keypoints, reproject the held-out ones, measure the
residual. That works, costs no annotation, and runs densely over every frame.
But a systematic left-right index swap maps to the *mirrored* court and is
therefore perfectly self-consistent — reprojection residuals would be near
zero. The proxy is structurally blind to the exact failure the court's symmetry
invites, and blind in the direction of a falsely clean result. Identification
therefore requires human verdicting, and the verdicted audit in §4 exists for
that reason alone.

---

## 2. The keypoint layout

Established by visually identifying every landmark on rendered training frames,
with 16 of 18 mirror pairs independently confirmed by reflection agreement
across 20 wide-angle frames.

| Index | Landmark |
|---|---|
| 0–5 | Left baseline, far sideline to near: corner, three-point intersection, lane corner, lane corner, three-point intersection, corner |
| 6 | Centre line at near sideline |
| 7 | Centre line at far sideline |
| 8, 9 | Left free-throw line, far and near lane corners |
| 10–15 | Right baseline, near sideline to far, mirroring 0–5 in reverse |
| 16, 17 | Right free-throw line, far and near lane corners |

Mirror mapping: `[15, 14, 13, 12, 11, 10, 6, 7, 16, 17, 5, 4, 3, 2, 1, 0, 8, 9]`.

The 5↔10 pair is confirmed by direct visual identification of each point's
court position, not by reflection agreement: the two are never co-visible in
this dataset, so no frame can exhibit the pairing. Recorded as a limitation of
the confirmation method, not a doubt about the pair.

---

## 3. Data and its two corrected defects

Roboflow `fyp-3bwmg/reloc2` v1, 1,468 images, 18 keypoints with three-state
visibility flags (0 absent, 1 occluded, 2 visible). Across all keypoints:
15,344 absent, 1,042 occluded, 10,056 visible. Ultralytics trains on states 1
and 2, excluding 0.

Frames carrying at least four visible keypoints: 95.9% of train, 94.1% of
valid, 97.3% of test as originally split. Homography is therefore viable on the
overwhelming majority of frames, and the "fewer than four correspondences"
fallback anticipated in the design is a genuine edge case rather than
the normal path.

**Defect 1: split contamination.** Roboflow splits randomly over images, but
images are frames sampled from source videos. Measured overlap by source video:
176 of 222 valid images (79%) and 57 of 74 test images (77%) come from videos
also present in train. Because `best.pt` is selected on validation fitness,
this is a model-selection defect and not merely a reporting one. Corrected by
`training/split_court_keypoints.py`, which regroups whole source videos into
70/15/15 (1,028 / 220 / 220 images across 65 / 55 / 55 videos, zero source
overlap verified on disk).

Consequence to disclose: `basketball_2` is 350 images, 23.8% of the dataset, so
it must sit entirely in train and the training distribution is skewed toward one
game's court and camera. Valid and test are each built from 55 distinct source
videos, which makes them more diverse than the training set, not less.

**Defect 2: flip_idx.** Roboflow ships the identity mapping, which passes
Ultralytics' only validation check (length equals keypoint count) but renumbers
nothing under horizontal flip. With `fliplr` active, roughly half of training
images would teach the model that each index means either of two
mirror-symmetric positions. Corrected as above. This defect was found by
inspection, not by any metric, and no metric computed on the contaminated
validation set would have revealed it.

**Uncorrected and carried as a stated risk:** the dataset's generation applies
adaptive contrast equalisation to every image. The three evaluation clips are
raw BGR frames. This is a train/inference distribution mismatch introduced by
the dataset, tested as an inference-time ablation in §5 rather than by
retraining.

---

## 4. Evaluation design

### 4.1 In-distribution: grouped test split

Standard pose metrics from Ultralytics on the 220-image grouped test set:
pose mAP50 and mAP50-95, plus box mAP for the court instance. Reported for both
training runs.

This measures generalisation to unseen *source videos* within the dataset's own
domain. It is not a measure of performance on the evaluation clips and is never
reported as one.

### 4.2 Localisation on the evaluation clips: leave-one-out reprojection

Label-free, dense across all 534 frames of the three clips.

For each frame with at least five predicted keypoints above the confidence
threshold: for each keypoint *k*, fit a homography from all other predicted
keypoints to their template positions, reproject *k*, and record the Euclidean
residual in pixels. Report the median and interquartile range per clip and per
keypoint index.

The court template's metric coordinates come from NBA regulation dimensions
already fixed in the project (28.65 m × 15.24 m), so template positions are
derived rather than measured.

**Stated limitation, in advance:** this measures internal geometric consistency,
not agreement with truth. A globally mirrored or systematically distorted
prediction set can produce low residuals. It is a necessary condition for
correctness, not a sufficient one, which is why §4.4 exists.

### 4.3 Sufficiency

Per frame, the count of predicted keypoints above the confidence threshold, and
the proportion of frames reaching four or more. Reported per clip. This is the
quantity committed to in June as the homography fallback
trigger rate, and it is the only measure here that Stage 8 consumes directly.

### 4.4 Identification: verdicted audit

**25 frames**, sampled evenly across the three clips, rendered with predicted
keypoints drawn as labelled dots. For every visible predicted keypoint the
human answers one question: is this index on the correct landmark?

Answers: `correct`, `wrong_landmark`, `not_on_court`. Expected volume is roughly
200 verdicts at the dataset's modal 8 keypoints per frame.

Verdicts are recorded per `(clip, frame_idx, keypoint_index)` to a CSV in
`data/annotations/`, appended per verdict, with last-wins dedup on that key —
the same persistence contract as the possession and team ground-truth tools.

Frames are presented in a fixed shuffled order (seed 42), matching the team
ground-truth tool rather than the possession tool. The reasoning differs by
task: possession is a temporal judgement needing neighbouring-frame context,
whereas a keypoint's correctness is judged from the single frame, and shuffling
prevents a verdict on one frame propagating to the visually near-identical next.

**These verdicts are not reusable across methods.** They are verdicts on one
model's output, so any comparator requires its own audit. This is the primary
reason a Hough-transform comparator is argued from the literature in related
work rather than implemented and measured (see §7).

### 4.5 Temporal stability

Per-keypoint frame-to-frame displacement across each clip, reported as a
distribution. The broadcast camera pans but does not cut within a clip, so a
stable landmark should move smoothly; large isolated jumps indicate jitter. This
addresses the keypoint-jitter risk identified in June, and it
requires no labels.

---

## 5. Ablations

Each is a re-run over cached predictions or a second inference pass, not a
retrain.

**A1 — training schedule.** Run A (`epochs=100`, `patience=30`) against Run B
(`epochs=500`, `patience=100`), identical in every other respect. Compared on
§4.1 and §4.2. Motivated by the 27 June 2026 finding on this project's own data
that a 250-epoch schedule converged worse than 100, mechanistically because
Ultralytics computes its learning-rate schedule as a fraction of total epochs,
so a longer run is a different trajectory rather than more time on the same one.
That finding was on a detection task and is tested here rather than assumed.

**A2 — contrast equalisation.** Inference on raw clip frames against inference
on CLAHE-equalised frames, comparing §4.2 and §4.3. Tests whether the dataset's
baked-in equalisation causes a measurable domain gap.

**A3 — filtering policy.** Per-keypoint confidence thresholding against
geometric consistency filtering (discarding keypoints inconsistent with the
court template's known proportions), compared on §4.2 and §4.3. These are the
two available answers to the question of which predicted keypoints to trust, and
which is adopted determines what Stage 8 receives.

---

## 6. Pre-registered predictions

**K1.** Run A reaches higher pose mAP50-95 on the grouped test set than Run B.
*Basis:* the 27 June finding plus the LR-annealing mechanism. Confidence
moderate, not high: that finding was on detection, and 1,028 images is small
enough that a longer schedule could plausibly win.

**K2.** Neither run triggers early stopping. *Basis:* `patience=30` never fired
on the fy4c2 detection run. Refutation would indicate a real plateau and that
schedule length, not training duration, is doing the work.

**K3.** Per-keypoint verdict accuracy in §4.4 is lower for the near-sideline
corner indices (4, 5, 10, 11) than for the lane corner indices (8, 9, 16, 17).
*Basis:* the corners are the rarest labels in the dataset at 267, 284, 320 and
354 instances respectively, and sit at frame edges where they are frequently
clipped.

**K4.** Both models degrade on the three evaluation clips relative to the
grouped test set, on any comparable measure. *Basis:* the equalisation mismatch
in §3 plus a different broadcast style. Direction predicted; magnitude not.
Registered as near-safe, and earning its place only by fixing the magnitude
question before the measurement exists.

**K5.** Fewer than 5% of verdicted keypoints in §4.4 are `wrong_landmark` with
the correct mirror index — that is, mirror confusion specifically. *Basis:* the
corrected `flip_idx` should have eliminated the mechanism. Refutation would mean
either the mapping is wrong or symmetry confusion has a second cause.

---

## 7. What is deliberately not done

**No Hough-transform comparator is implemented.** Classical line detection
recovers court geometry without landmark identity, and identity is precisely
what homography needs, so turning Hough output into correspondences requires a
model-fitting stage that is a project in itself. A comparator would also need
its own verdicted audit, doubling §4.4. Weighed against a likely-uninformative
result — learned keypoint regression outperforming classical line-fitting on
broadcast footage is largely settled in the literature — this fails the test the
K-means baseline passed, which was isolating a mechanism rather than
establishing a ranking. Hough is argued from the literature in related work as
the classical approach the learned formulation supersedes, with the identity
argument given as the reason.

**No re-tuning of upstream stages.** Possession is closed as a disclosed
limitation and events are measured. Stage 7 does not consume either.

**No confidence intervals on the verdicted audit.** Roughly 11 observations per
keypoint index; counts are primary, rates secondary, and interval estimates at
this scale would be theatre.

---

## Amendment to §4.4, 13 August 2026

Recorded before the labelling session began and before any verdict existed.
§4.4's original text stands above unchanged; this describes what the tool built
against it actually does and why it differs.

**Answer set widened from three to five.** §4.4 lists `correct`,
`wrong_landmark` and `not_on_court`. The tool adds `unclear` and `stop`.
`unclear` mirrors the possession and team ground-truth tools, where forcing a
judgement the labeller cannot make writes noise into the reference rather than
recording the ambiguity. `stop` is session control rather than a verdict and is
never persisted as one.

**A follow-up prompt on a wrong verdict, and a new `actual_index` column.** K5
asks what fraction of identification errors are mirror confusion specifically.
That is unanswerable from a bare `wrong_landmark` verdict, because it records
that an index is misplaced without recording where it actually sits. On a `w`
verdict only, the tool asks which index the point really lies on, accepting
0-17 or `?` for cannot tell, and writes it to `actual_index`. The column is
empty for every other verdict.

Without this, K5 could not be scored at all, so the amendment is a correction to
an incomplete pre-registration rather than a change of design.

**Per-verdict confidence is recorded.** Not specified in §4.4. K5 and the A3
ablation both relate verdicts to the model's confidence, and re-deriving it
afterwards would depend on the keypoint cache still matching the session. Cheap
to record now, unrecoverable later.

**Expected volume revised down.** §4.4 estimates roughly 200 verdicts from the
dataset's modal 8 keypoints per frame. The measurement run observed a median of
6 to 7 confident keypoints on the evaluation clips, so 150 to 175 is the
realistic figure. The frame count is unchanged at 25 (8 / 8 / 9).

**Sampling eligibility stated explicitly.** §4.4 does not say which frames are
eligible. Frames are sampled from those carrying at least one confident
keypoint, not those carrying four or more. Restricting to well-populated frames
would exclude precisely where errors are suspected: the observation that
motivated sharpening this audit came from a clip_3 frame with a single confident
keypoint, placed visibly off the line at confidence 1.0.
