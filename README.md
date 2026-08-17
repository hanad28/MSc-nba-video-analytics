# nba-video-analytics

A basketball video analytics pipeline that recovers player tracking,
team assignment, possession, passes and interceptions, court positions
in metres, and per-player speed and distance from a single monocular
broadcast video. No multi-camera array, no proprietary tracking
hardware, no sensor data: one camera, one clip, every metric.

This repository is the instrument and evidence base for an empirical
study of how accurately a single-camera pipeline can recover basketball
analytics, and where it fails. The dissertation reports the study; the
repository carries the code, the ground truth, and the measured
evidence. Submitted as the supporting material for an MSc dissertation
(Data Science and AI, Queen Mary University of London, 2026).

## Demo

[![Pipeline demo](https://img.youtube.com/vi/JtMLefkXz4Q/0.jpg)](https://youtu.be/JtMLefkXz4Q)

Eighteen seconds, three clips in sequence. Overlays: player tracks
coloured by team, ball trajectory, possession, pass and interception
captions, a tactical minimap, and per-player speed and distance. The
Release carries all three clips individually, including the two that
show the pipeline's documented failure modes.

## The pipeline

`main.py` runs nine stages in order; the court keypoint inference
(stage 7) runs at the start of the run alongside the other detectors,
and five of the nine stages cache their outputs so a re-run recomputes
only what changed.

1. Player detection and tracking (YOLOv8x + ByteTrack)
2. Ball detection, with a global-trajectory gate before any
   interpolation
3. Ball interpolation
4. Team classification (FashionCLIP zero-shot, per-track temporal
   smoothing)
5. Possession (proximity and containment with streak confirmation)
6. Passes and interceptions
7. Court keypoints (YOLOv8x-pose, 18 landmarks)
8. Homography and court mapping (no fallback: unmapped frames produce
   no positions)
9. Speed and distance (gap-suppressed: occlusions cannot masquerade as
   sprints)

Architecture, design rules and caching: `docs/pipeline_overview.md`.

## Headline results

| Stage | Result | Scope |
| --- | --- | --- |
| Tracking | IDF1 0.805, 14 identity switches (lower bound; production configuration) | 3 clips, 467 annotated boxes |
| Team classification | 94.0% effective accuracy (FashionCLIP, full-body crop; 94.4% on decided frames) | labelled frames, 3 clips |
| Possession | 67.8% frame agreement (production configuration, 488 scoreable frames) | 534 hand-labelled frames |
| Events | TP 2, FP 12, FN 5 (team criterion, tolerance 10 frames) | 3 clips against hand-labelled ground truth |
| Court keypoints | 0.9669 test pose mAP50-95 | 220-image grouped test split |

Full measured evidence, including results the dissertation does not
discuss in depth, is in `results/`, indexed by `results/README.md`.

## Reproducing

**Verify the code (no downloads).**

    pip install -r requirements.txt
    python -m pytest

1,112 tests pass with no model weights, no clips beyond those
committed, and no caches.

**Run the pipeline.** Download `ball.pt` and `keypoints.pt` from
[the latest Release](../../releases/latest) into `models/`
(see `models/README.md`), then:

    python main.py --input data/raw/clip_1.mp4

The annotated video is written to `data/outputs/`.

**Reproduce the evaluations.** The measurement scripts in `scripts/`
regenerate everything in `results/`. Some need artefacts that are not
submitted: the detection and possession caches (produced by running
`main.py` per clip) and the training datasets (public Roboflow
downloads; the training notebooks fetch them). Each script's error
message states exactly what it needs and where it normally lives.

## Repository structure

    assets/         court template image (the homography target)
    basketball/     the pipeline package (detection, team_classifier,
                    possession, events, keypoints, homography,
                    annotators, metrics, cache, labelling, utils)
    config/         default.yaml — every tunable the pipeline reads
    data/
      raw/          the three evaluation clips (committed)
      annotations/  hand-labelled ground truth (committed)
      processed/    per-clip caches (generated at run time)
      outputs/      annotated video (generated at run time)
    docs/           pipeline overview; pre-registered predictions
    evaluation/     CLEAR MOT harness (measures the pipeline)
    main.py         runs the full pipeline
    models/         weights directory (see models/README.md)
    results/        measured evidence, by stage (see results/README.md)
    scripts/        measurement, labelling and experiment instruments
    tests/          1,112 tests (verifies the code)
    training/       model training notebooks

`basketball/labelling/` lives inside the package deliberately: the
labelling tools render with the pipeline's own annotators and read the
same cached tracks, so ground truth was labelled against exactly the
representation the pipeline outputs.

## Data, ground truth and attribution

**Clips.** Three broadcast clips, committed under `data/raw/`:
clip_1 (Lakers–Clippers, 117 frames), clip_2 (Thunder–Timberwolves,
174 frames), clip_3 (Kings–Rockets, 243 frames), all 1280×720 at
30 fps, sourced from publicly available NBA broadcast footage on
YouTube.

**Ground truth.** Hand-labelled and committed under
`data/annotations/`: per-frame possession (534 frames), per-frame team
assignment, pass and interception events, tracking ground truth per
clip, and a 148-verdict keypoint audit.

**Training data.**
- Player and ball detection: Roboflow `workspace-5ujvu /
  basketball-players-fy4c2-vfsuv` v17, CC BY 4.0; trained the unified
  detector (`ball.pt`).
- Earlier ball-detection iteration: Roboflow `hanad-ali /
  nba-ball-detection-merged-4feju` v1, a merged single-class ball
  dataset built on images from a public Roboflow basketball detection
  dataset, CC BY 4.0; retained for comparison.
- Court keypoints: Roboflow `fyp-3bwmg / reloc2-den7l` v1. Licence
  field reads "Private"; publicly downloadable; attributed in full; no
  explicit reuse terms were granted by the uploader.
- Superseded player detector: Roboflow `hanad-ali /
  basketball-player-ball-detection-fjmmg` v1, CC BY 4.0; trained
  `players.pt`, retained as the tracking-evaluation comparator.

**Pretrained models.** YOLOv8 (Ultralytics, AGPL-3.0; base checkpoints
fetched automatically by the training notebooks). FashionCLIP
(`patrickjohncyh/fashion-clip`, MIT, via Hugging Face Transformers),
used zero-shot.

**Court template.** `assets/court_template.png` sourced from
shareplaypro.in, the homography target for the tactical view.

## Dissertation mapping

| Paper artefact | Produced by | Evidence |
| --- | --- | --- |
| MAPPING_PLACEHOLDER — completed when the dissertation is finalised | | |

## Licence

AGPL-3.0 (see `LICENSE`). The distributed model weights are fine-tuned
YOLOv8 checkpoints, and Ultralytics licenses YOLOv8 and its derivative
weights under AGPL-3.0.
