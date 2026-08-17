"""
mot_metrics.py

Computes CLEAR MOT evaluation metrics (MOTA, MOTP, IDF1) by comparing
pipeline tracking outputs against manually annotated ground truth, via the
py-motmetrics library. Evaluation runs only on frames that carry ground
truth rows: the ground truth is sparsely sampled (every 10th frame), and
a frame without GT rows is unlabelled, not empty; scoring it would turn
every correct detection there into a false positive.
"""
from __future__ import annotations

from dataclasses import dataclass

import motmetrics as mm
import numpy as np
import pandas as pd

from basketball.detection.player_detector import PlayerTrack
from evaluation.ground_truth import GTAnnotation

METRIC_NAMES = [
    'mota', 'motp', 'idf1', 'num_switches', 'num_false_positives',
    'num_misses', 'num_objects', 'num_matches', 'precision', 'recall',
]


@dataclass
class MOTResult:
    mota: float
    motp: float            # mean IoU overlap of matched pairs; see MOTEvaluator docstring
    idf1: float
    num_switches: int
    num_false_positives: int
    num_misses: int
    num_objects: int
    num_matches: int
    precision: float
    recall: float


def _xyxy_to_xywh(bbox: list[float]) -> list[float]:
    """Convert an [x1, y1, x2, y2] box to the [x, y, width, height] form motmetrics expects."""
    x1, y1, x2, y2 = bbox
    return [x1, y1, x2 - x1, y2 - y1]


def _zero_if_nan(value: float) -> float:
    """Sensible zero for ratio metrics that motmetrics leaves undefined (NaN) on empty input."""
    return 0.0 if np.isnan(value) else float(value)


def _zero_result() -> MOTResult:
    """The well-formed result for an evaluation with no labelled frames: never NaN, never an exception."""
    return MOTResult(
        mota=0.0, motp=0.0, idf1=0.0, num_switches=0, num_false_positives=0,
        num_misses=0, num_objects=0, num_matches=0, precision=0.0, recall=0.0,
    )


class MOTEvaluator:
    """
    Evaluates multi-object tracking against ground truth using CLEAR MOT
    metrics (MOTA, MOTP, IDF1). MOTResult.motp carries the overlap
    convention: mean IoU of matched pairs, higher is better, converted from
    motmetrics' native distance convention (mean 1 - IoU, lower is better).
    Both conventions appear in the literature, and comparing across them
    silently is a real error; any number cited from here is an overlap.
    """

    IOU_THRESHOLD = 0.5

    def _iou_distance_matrix(self, gt_boxes: list[list[float]], tracker_boxes: list[list[float]]) -> np.ndarray:
        """Pairwise 1 - IoU distances over [x, y, width, height] boxes, with pairs below IOU_THRESHOLD set to NaN."""
        # Reproduces mm.distances.iou_matrix(..., max_iou=1 - IOU_THRESHOLD)
        # exactly, using the mm.distances.boxiou primitive it wraps:
        # iou_matrix() itself calls np.asfarray, which numpy 2.0 removed, and
        # motmetrics 1.4.0 (the latest release) ships no fix; it is unusable
        # against the pinned numpy.
        max_distance = 1.0 - self.IOU_THRESHOLD
        matrix = np.empty((len(gt_boxes), len(tracker_boxes)))
        for i, gt_box in enumerate(gt_boxes):
            for j, tracker_box in enumerate(tracker_boxes):
                distance = 1.0 - mm.distances.boxiou(np.asarray(gt_box), np.asarray(tracker_box))
                matrix[i, j] = np.nan if distance > max_distance else distance
        return matrix

    def _build_accumulator(
        self,
        ground_truth: list[GTAnnotation],
        tracks: list[dict[int, PlayerTrack]],
    ) -> mm.MOTAccumulator | None:
        """Association events for one sequence over exactly its labelled frames, or None when the GT is empty."""
        gt_by_frame: dict[int, list[GTAnnotation]] = {}
        for annotation in ground_truth:
            gt_by_frame.setdefault(annotation.frame, []).append(annotation)

        if not gt_by_frame:
            return None

        # A ground-truth frame beyond the tracker output means mismatched
        # inputs (the wrong clip's tracks, or a truncated run), not a tracker
        # that produced nothing, and must be surfaced rather than scored.
        max_gt_frame = max(gt_by_frame)
        if max_gt_frame >= len(tracks):
            out_of_range = sum(1 for frame in gt_by_frame if frame >= len(tracks))
            raise ValueError(
                f'Ground truth extends beyond the tracker output: highest GT frame index '
                f'{max_gt_frame} against {len(tracks)} tracker frames — {out_of_range} of '
                f'{len(gt_by_frame)} GT frames out of range. Mismatched inputs (wrong '
                f'clip\'s tracks, or a truncated run), not a tracker that produced nothing.'
            )

        accumulator = mm.MOTAccumulator(auto_id=False)

        for frame in sorted(gt_by_frame):
            annotations = gt_by_frame[frame]
            tracker_frame = tracks[frame]

            gt_ids = [annotation.track_id for annotation in annotations]
            gt_boxes = [_xyxy_to_xywh(annotation.bbox) for annotation in annotations]
            tracker_ids = list(tracker_frame.keys())
            tracker_boxes = [_xyxy_to_xywh(track.bbox) for track in tracker_frame.values()]

            distances = self._iou_distance_matrix(gt_boxes, tracker_boxes)
            accumulator.update(gt_ids, tracker_ids, distances, frameid=frame)

        return accumulator

    def _result_from_row(self, row: pd.Series) -> MOTResult:
        """Convert one motmetrics summary row into a MOTResult, applying the overlap and NaN conventions."""
        # motmetrics reports motp as mean IoU distance; convert to overlap
        # before the NaN guard so an empty match set reads 0.0, not 1.0.
        motp_overlap = _zero_if_nan(1.0 - row['motp'])

        return MOTResult(
            mota=_zero_if_nan(row['mota']),
            motp=motp_overlap,
            idf1=_zero_if_nan(row['idf1']),
            num_switches=int(row['num_switches']),
            num_false_positives=int(row['num_false_positives']),
            num_misses=int(row['num_misses']),
            num_objects=int(row['num_objects']),
            num_matches=int(row['num_matches']),
            precision=_zero_if_nan(row['precision']),
            recall=_zero_if_nan(row['recall']),
        )

    def evaluate(self, ground_truth: list[GTAnnotation], tracks: list[dict[int, PlayerTrack]]) -> MOTResult:
        """Compute CLEAR MOT metrics over exactly the frames that carry ground-truth rows."""
        accumulator = self._build_accumulator(ground_truth, tracks)

        # No labelled frames means nothing can be scored; the contract is a
        # well-formed all-zero result, never NaN and never an exception.
        if accumulator is None:
            return _zero_result()

        summary = mm.metrics.create().compute(accumulator, metrics=METRIC_NAMES, name='evaluation')
        return self._result_from_row(summary.loc['evaluation'])

    def evaluate_pooled(
        self,
        sequences: list[tuple[list[GTAnnotation], list[dict[int, PlayerTrack]]]],
    ) -> MOTResult:
        """Compute CLEAR MOT metrics pooled across sequences at the event level, never an average of per-clip metrics."""
        # Averaging MOTA across clips of different sizes is wrong: pooling must
        # accumulate association events. compute_many's OVERALL row merges the
        # per-sequence accumulators with object/hypothesis IDs kept distinct per
        # sequence, so identities never bleed between clips.
        accumulators = [
            accumulator
            for ground_truth, tracks in sequences
            if (accumulator := self._build_accumulator(ground_truth, tracks)) is not None
        ]

        if not accumulators:
            return _zero_result()

        summary = mm.metrics.create().compute_many(
            accumulators,
            metrics=METRIC_NAMES + ['num_detections'],
            names=[str(index) for index in range(len(accumulators))],
            generate_overall=True,
        )
        pooled = self._result_from_row(summary.loc['OVERALL'])
        pooled.motp = self._pooled_motp(summary.drop(index='OVERALL'))
        return pooled

    def _pooled_motp(self, per_sequence: pd.DataFrame) -> float:
        """Matched-pair-weighted mean MOTP (overlap) across sequences, immune to matchless-sequence NaNs."""
        # motmetrics aggregates OVERALL motp as a detection-weighted mean in
        # which a matchless sequence contributes NaN * 0 = NaN, poisoning the
        # pooled value, which the NaN guard would then mask as a misleading
        # 0.0. Recomputed here from the per-sequence rows instead, weighting
        # each sequence's MOTP by its matched-pair count (motmetrics'
        # num_detections, the count MOTP actually averages over; num_matches
        # would drop switch-frame pairs from the weight) and skipping
        # sequences with no matched pairs entirely. With no matched pair
        # anywhere, 0.0 is then genuinely correct rather than misleading.
        total_pairs = per_sequence['num_detections'].sum()
        if total_pairs == 0:
            return 0.0

        weighted_overlap = sum(
            (1.0 - row['motp']) * row['num_detections']
            for _, row in per_sequence.iterrows()
            if row['num_detections'] > 0
        )
        return float(weighted_overlap / total_pairs)

    def switch_frames(self, ground_truth: list[GTAnnotation], tracks: list[dict[int, PlayerTrack]]) -> list[int]:
        """Frames on which an identity switch is scored; sampled GT can only lower-bound the true switch count."""
        accumulator = self._build_accumulator(ground_truth, tracks)
        if accumulator is None:
            return []

        events = accumulator.mot_events
        switch_events = events[events['Type'] == 'SWITCH']
        return sorted({int(frame) for frame, _ in switch_events.index})
