"""
classifier.py

Team assignment via FashionCLIP zero-shot classification (primary), K-means
colour clustering (baseline), and prompt-free FashionCLIP embedding
clustering (third arm). All three share the TeamClassifier interface for
evaluation parity.
"""
from __future__ import annotations

import csv
from abc import ABC, abstractmethod
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from transformers import CLIPModel, CLIPProcessor

from basketball.cache.cache_utils import data_digest, load_valid_cache, save_cache_with_meta
from basketball.detection.player_detector import PlayerTrack

# Outer ring thickness as a fraction of min(crop_h, crop_w). 10% is thick
# enough that a few corner pixels sitting on another player or the crowd
# cannot dominate the background estimate the way four corner samples
# would, while still leaving the jersey interior for the foreground mean.
BORDER_RING_FRACTION = 0.10
# Minimum foreground pixels after background removal: below this the crop
# is unusable and we abstain (team 0), never silently fall back to
# background-contaminated pixels (the abstention contract).
MIN_FOREGROUND_PIXELS = 10

# Shared by all three classifiers' raw-prediction artefacts: one row per
# player per frame, predicted_team is the RAW pre-threshold decision (never
# 0; see each classifier's placeholder note for crop_ok=False rows), so the
# confidence threshold and any aggregation policy can be swept offline with
# zero re-inference.
PREDICTION_CSV_HEADER = ['clip', 'frame_idx', 'player_id', 'predicted_team', 'confidence', 'crop_ok']

# FashionCLIPClassifier's temporal aggregation policies. 'none' decides each
# frame on its own raw prediction; 'window_weighted' is the adopted centred
# confidence-weighted sliding window (see _aggregate_window_weighted).
AGGREGATION_POLICIES = ('none', 'window_weighted')


def _write_prediction_csv(rows: list[tuple], path: str) -> None:
    """Write the raw per-player-per-frame prediction rows to CSV, creating the parent directory if needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(PREDICTION_CSV_HEADER)
        writer.writerows(rows)


class TeamClassifier(ABC):
    """Abstract interface for assigning team membership to tracked players."""

    @abstractmethod
    def assign_teams(
        self,
        video_frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
        cache_path: str | None = None,
        record_path: str | None = None,
        clip_name: str | None = None,
    ) -> list[dict[int, int]]:
        """Assign team IDs to tracked players across all frames.

        Returns a list of per-frame dicts mapping player_id to team_id.
        team_id is 1 or 2. team_id 0 means unresolvable (implementation-dependent).

        record_path, when given (with clip_name required alongside it), writes
        the raw pre-threshold prediction for every player in every frame to a
        CSV at that path, a side channel, not a mode switch: the returned
        assignment list is unaffected by whether recording is enabled.
        """


class FashionCLIPClassifier(TeamClassifier):
    """Assigns teams via FashionCLIP zero-shot matching of jersey crops against team descriptions."""

    # Bumped whenever the inference call, the batching/caching mechanics or
    # anything else the parameter keys below cannot see changes; same rule
    # as PlayerDetector/BallDetector.INFERENCE_REVISION and
    # PossessionTracker.LOGIC_REVISION. 1: GPU placement, cached text
    # embeddings and batched crops. 2: use_assignment_cache defaults to
    # off, which changes assignments under the earlier default and must
    # invalidate every cache written under it. 3: aggregation defaults to
    # 'window_weighted', for the same reason.
    CLASSIFIER_REVISION = 3

    def __init__(
        self,
        team_1_description: str,
        team_2_description: str,
        reset_interval: int,
        crop_fraction: float,
        confidence_threshold: float,
        use_assignment_cache: bool = False,
        aggregation: str = 'window_weighted',
        aggregation_window: int = 9
    ) -> None:
        if aggregation not in AGGREGATION_POLICIES:
            raise ValueError(
                f'Unknown aggregation {aggregation!r}; expected one of {sorted(AGGREGATION_POLICIES)}.'
            )
        if aggregation_window < 1:
            # A zero or negative window empties every rolling window, which
            # ties w1 == w2 == 0 and would silently label EVERY player team
            # 1 rather than failing, the config typo this guard exists for.
            raise ValueError(f'aggregation_window must be at least 1, got {aggregation_window}.')
        if use_assignment_cache and aggregation != 'none':
            # The sweep measured these as ALTERNATIVE aggregation policies
            # ('memoise_reset' vs 'window_weighted'), never stacked. Raising
            # rather than silently letting one override the other follows
            # main.py's precedent for an unrecognised method: never report
            # results from a policy the caller did not ask for.
            raise ValueError(
                f'use_assignment_cache=True cannot be combined with aggregation={aggregation!r} — '
                f'they are alternative aggregation policies; set aggregation=\'none\' to use the cache.'
            )

        self.team_1_description = team_1_description
        self.team_2_description = team_2_description
        self.reset_interval = reset_interval
        self.crop_fraction = crop_fraction
        self.confidence_threshold = confidence_threshold
        # Off by default: the 11 August 2026 sweep measured this as the
        # 'memoise_reset' aggregation policy and it cost 2.7pp of mean
        # held-out effective accuracy against deciding every frame fresh
        # (0.911 vs 0.938), so it must not be active in production.
        # reset_interval has no effect while this is False: every frame is
        # classified independently and assignment_cache is never consulted
        # or populated. Kept, not deleted: it is a measured comparator the
        # sweep result depends on still existing, reachable via True.
        self.use_assignment_cache = use_assignment_cache
        # 'window_weighted' by default: the same sweep measured centred
        # confidence-weighted windowing at 0.9443 mean held-out effective
        # accuracy against 0.9242 for per-frame decisions, and ~20x less
        # visible label flicker on the worst clip. 'none' (per-frame) stays
        # reachable as the measured comparator.
        self.aggregation = aggregation
        self.aggregation_window = aggregation_window
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.processor = None
        self.text_embeds: np.ndarray | None = None  # (2, dim), L2-normalised; set once in _load_model()
        self.assignment_cache: dict[int, int] = {}
        self.empty_crop_count = 0
        # Single-entry memo for _encode_crops(): _encode_cache_frame holds a
        # strong reference to the exact frame array the memo was computed
        # for, so the cache key is an identity check (`is`), not an id()
        # comparison. id(frame) alone is unsafe here: once a frame array is
        # garbage-collected, CPython is free to reuse its memory address for
        # an unrelated object, so a later, different frame could collide
        # with a stale id() and produce a false cache hit. Holding the
        # reference keeps the cached frame alive for as long as it is the
        # memo key, making that address reuse impossible by construction,
        # and caps memory at exactly one retained frame.
        self._encode_cache_frame: np.ndarray | None = None
        self._encode_cache_tracks_key = None
        self._encode_cache_value: dict[int, tuple[np.ndarray | None, bool]] | None = None

    def _load_vision_model(self) -> None:
        """Load the FashionCLIP model and processor for image encoding only, without computing text embeddings; a no-op if already loaded."""
        # Idempotent so a caller that only ever needs get_image_embeddings()
        # (EmbeddingClusteringClassifier) can call this unconditionally
        # before every use without re-downloading or re-placing the model
        # on repeat calls. This is the ONE place the vision model is loaded;
        # _load_model() below calls this rather than duplicating it, so the
        # two load paths cannot drift apart.
        if self.model is not None and self.processor is not None:
            return
        self.model = CLIPModel.from_pretrained('patrickjohncyh/fashion-clip').to(self.device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained('patrickjohncyh/fashion-clip')
        self._encode_cache_frame = None
        self._encode_cache_tracks_key = None
        self._encode_cache_value = None

    def _load_model(self) -> None:
        """Lazily load the FashionCLIP model and processor from HuggingFace, and cache the two prompts' text embeddings."""
        self._load_vision_model()
        self.text_embeds = self._compute_text_embeds()

    def _compute_text_embeds(self) -> np.ndarray:
        """Tokenise and embed the two fixed team prompts once; L2-normalised, matching CLIPModel.__call__'s text path."""
        # The two prompts are fixed for a run, so this happens exactly once
        # per _load_model() call rather than being re-tokenised and
        # re-embedded on every classify.
        inputs = self.processor(
            text=[self.team_1_description, self.team_2_description],
            return_tensors='pt',
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with torch.inference_mode():
            text_embeds = self.model.get_text_features(**inputs)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        return text_embeds.detach().cpu().numpy()

    def _crop_player(self, frame: np.ndarray, bbox: list[float]) -> Image.Image | None:
        """Crop the upper jersey region of a player bbox and return it as a PIL image, or None when the crop is empty."""
        h, w = frame.shape[:2]
        raw_x1, raw_y1, raw_x2, raw_y2 = (int(coord) for coord in bbox)

        # crop_fraction is a fraction of the RAW box height, not whatever is
        # left after clamping to the frame: deriving crop_height from the
        # clamped y1/y2 would make a player touching the top/bottom edge
        # sample a different fraction of their actual body than a fully
        # on-frame player, which crop_fraction is not supposed to mean.
        # Compute the intended slice end from the raw box, then clamp both
        # the slice start and end into [0, h] only for the actual slice.
        crop_height = int((raw_y2 - raw_y1) * self.crop_fraction)
        intended_y2 = raw_y1 + crop_height

        # x has no such fraction to preserve, but the same two-directional
        # clamp applies: clamping only x1 up and x2 down leaves a box
        # entirely off one edge with an out-of-range far coordinate, and
        # slicing with that raw value triggers Python's negative-index
        # wraparound, silently returning unrelated content from the far end
        # of the frame instead of an empty crop.
        x1 = min(max(raw_x1, 0), w)
        x2 = min(max(raw_x2, 0), w)
        y1 = min(max(raw_y1, 0), h)
        y2 = min(max(intended_y2, 0), h)
        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)

    def _encode_crops(
        self,
        frame: np.ndarray,
        tracks: dict[int, PlayerTrack],
    ) -> dict[int, tuple[np.ndarray | None, bool]]:
        """One batched forward pass over every player's crop in this frame, memoised against the immediately preceding call for the same frame and track mapping."""
        if self.model is None or self.processor is None:
            raise RuntimeError('FashionCLIP model is not loaded — call _load_model() before classifying jerseys.')

        # The mapping back to player_id is explicit (built from the same id
        # list fed into the batch below), not positional-by-luck.
        tracks_key = tuple(sorted((pid, tuple(t.bbox)) for pid, t in tracks.items()))
        # Memoised so a classify call followed by a get_image_embeddings()
        # call (or vice versa) for the same frame triggers only one forward
        # pass, not two. The memo is keyed on the frame by identity (`is`),
        # against a frame object this method itself keeps alive (see
        # _encode_cache_frame), not by id(frame), which is only safe as a
        # key for as long as the object it names is guaranteed alive.
        # Callers must not mutate a frame array in place between calls that
        # are meant to hit this memo (e.g. classify_jersey() followed by
        # get_image_embeddings() for "the same" frame): an in-place mutation
        # is invisible to an identity check and would silently return stale
        # embeddings for the new pixel content.
        if self._encode_cache_frame is frame and self._encode_cache_tracks_key == tracks_key:
            return self._encode_cache_value

        result: dict[int, tuple[np.ndarray | None, bool]] = {}
        images: list[Image.Image] = []
        valid_ids: list[int] = []
        for player_id, track in tracks.items():
            image = self._crop_player(frame, track.bbox)
            if image is None:
                self.empty_crop_count += 1
                result[player_id] = (None, False)
            else:
                images.append(image)
                valid_ids.append(player_id)

        if images:
            inputs = self.processor(images=images, return_tensors='pt', padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                image_embeds = self.model.get_image_features(**inputs)
            image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
            image_embeds = image_embeds.detach().cpu().numpy()
            for player_id, embed in zip(valid_ids, image_embeds):
                result[player_id] = (embed, True)

        self._encode_cache_frame = frame
        self._encode_cache_tracks_key = tracks_key
        self._encode_cache_value = result
        return result

    def get_image_embeddings(
        self,
        frame: np.ndarray,
        tracks: dict[int, PlayerTrack],
    ) -> dict[int, np.ndarray | None]:
        """Return the L2-normalised FashionCLIP image embedding for every player crop in this frame, keyed by player_id (None for an unusable crop). frame must not be mutated in place before another call meant to reuse this frame's cached encoding; see _encode_crops."""
        # Reuses the same batched forward pass as classification, so
        # EmbeddingClusteringClassifier (below) can call this with zero
        # re-inference when it runs alongside team classification for the
        # same frame.
        encoded = self._encode_crops(frame, tracks)
        return {player_id: (embed if ok else None) for player_id, (embed, ok) in encoded.items()}

    def _classify_batch(
        self,
        frame: np.ndarray,
        tracks: dict[int, PlayerTrack],
    ) -> dict[int, tuple[int, float, bool]]:
        """Classify every player in tracks against the cached text embeddings in one batched forward pass, returning {player_id: (predicted_team, confidence, crop_ok)} where predicted_team is the RAW argmax, never gated on the confidence threshold itself; callers must gate on crop_ok, not on the value."""
        if self.text_embeds is None:
            raise RuntimeError('FashionCLIP model is not loaded — call _load_model() before classifying jerseys.')

        encoded = self._encode_crops(frame, tracks)
        logit_scale = float(self.model.logit_scale.exp().item())

        results: dict[int, tuple[int, float, bool]] = {}
        for player_id, (embed, ok) in encoded.items():
            if not ok:
                # A crop that could not be embedded gets the documented
                # placeholder predicted_team=1, confidence=0.0, crop_ok=False:
                # there is no embedding to argmax over, but a
                # raw-prediction row must never carry team 0.
                results[player_id] = (1, 0.0, False)
                continue

            # CRITICAL: for a valid crop, this must reproduce
            # CLIPModel.__call__'s logits_per_image exactly (image_embeds @
            # text_embeds.T, scaled by logit_scale.exp(), then softmax);
            # see test_team_classifier.py's equivalence test.
            logits = logit_scale * (embed @ self.text_embeds.T)
            probs = _softmax(logits)
            team_idx = int(np.argmax(probs))
            results[player_id] = (team_idx + 1, float(probs[team_idx]), True)
        return results

    def classify_jersey(self, frame: np.ndarray, bbox: list[float]) -> tuple[int, float]:
        """Classify a single player crop and return its team ID and confidence, returning team 0 when unresolved. frame must not be mutated in place before another call meant to reuse the same crop-encoding memo; see get_image_embeddings()."""
        result = self._classify_batch(frame, {0: PlayerTrack(track_id=0, bbox=list(bbox), confidence=0.0)})
        predicted_team, confidence, crop_ok = result[0]

        if not crop_ok:
            return (0, 0.0)
        if confidence < self.confidence_threshold:
            return (0, confidence)
        return (predicted_team, confidence)

    def assign_team(self, frame: np.ndarray, bbox: list[float], player_id: int) -> int:
        """Return the team ID for a player, consulting and populating assignment_cache only when use_assignment_cache is True. frame must not be mutated in place before another call meant to reuse the same crop-encoding memo; see get_image_embeddings()."""
        if self.use_assignment_cache and player_id in self.assignment_cache:
            return self.assignment_cache[player_id]

        team_id, confidence = self.classify_jersey(frame, bbox)
        if self.use_assignment_cache and team_id != 0:
            self.assignment_cache[player_id] = team_id

        return team_id

    def _resolve_team(self, predicted_team: int, confidence: float, crop_ok: bool) -> int:
        """Resolve a (possibly aggregated) prediction to a returned team ID, abstaining (team 0) on an unusable crop or a below-threshold confidence."""
        if not crop_ok:
            return 0
        if confidence < self.confidence_threshold:
            return 0
        return predicted_team

    def _aggregate_window_weighted(
        self,
        raw_frames: list[dict[int, tuple[int, float, bool]]],
    ) -> list[dict[int, tuple[int, float, bool]]]:
        """Replace each player's per-frame team with the confidence-weighted majority over a centred window of that player's own labelled frames, leaving confidence and crop_ok untouched."""
        # Window bounds reproduce pandas' rolling(window, center=True,
        # min_periods=1) exactly: before = window // 2, after =
        # (window - 1) // 2, truncated at a track's ends rather than padded.
        # Confirmed empirically against pandas for odd and even windows;
        # the offline sweep that selected this policy was computed with that
        # expression, so a divergence here would silently invalidate the
        # adopted accuracy figure. Pinned by the equivalence test in
        # tests/test_team_classifier.py.
        before = self.aggregation_window // 2
        after = (self.aggregation_window - 1) // 2

        frame_indices_by_player: dict[int, list[int]] = {}
        for frame_idx, frame_raw in enumerate(raw_frames):
            for player_id in frame_raw:
                frame_indices_by_player.setdefault(player_id, []).append(frame_idx)

        aggregated = [dict(frame_raw) for frame_raw in raw_frames]
        for player_id, frame_indices in frame_indices_by_player.items():
            # Positional over the player's OWN labelled frames, so a gap
            # where the track is absent closes up rather than padding the
            # window, exactly what a per-track pandas Series does, and why
            # this is grouped by player rather than swept over frame_idx.
            track = [raw_frames[frame_idx][player_id] for frame_idx in frame_indices]
            for position, frame_idx in enumerate(frame_indices):
                window = track[max(0, position - before):position + after + 1]
                weight_1 = sum(confidence for team, confidence, _ok in window if team == 1)
                weight_2 = sum(confidence for team, confidence, _ok in window if team == 2)
                # >= ties to team 1, matching np.where(w1 >= w2, 1, 2). The
                # tie-break direction is load-bearing: the offline result
                # was produced with it.
                team = 1 if weight_1 >= weight_2 else 2
                _raw_team, confidence, crop_ok = raw_frames[frame_idx][player_id]
                aggregated[frame_idx][player_id] = (team, confidence, crop_ok)
        return aggregated

    def _cache_fingerprint(self, player_tracks: list[dict[int, PlayerTrack]], n_frames: int) -> dict:
        """Fingerprint of the classification inputs and parameters; the tracks digest catches upstream detector changes."""
        return {
            'player_tracks_digest': data_digest(player_tracks),
            'method': 'fashionclip',
            'n_frames': n_frames,
            'team_1_description': self.team_1_description,
            'team_2_description': self.team_2_description,
            'reset_interval': self.reset_interval,
            'crop_fraction': self.crop_fraction,
            'confidence_threshold': self.confidence_threshold,
            'use_assignment_cache': self.use_assignment_cache,
            'aggregation': self.aggregation,
            'aggregation_window': self.aggregation_window,
            'classifier_revision': self.CLASSIFIER_REVISION,
        }

    def assign_teams(
        self,
        video_frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
        cache_path: str | None = None,
        record_path: str | None = None,
        clip_name: str | None = None,
    ) -> list[dict[int, int]]:
        """Assign team IDs to tracked players across all frames, applying the configured temporal aggregation to the raw per-frame decisions before thresholding."""
        if len(player_tracks) != len(video_frames):
            raise ValueError(
                f'Got {len(player_tracks)} track frames for {len(video_frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )
        if record_path is not None and clip_name is None:
            raise ValueError('clip_name is required when record_path is set — the raw artefact records it per row.')

        fingerprint = None
        cached = None
        if cache_path is not None and record_path is None:
            # A cache written while recording would carry the recording run's
            # own risk of drift as a silent, uninspectable fact: never write
            # (or read) cache_path at all when record_path is set, even though
            # (a) below makes the two runs' returned assignments identical.
            fingerprint = self._cache_fingerprint(player_tracks, len(video_frames))
            cached = load_valid_cache(cache_path, fingerprint)
        if cached is not None:
            return cached

        self._load_model()
        self.empty_crop_count = 0

        assignment: list[dict[int, int]] = []
        # Raw (predicted_team, confidence, crop_ok) per player per frame,
        # kept for the aggregation pass below. Complete whenever aggregation
        # needs it: use_assignment_cache (the only thing that skips
        # classifying a player) cannot be True alongside an aggregation
        # policy; __init__ rejects that combination.
        raw_frames: list[dict[int, tuple[int, float, bool]]] = []
        record_rows: list[tuple] | None = [] if record_path is not None else None

        for frame_idx, (frame, tracks) in enumerate(zip(video_frames, player_tracks)):
            if self.use_assignment_cache and frame_idx % self.reset_interval == 0:
                self.assignment_cache = {}

            frame_assignment: dict[int, int] = {}
            frame_raw: dict[int, tuple[int, float, bool]] = {}
            if record_rows is not None:
                # Every player is classified fresh every frame: the raw
                # artefact must be a genuine forward-pass result for every
                # (frame, player) pair, not a value inherited from the
                # assignment_cache up to reset_interval frames earlier; that
                # would defeat the artefact's purpose of offline threshold
                # sweeps with zero re-inference. But the RETURNED assignment
                # must still honour assignment_cache exactly like the
                # non-recording branch below (when use_assignment_cache is
                # True): a player already confidently resolved this reset
                # window keeps that sticky value even if this frame's fresh
                # confidence happens to dip: recording is a side channel on
                # top of the raw scores, not a second, diverging assignment
                # policy.
                raw = self._classify_batch(frame, tracks)
                for player_id, (predicted_team, confidence, crop_ok) in raw.items():
                    # RAW, pre-aggregation, pre-threshold: the artefact is
                    # the input every offline sweep (including the one that
                    # selected the aggregation policy below) reads, so it
                    # must never carry an aggregated or thresholded value.
                    record_rows.append((clip_name, frame_idx, player_id, predicted_team, confidence, crop_ok))
                    frame_raw[player_id] = (predicted_team, confidence, crop_ok)
                    if self.use_assignment_cache and player_id in self.assignment_cache:
                        frame_assignment[player_id] = self.assignment_cache[player_id]
                        continue
                    team_id = self._resolve_team(predicted_team, confidence, crop_ok)
                    frame_assignment[player_id] = team_id
                    if self.use_assignment_cache and team_id != 0:
                        self.assignment_cache[player_id] = team_id
            else:
                uncached = (
                    {pid: t for pid, t in tracks.items() if pid not in self.assignment_cache}
                    if self.use_assignment_cache else tracks
                )
                raw = self._classify_batch(frame, uncached) if uncached else {}
                for player_id, track in tracks.items():
                    if self.use_assignment_cache and player_id in self.assignment_cache:
                        frame_assignment[player_id] = self.assignment_cache[player_id]
                        continue
                    predicted_team, confidence, crop_ok = raw[player_id]
                    frame_raw[player_id] = (predicted_team, confidence, crop_ok)
                    team_id = self._resolve_team(predicted_team, confidence, crop_ok)
                    frame_assignment[player_id] = team_id
                    if self.use_assignment_cache and team_id != 0:
                        self.assignment_cache[player_id] = team_id

            assignment.append(frame_assignment)
            raw_frames.append(frame_raw)

        if self.aggregation == 'window_weighted':
            # Supersedes the per-frame decisions built above (which cost
            # only dict writes; no inference is repeated). Aggregation runs
            # here, after every raw decision exists, because a centred
            # window needs the frames on BOTH sides of each frame.
            aggregated_frames = self._aggregate_window_weighted(raw_frames)
            assignment = [
                {
                    player_id: self._resolve_team(team, confidence, crop_ok)
                    for player_id, (team, confidence, crop_ok) in aggregated_frame.items()
                }
                for aggregated_frame in aggregated_frames
            ]

        # Derived from the returned assignment itself, not from
        # empty_crop_count: empty_crop_count is mutated inside the memoised
        # _encode_crops() every time it actually sees an unusable crop, but
        # "every time it sees" differs by mode: the non-recording branch
        # above only ever passes uncached players in, so a player already
        # resolved via assignment_cache is never counted again, while the
        # recording branch passes the full tracks dict every frame, so the
        # same resolved player's empty crop (if any) gets counted on every
        # single frame even though their assignment is served from the
        # cache and is not team 0. Counting (frame, player) pairs where the
        # returned team is 0 means the same thing in both modes and matches
        # what the message claims.
        unresolved = sum(team_id == 0 for frame_teams in assignment for team_id in frame_teams.values())
        if unresolved > 0:
            print(
                f'[FashionCLIPClassifier] {unresolved} (frame, player) assignments left unresolved (team 0).'
            )
        if self.empty_crop_count > 0:
            # Raw crop-encoding diagnostic only, not an "unresolved" count:
            # it can overcount relative to `unresolved` above while
            # recording, since every player is re-encoded every frame
            # regardless of assignment_cache membership.
            print(
                f'[FashionCLIPClassifier] {self.empty_crop_count} raw crop-encoding attempts were empty.'
            )

        if record_path is not None:
            _write_prediction_csv(record_rows, record_path)

        if cache_path is not None and record_path is None:
            save_cache_with_meta(assignment, cache_path, fingerprint)
        return assignment


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over a 1-D array."""
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


class KMeansClassifier(TeamClassifier):
    """Baseline team assignment via K-means clustering on background-removed jersey colours."""

    # Same increment rule as FashionCLIPClassifier.CLASSIFIER_REVISION: bumped
    # whenever anything the parameter keys cannot see changes. 1: bbox
    # clipping and raw-prediction recording. 2: whole-clip fitting scope,
    # border-ring background estimation, and abstention on
    # too-few-foreground crops and on low margin, which change assignments
    # for the same parameters and must therefore invalidate every cache
    # written before them.
    CLASSIFIER_REVISION = 2

    def __init__(
        self,
        crop_fraction: float,
        fit_stride: int,
        bg_distance_threshold: int,
        margin_threshold: float
    ) -> None:
        self.crop_fraction = crop_fraction
        self.fit_stride = fit_stride
        self.bg_distance_threshold = bg_distance_threshold
        self.margin_threshold = margin_threshold
        self.team_centres: np.ndarray | None = None
        self.team_labels: dict[int, int] = {}
        self.empty_crop_count = 0
        self.sparse_crop_count = 0
        self.low_margin_count = 0

    def _crop_player(self, frame: np.ndarray, bbox: list[float]) -> np.ndarray:
        """Crop the upper jersey region of a player bbox and return it as a BGR array (zero-size when the box is invalid)."""
        h, w = frame.shape[:2]
        raw_x1, raw_y1, raw_x2, raw_y2 = (int(coord) for coord in bbox)

        # crop_height must come from the RAW box height, not the
        # frame-clamped one; see FashionCLIPClassifier._crop_player's note:
        # otherwise a player touching the top/bottom edge samples a
        # different fraction of their actual body than a fully on-frame
        # player. Compute the intended slice end from the raw box, then
        # clamp both the slice start and end into [0, h] only for slicing.
        crop_height = int((raw_y2 - raw_y1) * self.crop_fraction)
        intended_y2 = raw_y1 + crop_height

        # x has no such fraction to preserve, but the same two-directional
        # clamp applies; see FashionCLIPClassifier._crop_player's note: a
        # one-directional clamp leaves a box entirely off one edge with an
        # out-of-range far coordinate, and slicing with that raw value
        # wraps around instead of yielding an empty crop.
        x1 = min(max(raw_x1, 0), w)
        x2 = min(max(raw_x2, 0), w)
        y1 = min(max(raw_y1, 0), h)
        y2 = min(max(intended_y2, 0), h)
        if x2 <= x1 or y2 <= y1:
            return np.empty((0, 0, 3), dtype=frame.dtype)

        return frame[y1:y2, x1:x2]

    def _border_ring_colour(self, crop: np.ndarray) -> np.ndarray:
        """Estimate the crop's background colour as the mean of its outer border ring."""
        # A four-corner-pixel estimate is four samples: at
        # crop_fraction 0.667 the corners of a jersey crop frequently land on
        # another player or the crowd, and one such corner moves the estimate
        # by a quarter of its own value. The ring averages hundreds of
        # perimeter pixels instead, so the same intrusion is diluted rather
        # than dominant. BORDER_RING_FRACTION is scaled off the SHORTER side
        # so a tall thin crop still gets a ring rather than two overlapping
        # slabs.
        h, w = crop.shape[:2]
        thickness = max(1, int(round(min(h, w) * BORDER_RING_FRACTION)))

        if 2 * thickness >= h or 2 * thickness >= w:
            # Crop too small for a genuine ring: every pixel is border.
            return crop.reshape(-1, 3).astype(np.float32).mean(axis=0)

        ring = np.concatenate([
            crop[:thickness, :].reshape(-1, 3),
            crop[-thickness:, :].reshape(-1, 3),
            crop[thickness:-thickness, :thickness].reshape(-1, 3),
            crop[thickness:-thickness, -thickness:].reshape(-1, 3),
        ]).astype(np.float32)
        return ring.mean(axis=0)

    def _remove_background(self, crop: np.ndarray) -> np.ndarray:
        """Return the crop's foreground pixels (those far enough from the border-ring background colour), or an empty array when too few survive."""
        bg_colour = self._border_ring_colour(crop)

        pixels = crop.reshape(-1, 3).astype(np.float32)
        distances = np.linalg.norm(pixels - bg_colour, axis=1)
        foreground = pixels[distances > self.bg_distance_threshold]

        if len(foreground) < MIN_FOREGROUND_PIXELS:
            # Abstain rather than fall back to crop.reshape(-1, 3), which
            # would return every pixel INCLUDING the background it had just
            # failed to separate, a fabricated colour dressed up as a
            # measured one, contradicting the abstention contract that an
            # unusable crop yields no answer. Callers read the empty array
            # as "no foreground", count it, and resolve to team 0.
            return np.empty((0, 3), dtype=np.float32)

        return foreground

    def _representative_colour(self, pixels: np.ndarray) -> np.ndarray | None:
        """Return the mean foreground colour of a pixel array, or None when too few pixels remain to be representative."""
        if len(pixels) < 4:
            return None
        return pixels.mean(axis=0)

    def _fit_team_centres(
        self,
        video_frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]]
    ) -> None:
        """Fit the two team colour centres from player crops sampled every fit_stride-th frame across the WHOLE clip."""
        # Fitting spans the whole clip, never only its first few frames.
        # Frames 0-1 routinely carry no confirmed ByteTrack tracks yet, so a
        # first-frames fit would draw a whole clip's reference colours from
        # roughly three frames of whoever happened to be on screen at second
        # zero, handicapping the baseline by sampling rather than by method.
        # Spanning the clip represents both teams' jerseys under the lighting
        # and pose variation they actually appear in. Measured cost at the
        # worst case in this dataset
        # (clip_3, 243 frames, 12 boxes/frame, 1280x720): 3.3 s for a
        # stride-1 fit over every frame, against minutes for the detection
        # stage feeding it, so stride 1 (all frames) is the default and
        # fit_stride exists only as an escape hatch for longer footage.
        representative_colours: list[np.ndarray] = []
        for frame, tracks in zip(video_frames[::self.fit_stride], player_tracks[::self.fit_stride]):
            for track in tracks.values():
                crop = self._crop_player(frame, track.bbox)
                if crop.size == 0:
                    continue
                pixels = self._remove_background(crop)
                colour = self._representative_colour(pixels)

                # No stand-in colour is fitted for a crop with too few
                # pixels: a neutral-grey placeholder would bias both centres.
                if colour is None:
                    self.sparse_crop_count += 1
                    continue

                representative_colours.append(colour)

        if self.sparse_crop_count > 0:
            print(
                f'[KMeansClassifier] {self.sparse_crop_count} player crops had too few '
                f'foreground pixels to fit and were skipped.'
            )

        if len(representative_colours) < 2:
            raise ValueError('Insufficient player crops to fit team colour centres. Lower kmeans_fit_stride in config.')

        colours = np.array(representative_colours)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(colours)
        centres = kmeans.cluster_centers_

        luminances = centres.sum(axis=1)
        sorted_indices = np.argsort(luminances)[::-1]
        self.team_centres = centres
        self.team_labels = {int(sorted_indices[0]): 1, int(sorted_indices[1]): 2}

    def _classify_with_margin(self, frame: np.ndarray, bbox: list[float]) -> tuple[int, float, bool]:
        """Classify one crop by nearest colour centre and return (predicted_team, confidence_margin, crop_ok); predicted_team is the RAW assignment, never 0; callers must gate on crop_ok, not on the value."""
        if self.team_centres is None:
            raise RuntimeError('Team colour centres are not fitted — call _fit_team_centres() before classifying jerseys.')

        crop = self._crop_player(frame, bbox)
        if crop.size == 0:
            # Documented placeholder for an unusable crop: there is no
            # colour to assign a centre to, but a raw-prediction row must
            # never carry team 0; classify_jersey() resolves crop_ok=False
            # to team 0 for callers that don't need the raw value.
            self.empty_crop_count += 1
            return (1, 0.0, False)

        pixels = self._remove_background(crop)
        colour = self._representative_colour(pixels)
        if colour is None:
            self.sparse_crop_count += 1
            return (1, 0.0, False)

        distances = np.linalg.norm(self.team_centres - colour, axis=1)
        cluster_idx = int(distances.argmin())
        other_idx = 1 - cluster_idx
        predicted_team = self.team_labels[cluster_idx]
        # confidence_margin is a K-means-appropriate analogue of FashionCLIP's
        # softmax confidence: the nearest centre's share of the two centres'
        # summed distance: in [0.5, 1] when a genuine nearest centre exists,
        # ~0.5 when the two centres are equidistant (a genuine tie).
        margin = float(distances[other_idx] / (distances[cluster_idx] + distances[other_idx] + 1e-9))
        return (predicted_team, margin, True)

    def _resolve_team(self, predicted_team: int, confidence: float, crop_ok: bool) -> int:
        """Resolve a raw prediction to a returned team ID, abstaining (team 0) on an unusable crop or a below-threshold margin."""
        # Abstention on a weak decision, not only on an unusable crop: a
        # jersey sitting almost exactly between the two colour centres must
        # not be forced to a side. Without this, FashionCLIP can abstain on
        # low confidence while K-means cannot, an asymmetry that would read
        # as a finding about K-means when it is a finding about the
        # implementation.
        if not crop_ok:
            return 0
        if confidence < self.margin_threshold:
            self.low_margin_count += 1
            return 0
        return predicted_team

    def classify_jersey(self, frame: np.ndarray, bbox: list[float]) -> int:
        """Classify a single player crop and return its team ID by nearest colour centre, or 0 when the crop is unusable or the margin is below threshold."""
        predicted_team, confidence, crop_ok = self._classify_with_margin(frame, bbox)
        return self._resolve_team(predicted_team, confidence, crop_ok)

    def _cache_fingerprint(self, player_tracks: list[dict[int, PlayerTrack]], n_frames: int) -> dict:
        """Fingerprint of the classification inputs and parameters; the tracks digest catches upstream detector changes."""
        return {
            'player_tracks_digest': data_digest(player_tracks),
            'method': 'kmeans',
            'n_frames': n_frames,
            'crop_fraction': self.crop_fraction,
            'fit_stride': self.fit_stride,
            'bg_distance_threshold': self.bg_distance_threshold,
            'margin_threshold': self.margin_threshold,
            'classifier_revision': self.CLASSIFIER_REVISION,
        }

    def assign_teams(
        self,
        video_frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
        cache_path: str | None = None,
        record_path: str | None = None,
        clip_name: str | None = None,
    ) -> list[dict[int, int]]:
        """Assign team IDs to all tracked players across all frames after fitting colour centres."""
        if len(player_tracks) != len(video_frames):
            raise ValueError(
                f'Got {len(player_tracks)} track frames for {len(video_frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )
        if record_path is not None and clip_name is None:
            raise ValueError('clip_name is required when record_path is set — the raw artefact records it per row.')

        fingerprint = None
        cached = None
        if cache_path is not None and record_path is None:
            # Never read or write cache_path while recording; see
            # FashionCLIPClassifier.assign_teams's note; the same invariant
            # applies here even though K-means' returned assignment happens
            # not to diverge between the two modes.
            fingerprint = self._cache_fingerprint(player_tracks, len(video_frames))
            cached = load_valid_cache(cache_path, fingerprint)
        if cached is not None:
            return cached

        self.empty_crop_count = 0
        self.sparse_crop_count = 0
        self.low_margin_count = 0

        self._fit_team_centres(video_frames, player_tracks)

        assignment: list[dict[int, int]] = []
        record_rows: list[tuple] | None = [] if record_path is not None else None

        for frame_idx, (frame, tracks) in enumerate(zip(video_frames, player_tracks)):
            frame_assignment: dict[int, int] = {}
            for player_id, track in tracks.items():
                predicted_team, confidence, crop_ok = self._classify_with_margin(frame, track.bbox)
                frame_assignment[player_id] = self._resolve_team(predicted_team, confidence, crop_ok)
                if record_rows is not None:
                    # The recorded row stays PRE-threshold (the raw team and
                    # its margin) so the margin threshold can be swept
                    # offline without re-running the fit.
                    record_rows.append((clip_name, frame_idx, player_id, predicted_team, confidence, crop_ok))
            assignment.append(frame_assignment)

        # Derived from the returned assignment itself, not from
        # empty_crop_count/sparse_crop_count; see FashionCLIPClassifier.
        # assign_teams's note: counting (frame, player) pairs where the
        # returned team is 0 means the same thing regardless of how the
        # per-crop counters were accumulated, and matches what the message
        # claims.
        unresolved = sum(team_id == 0 for frame_teams in assignment for team_id in frame_teams.values())
        if unresolved > 0:
            print(f'[KMeansClassifier] {unresolved} (frame, player) assignments left unresolved (team 0).')
        if self.empty_crop_count > 0 or self.sparse_crop_count > 0 or self.low_margin_count > 0:
            # Raw per-crop diagnostic only, not an "unresolved" count. The
            # three reasons are reported separately so an abstention caused
            # by a weak decision is never mistaken for an unusable crop.
            print(
                f'[KMeansClassifier] {self.empty_crop_count} raw crops were empty, '
                f'{self.sparse_crop_count} had too few foreground pixels, '
                f'{self.low_margin_count} fell below the margin threshold.'
            )

        if record_path is not None:
            _write_prediction_csv(record_rows, record_path)

        if cache_path is not None and record_path is None:
            save_cache_with_meta(assignment, cache_path, fingerprint)
        return assignment


class EmbeddingClusteringClassifier(TeamClassifier):
    """Prompt-free team assignment by K-means clustering of FashionCLIP image embeddings, fitted per clip."""

    # Same increment rule as the other two classifiers' CLASSIFIER_REVISION.
    # Starts at 1 with this classifier's introduction.
    CLASSIFIER_REVISION = 1

    def __init__(
        self,
        encoder: FashionCLIPClassifier,
        fit_stride: int,
        margin_threshold: float
    ) -> None:
        # The encoder is a FashionCLIPClassifier used ONLY through its
        # get_image_embeddings() accessor. No text description is ever read
        # here, for any matchup; that is the entire point of this arm: it
        # isolates whether FashionCLIP's advantage over pixel colour comes
        # from its semantic representation or from text conditioning. The
        # encoder's own prompted classification path is untouched.
        self.encoder = encoder
        self.fit_stride = fit_stride
        self.margin_threshold = margin_threshold
        self.cluster_centres: np.ndarray | None = None
        self.team_labels: dict[int, int] = {}
        self.empty_crop_count = 0
        self.low_margin_count = 0
        # Populated by _collect_embeddings(): {frame_idx (into the ORIGINAL
        # video_frames list): {player_id: embedding-or-None}}. See
        # _collect_embeddings' docstring for why this exists.
        self._embeddings_by_frame: dict[int, dict[int, np.ndarray | None]] = {}

    def _collect_embeddings(
        self,
        video_frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
    ) -> np.ndarray:
        """Collect the L2-normalised image embedding of every usable player crop across the sampled frames, as an (n_crops, dim) array."""
        # The encoder's own get_image_embeddings() memo is a SINGLE slot
        # keyed by frame identity: it only prevents a second forward pass
        # for the immediately preceding call, not across an entire clip. If
        # this loop's result were discarded, the assignment loop below would
        # have to call get_image_embeddings() again for every frame, and by
        # then the encoder's memo has long since been overwritten by every
        # OTHER frame in between: a full second forward pass over the whole
        # clip, doubling the GPU cost the design intended. Every frame's
        # result is cached here instead, by its ORIGINAL index into
        # video_frames, so the assignment loop can reuse it directly. At the
        # largest clip in this dataset (~243 frames * ~12 players * 512
        # float32 components) that is 243 * 12 * 512 * 4 bytes =~ 6 MB,
        # negligible next to the model itself, and no worse a design than a
        # true single pass would require: the cluster centres depend on
        # every sampled frame's embeddings, so no frame can be assigned a
        # team until fitting has already visited all of them, which rules
        # out folding fitting and assignment into one physical loop.
        self._embeddings_by_frame = {}
        embeddings: list[np.ndarray] = []
        for frame_idx in range(0, len(video_frames), self.fit_stride):
            frame_embeddings = self.encoder.get_image_embeddings(video_frames[frame_idx], player_tracks[frame_idx])
            self._embeddings_by_frame[frame_idx] = frame_embeddings
            for embedding in frame_embeddings.values():
                if embedding is None:
                    continue
                embeddings.append(embedding)
        if not embeddings:
            return np.empty((0, 0), dtype=np.float32)
        return np.stack(embeddings)

    def _fit_clusters(
        self,
        video_frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
    ) -> None:
        """Fit the two embedding cluster centres over the whole clip's player crops."""
        # DISCLOSED LIMITATION: fitting is per clip over all that clip's
        # sampled frames, not per frame. Roughly eight players are visible
        # in any single frame, which is far too few to fit two stable
        # clusters, so this method is offline-batch: it needs the whole
        # clip in hand before it can label its first frame, exactly like the
        # adopted ball gate (filter_detections_global_trajectory). It is
        # therefore NOT a streaming/online classifier and must not be
        # reported as one.
        embeddings = self._collect_embeddings(video_frames, player_tracks)
        if len(embeddings) < 2:
            raise ValueError(
                'Insufficient usable player crops to fit embedding clusters. '
                'Lower embedding_fit_stride in config.'
            )

        # K-means, NOT agglomerative, deliberately: the colour baseline uses
        # K-means, so keeping the algorithm identical makes the
        # representation (raw pixels vs FashionCLIP embedding) the ONLY
        # difference between the two arms. On L2-normalised vectors squared
        # Euclidean distance is 2 - 2*cosine similarity, i.e. monotonically
        # related to cosine distance, so Euclidean K-means is already the
        # cosine-appropriate choice here and needs no special metric.
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=42)
        kmeans.fit(embeddings)
        self.cluster_centres = kmeans.cluster_centers_

        # Cluster index -> team ID is arbitrary in an unsupervised method:
        # nothing in the embedding says which cluster is "team 1", and no
        # reference exists to align a prompt-free method to. Ordered from
        # the two centroid vectors themselves (lexicographic comparison),
        # NOT from which player happened to be first in tracks.values() on
        # the first sampled frame; that depended on dict/track iteration
        # order, which is fully deterministic for one fixed run but flips
        # silently on any upstream change to insertion order despite
        # nothing else about the fit having changed. The lexicographic
        # choice is itself arbitrary, but it is STABLE and reproducible for
        # a given fitted pair of centroids regardless of iteration order,
        # which is all that is required: evaluation scores this arm up to a
        # team-label permutation, as for any clustering method.
        centre_a, centre_b = self.cluster_centres
        if tuple(centre_a) <= tuple(centre_b):
            self.team_labels = {0: 1, 1: 2}
        else:
            self.team_labels = {0: 2, 1: 1}

    def _classify_with_margin(self, embedding: np.ndarray | None) -> tuple[int, float, bool]:
        """Assign one embedding to its nearest cluster centre and return (predicted_team, confidence_margin, crop_ok); predicted_team is the RAW assignment, never 0; callers must gate on crop_ok, not on the value."""
        if self.cluster_centres is None:
            raise RuntimeError('Embedding clusters are not fitted — call _fit_clusters() before assigning teams.')

        if embedding is None:
            # Documented placeholder for an unusable crop, matching both
            # other classifiers: a raw-prediction row must never carry
            # team 0, so crop_ok=False is the only signal of abstention.
            self.empty_crop_count += 1
            return (1, 0.0, False)

        distances = np.linalg.norm(self.cluster_centres - embedding, axis=1)
        cluster_idx = int(distances.argmin())
        other_idx = 1 - cluster_idx
        # Same margin form as KMeansClassifier (d_other / (d_near + d_other)),
        # so all three methods emit a score on a comparable RANGE. Those
        # scores are NOT calibrated probabilities and must never be compared
        # ACROSS methods: 0.7 here does not mean what 0.7 means for
        # FashionCLIP's softmax or for the colour baseline. Thresholds are
        # swept per method, never shared between them.
        margin = float(distances[other_idx] / (distances[cluster_idx] + distances[other_idx] + 1e-9))
        return (self.team_labels[cluster_idx], margin, True)

    def _resolve_team(self, predicted_team: int, confidence: float, crop_ok: bool) -> int:
        """Resolve a raw prediction to a returned team ID, abstaining (team 0) on an unusable crop or a below-threshold margin."""
        if not crop_ok:
            return 0
        if confidence < self.margin_threshold:
            self.low_margin_count += 1
            return 0
        return predicted_team

    def _cache_fingerprint(self, player_tracks: list[dict[int, PlayerTrack]], n_frames: int) -> dict:
        """Fingerprint of the classification inputs and parameters; the tracks digest catches upstream detector changes."""
        return {
            'player_tracks_digest': data_digest(player_tracks),
            'method': 'embedding',
            'n_frames': n_frames,
            # The encoder's crop geometry and revision belong in this
            # fingerprint: both change the embeddings this method clusters,
            # even though no prompt does.
            'crop_fraction': self.encoder.crop_fraction,
            'encoder_revision': self.encoder.CLASSIFIER_REVISION,
            'fit_stride': self.fit_stride,
            'margin_threshold': self.margin_threshold,
            'classifier_revision': self.CLASSIFIER_REVISION,
        }

    def assign_teams(
        self,
        video_frames: list[np.ndarray],
        player_tracks: list[dict[int, PlayerTrack]],
        cache_path: str | None = None,
        record_path: str | None = None,
        clip_name: str | None = None,
    ) -> list[dict[int, int]]:
        """Assign team IDs to all tracked players across all frames by nearest embedding cluster centre, after fitting those clusters over the whole clip."""
        if len(player_tracks) != len(video_frames):
            raise ValueError(
                f'Got {len(player_tracks)} track frames for {len(video_frames)} video frames — '
                f'the two must be aligned frame-for-frame.'
            )
        if record_path is not None and clip_name is None:
            raise ValueError('clip_name is required when record_path is set — the raw artefact records it per row.')

        fingerprint = None
        cached = None
        if cache_path is not None and record_path is None:
            # Never read or write cache_path while recording; same
            # invariant as the other two classifiers.
            fingerprint = self._cache_fingerprint(player_tracks, len(video_frames))
            cached = load_valid_cache(cache_path, fingerprint)
        if cached is not None:
            return cached

        # Load the encoder's vision model before touching
        # get_image_embeddings() below: the encoder is constructed but
        # never itself loaded by main.py, and get_image_embeddings() raises
        # RuntimeError against an unloaded model rather than loading it
        # implicitly. Vision-only and idempotent: never computes the text
        # embeddings the prompt-free contract forbids, and safe to call even
        # if the same encoder is already loaded (e.g. shared with a
        # FashionCLIPClassifier run earlier in the same session).
        self.encoder._load_vision_model()

        self.empty_crop_count = 0
        self.low_margin_count = 0

        self._fit_clusters(video_frames, player_tracks)

        assignment: list[dict[int, int]] = []
        record_rows: list[tuple] | None = [] if record_path is not None else None

        for frame_idx, (frame, tracks) in enumerate(zip(video_frames, player_tracks)):
            # Reuses _fit_clusters' cache for every frame it already
            # embedded (all of them, at the default fit_stride=1); only a
            # frame fit_stride skipped needs a fresh pass here. See
            # _collect_embeddings' docstring for why the encoder's own
            # single-slot memo cannot provide this by itself.
            if frame_idx in self._embeddings_by_frame:
                frame_embeddings = self._embeddings_by_frame[frame_idx]
            else:
                frame_embeddings = self.encoder.get_image_embeddings(frame, tracks)
            frame_assignment: dict[int, int] = {}
            for player_id in tracks:
                predicted_team, confidence, crop_ok = self._classify_with_margin(frame_embeddings[player_id])
                frame_assignment[player_id] = self._resolve_team(predicted_team, confidence, crop_ok)
                if record_rows is not None:
                    record_rows.append((clip_name, frame_idx, player_id, predicted_team, confidence, crop_ok))
            assignment.append(frame_assignment)

        # Derived from the returned assignment, not the per-crop counters;
        # see FashionCLIPClassifier.assign_teams's note.
        unresolved = sum(team_id == 0 for frame_teams in assignment for team_id in frame_teams.values())
        if unresolved > 0:
            print(f'[EmbeddingClusteringClassifier] {unresolved} (frame, player) assignments left unresolved (team 0).')
        if self.empty_crop_count > 0 or self.low_margin_count > 0:
            print(
                f'[EmbeddingClusteringClassifier] {self.empty_crop_count} crops could not be embedded, '
                f'{self.low_margin_count} fell below the margin threshold.'
            )

        if record_path is not None:
            _write_prediction_csv(record_rows, record_path)

        if cache_path is not None and record_path is None:
            save_cache_with_meta(assignment, cache_path, fingerprint)
        return assignment
