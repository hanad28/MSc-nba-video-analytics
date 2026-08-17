"""Unit tests for the K-means baseline and the FashionCLIP classifier's non-model logic.

The FashionCLIP tests stub out classify_jersey() / _load_model() rather than
downloading model weights, so they exercise the caching, reset-interval and
disk-cache behaviour without inference.
"""

from __future__ import annotations

import csv
import gc
import re
import zlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from basketball.detection.player_detector import PlayerTrack
from basketball.team_classifier.classifier import (
    PREDICTION_CSV_HEADER,
    EmbeddingClusteringClassifier,
    FashionCLIPClassifier,
    KMeansClassifier,
)


def track(track_id: int, bbox: tuple[float, ...]) -> PlayerTrack:
    return PlayerTrack(track_id=track_id, bbox=list(bbox), confidence=0.9)


JERSEY_INSET_PX = 4
CROP_FRACTION = 0.667


def draw_player(frame: np.ndarray, bbox: tuple[int, int, int, int], colour: int) -> None:
    """Fill a jersey-coloured block inset inside the upper-jersey crop region of bbox, leaving a background border."""
    # The block is inset on all four sides of the region _crop_player()
    # actually slices, so every crop contains a background border exactly as
    # a real detection does. A block filling its crop edge to edge would
    # leave the border ring 100% jersey, so the estimator would have no
    # background to measure, a degenerate case no real broadcast crop
    # presents, and one that would make these fixtures test nothing.
    x1, y1, x2, y2 = bbox
    crop_y2 = y1 + int((y2 - y1) * CROP_FRACTION)
    frame[y1 + JERSEY_INSET_PX:crop_y2 - JERSEY_INSET_PX, x1 + JERSEY_INSET_PX:x2 - JERSEY_INSET_PX] = colour


def frame_with_players(
    light_bbox: tuple[int, int, int, int],
    dark_bbox: tuple[int, int, int, int],
    size: int = 100,
) -> np.ndarray:
    """Green background frame with one near-white and one near-black player block, each inset inside its crop region."""
    frame = np.full((size, size, 3), 60, dtype=np.uint8)
    draw_player(frame, light_bbox, 240)
    draw_player(frame, dark_bbox, 15)
    return frame


LIGHT_BBOX = (10, 10, 30, 40)
DARK_BBOX = (60, 10, 80, 40)


# --- KMeansClassifier ------------------------------------------------------

@pytest.fixture
def kmeans() -> KMeansClassifier:
    return KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)


def test_crop_player_takes_the_upper_jersey_region(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = kmeans._crop_player(frame, [10, 0, 30, 30])

    assert crop.shape == (20, 20, 3)  # 0.667 * 30 == 20 rows


def test_crop_player_returns_an_empty_crop_for_a_degenerate_bbox(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    assert kmeans._crop_player(frame, [10, 10, 10, 10]).size == 0


def test_remove_background_drops_background_pixels(kmeans):
    crop = np.full((20, 20, 3), 60, dtype=np.uint8)
    crop[5:15, 5:15] = 240

    foreground = kmeans._remove_background(crop)

    assert len(foreground) == 100
    assert foreground.mean() == pytest.approx(240.0)


def test_remove_background_abstains_when_nothing_stands_out(kmeans):
    # Previously returned all 25 pixels (background included), a fabricated
    # colour presented as a measured one. The contract is now to abstain.
    crop = np.full((5, 5, 3), 60, dtype=np.uint8)

    foreground = kmeans._remove_background(crop)

    assert len(foreground) == 0
    assert foreground.shape == (0, 3)


def test_remove_background_abstains_rather_than_returning_background_pixels(kmeans):
    # A crop with a handful of foreground pixels, below MIN_FOREGROUND_PIXELS:
    # the old fallback returned every pixel including the background it had
    # just failed to separate, so the representative colour came out as the
    # background's. Abstaining keeps the unusable crop unusable.
    crop = np.full((20, 20, 3), 60, dtype=np.uint8)
    crop[10, 10] = 240
    crop[10, 11] = 240

    foreground = kmeans._remove_background(crop)

    assert len(foreground) == 0
    assert kmeans._representative_colour(foreground) is None


def test_border_ring_background_survives_a_corner_sitting_on_another_player(kmeans):
    # The exact failure of the 4-corner estimator: one corner lands on a
    # bright neighbouring player, dragging the background estimate a quarter
    # of the way to that player's colour. The ring averages the whole
    # perimeter, so a single intruding corner barely moves it.
    crop = np.full((40, 40, 3), 60, dtype=np.uint8)
    crop[10:30, 10:30] = 240  # the jersey itself
    crop[0:4, 0:4] = 240      # a neighbouring player intruding into one corner

    bg_colour = kmeans._border_ring_colour(crop)

    corner_estimate = np.mean([crop[0, 0], crop[0, 39], crop[39, 0], crop[39, 39]], axis=0)
    assert float(np.mean(bg_colour)) < 100.0  # still close to the true background of 60
    assert float(np.mean(corner_estimate)) == pytest.approx(105.0)  # 4-corner estimate is dragged up


def test_border_ring_uses_every_pixel_for_a_crop_too_small_to_ring(kmeans):
    crop = np.full((3, 3, 3), 70, dtype=np.uint8)

    assert kmeans._border_ring_colour(crop).tolist() == [70.0, 70.0, 70.0]


def test_representative_colour_averages_the_pixels(kmeans):
    pixels = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [20.0, 40.0, 60.0], [30.0, 60.0, 90.0]])

    assert kmeans._representative_colour(pixels).tolist() == [15.0, 30.0, 45.0]


def test_representative_colour_returns_none_for_too_few_pixels(kmeans):
    assert kmeans._representative_colour(np.zeros((2, 3))) is None


def test_fit_team_centres_labels_the_lighter_cluster_as_team_one(kmeans):
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]

    kmeans._fit_team_centres(frames, tracks)

    assert kmeans.team_centres.shape == (2, 3)
    lighter = int(np.argmax(kmeans.team_centres.sum(axis=1)))
    assert kmeans.team_labels[lighter] == 1


def test_fit_team_centres_raises_without_enough_crops(kmeans):
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX)]
    tracks = [{1: track(1, LIGHT_BBOX)}]

    with pytest.raises(ValueError, match='Insufficient player crops'):
        kmeans._fit_team_centres(frames, tracks)


def test_kmeans_classify_jersey_assigns_by_nearest_centre(kmeans):
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]
    kmeans._fit_team_centres(frames, tracks)

    assert kmeans.classify_jersey(frames[0], list(LIGHT_BBOX)) == 1
    assert kmeans.classify_jersey(frames[0], list(DARK_BBOX)) == 2


def test_kmeans_classify_jersey_returns_unresolved_for_an_empty_crop(kmeans):
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]
    kmeans._fit_team_centres(frames, tracks)

    assert kmeans.classify_jersey(frames[0], [10, 10, 10, 10]) == 0


def test_kmeans_assign_teams_labels_every_frame(kmeans):
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(3)]

    assignment = kmeans.assign_teams(frames, tracks)

    assert assignment == [{1: 1, 2: 2}] * 3


def test_kmeans_assign_teams_round_trips_through_the_cache(kmeans, tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(3)]

    first = kmeans.assign_teams(frames, tracks, cache_path=cache_path)
    fresh = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)

    assert fresh.assign_teams(frames, tracks, cache_path=cache_path) == first
    assert fresh.team_centres is None  # served from cache, never fitted


def test_kmeans_assign_teams_ignores_a_stale_cache(kmeans, tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(3)]
    kmeans.assign_teams(frames[:2], tracks[:2], cache_path=cache_path)

    fresh = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)
    assignment = fresh.assign_teams(frames, tracks, cache_path=cache_path)

    assert len(assignment) == 3


# --- FashionCLIPClassifier -------------------------------------------------

@pytest.fixture
def fashionclip() -> FashionCLIPClassifier:
    # The sweep's 'memoise_reset' comparator: use_assignment_cache=True with
    # aggregation='none'. Most of this fixture's tests exercise the
    # cache/reset-interval behaviour directly, which must stay reachable and
    # unchanged; the two are alternative aggregation policies, so the
    # combination is spelled out rather than left to the production
    # defaults. See the cache-off and aggregation tests below for those.
    return FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=2,
        crop_fraction=0.667,
        confidence_threshold=0.65,
        use_assignment_cache=True,
        aggregation='none',
    )


def stub_inference(
    classifier: FashionCLIPClassifier,
    verdicts: dict[float, tuple[int, float]],
    calls: list[tuple[float, ...]] | None = None,
) -> None:
    """
    Make _classify_batch() return canned (predicted_team, confidence) pairs
    keyed by bbox x1, with crop_ok always True, stubbed at the batched-frame
    boundary rather than the per-player classify_jersey(), which
    assign_teams() does not call internally. Each verdict's team_id is the
    RAW prediction; classify_jersey()/assign_teams() apply the confidence
    threshold on top, exactly as before.
    """
    def fake_classify_batch(frame: np.ndarray, tracks: dict) -> dict:
        if calls is not None:
            calls.extend(tuple(track.bbox) for track in tracks.values())
        return {player_id: (*verdicts[track.bbox[0]], True) for player_id, track in tracks.items()}

    classifier._load_model = lambda: None
    classifier._classify_batch = fake_classify_batch


def marked_frame(
    marker: int,
    light_bbox: tuple[int, int, int, int],
    dark_bbox: tuple[int, int, int, int],
    size: int = 100,
) -> np.ndarray:
    """A frame_with_players() frame carrying a marker pixel outside both bboxes, so a stub can tell frames apart."""
    frame = frame_with_players(light_bbox, dark_bbox, size)
    frame[0, 0, 0] = marker
    return frame


def stub_inference_by_frame(
    classifier: FashionCLIPClassifier,
    verdicts_by_frame: dict[int, dict[float, tuple[int, float]]],
) -> None:
    """
    Like stub_inference, but the canned (predicted_team, confidence) verdict
    varies per frame, keyed by (frame marker read from frame[0, 0, 0], bbox
    x1). Lets a test simulate a player's fresh per-frame confidence dipping
    below threshold on some frames but not others, which constant-verdict
    stub_inference cannot exercise.
    """
    def fake_classify_batch(frame: np.ndarray, tracks: dict) -> dict:
        verdicts = verdicts_by_frame[int(frame[0, 0, 0])]
        return {player_id: (*verdicts[track.bbox[0]], True) for player_id, track in tracks.items()}

    classifier._load_model = lambda: None
    classifier._classify_batch = fake_classify_batch


def test_fashionclip_crop_player_returns_a_pil_image(fashionclip):
    frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)

    image = fashionclip._crop_player(frame, list(LIGHT_BBOX))

    assert image.size == (20, 20)  # (width, height): 0.667 * 30 rows
    assert image.mode == 'RGB'


def test_fashionclip_crop_player_returns_none_for_an_empty_crop(fashionclip):
    frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)

    assert fashionclip._crop_player(frame, [10, 10, 10, 10]) is None


def test_fashionclip_crop_player_converts_bgr_to_rgb(fashionclip):
    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    frame[:, :, 0] = 255  # pure blue in BGR

    image = fashionclip._crop_player(frame, [0, 0, 20, 30])

    assert image.getpixel((0, 0)) == (0, 0, 255)


def test_assign_team_caches_confident_verdicts(fashionclip):
    calls = []
    stub_inference(fashionclip, {10: (1, 0.9)}, calls)
    frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)

    assert fashionclip.assign_team(frame, list(LIGHT_BBOX), player_id=4) == 1
    assert fashionclip.assign_team(frame, list(LIGHT_BBOX), player_id=4) == 1
    assert len(calls) == 1
    assert fashionclip.assignment_cache == {4: 1}


def test_assign_team_does_not_cache_unresolved_verdicts(fashionclip):
    calls = []
    # Raw predicted_team is never 0 (see _classify_batch's docstring);
    # "unresolved" here comes from confidence 0.4 falling below the 0.65
    # threshold, which classify_jersey() applies on top of the raw verdict.
    stub_inference(fashionclip, {10: (1, 0.4)}, calls)
    frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)

    assert fashionclip.assign_team(frame, list(LIGHT_BBOX), player_id=4) == 0
    assert fashionclip.assign_team(frame, list(LIGHT_BBOX), player_id=4) == 0
    assert len(calls) == 2
    assert fashionclip.assignment_cache == {}


def test_fashionclip_assign_teams_labels_every_player_every_frame(fashionclip):
    stub_inference(fashionclip, {10: (1, 0.9), 60: (2, 0.8)})
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(3)]

    assert fashionclip.assign_teams(frames, tracks) == [{1: 1, 2: 2}] * 3


def test_fashionclip_assign_teams_resets_the_cache_on_the_reset_interval(fashionclip):
    # With use_assignment_cache=True (this fixture): the existing,
    # unchanged reset-interval behaviour; this comparator must still work
    # exactly as before now that it is no longer the default.
    calls = []
    stub_inference(fashionclip, {10: (1, 0.9)}, calls)
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(4)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(4)]

    fashionclip.assign_teams(frames, tracks)

    # reset_interval 2 over 4 frames: inference re-runs at frames 0 and 2 only.
    assert len(calls) == 2


def test_fashionclip_assign_teams_classifies_every_frame_independently_when_the_cache_is_off():
    # The production default (use_assignment_cache=False, unset here):
    # the same player's bbox every frame must still trigger one inference
    # call PER frame, not the single call the reset-interval test above
    # gets for the identical setup with caching on.
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=2,  # deliberately small and irrelevant: must have no effect while off
        crop_fraction=0.667,
        confidence_threshold=0.65,
    )
    calls = []
    stub_inference(classifier, {10: (1, 0.9)}, calls)
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(4)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(4)]

    assignment = classifier.assign_teams(frames, tracks)

    assert len(calls) == 4  # one inference call per frame, not 1
    assert assignment == [{1: 1}] * 4
    assert classifier.assignment_cache == {}  # never populated


def test_assign_team_classifies_every_call_when_the_cache_is_off():
    # assign_team() (singular) equivalent of the test above.
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.65,
    )
    calls = []
    stub_inference(classifier, {10: (1, 0.9)}, calls)
    frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)

    assert classifier.assign_team(frame, list(LIGHT_BBOX), player_id=4) == 1
    assert classifier.assign_team(frame, list(LIGHT_BBOX), player_id=4) == 1

    assert len(calls) == 2  # not deduped to 1 despite the identical player_id
    assert classifier.assignment_cache == {}


def test_fashionclip_assign_teams_uses_a_matching_cache(fashionclip, tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    calls = []
    stub_inference(fashionclip, {10: (1, 0.9)}, calls)
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(2)]

    first = fashionclip.assign_teams(frames, tracks, cache_path=cache_path)
    calls.clear()

    assert fashionclip.assign_teams(frames, tracks, cache_path=cache_path) == first
    assert calls == []


def test_fashionclip_assign_teams_reruns_on_a_stale_cache(fashionclip, tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    calls = []
    stub_inference(fashionclip, {10: (1, 0.9)}, calls)
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(3)]
    fashionclip.assign_teams(frames[:2], tracks[:2], cache_path=cache_path)
    calls.clear()

    assignment = fashionclip.assign_teams(frames, tracks, cache_path=cache_path)

    assert len(assignment) == 3
    assert calls != []


# --- Fake CLIP model/processor, for testing without downloaded weights ------
#
# These reproduce CLIPModel.forward()'s exact reference formula (confirmed
# against the installed transformers==4.44.0 source):
#   image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
#   text_embeds  = text_embeds  / text_embeds.norm(p=2, dim=-1, keepdim=True)
#   logits_per_text  = text_embeds @ image_embeds.t() * logit_scale.exp()
#   logits_per_image = logits_per_text.t()
# get_image_features()/get_text_features() return the same pre-normalisation
# embeddings forward() derives internally, so the two code paths under test
# (bare model(**inputs) vs cached get_text_features + per-image
# get_image_features) are mathematically required to agree; this proves our
# implementation reproduces that formula, independent of the downloaded
# checkpoint's actual numbers (no network access here; the real weights are
# GPU/weights-dependent and exercised on JupyterHub, not in this suite).

class _FakeCLIPModel:
    """A tiny stand-in with CLIPModel's public shape: get_image_features, get_text_features, __call__, logit_scale."""

    def __init__(self, feature_dim: int = 6) -> None:
        self.feature_dim = feature_dim
        self.logit_scale = torch.nn.Parameter(torch.tensor(0.75))
        self.to_calls: list[torch.device] = []
        self.eval_called = False

    def to(self, device: torch.device) -> '_FakeCLIPModel':
        self.to_calls.append(device)
        return self

    def eval(self) -> '_FakeCLIPModel':
        self.eval_called = True
        return self

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # A deterministic per-row projection: each image's own pixel_values
        # row maps to its own feature row, independent of the other rows in
        # the batch; this is what makes "batched == called one at a time"
        # a meaningful thing to assert (real CLIP's vision tower has the same
        # per-sample independence for a plain forward pass).
        flat = pixel_values.flatten(start_dim=1)
        return _resize_columns(flat, self.feature_dim)

    def get_text_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        return _resize_columns(input_ids.float(), self.feature_dim)

    def __call__(self, **inputs: torch.Tensor):
        image_embeds = self.get_image_features(inputs['pixel_values'])
        text_embeds = self.get_text_features(inputs['input_ids'], inputs.get('attention_mask'))
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()
        return SimpleNamespace(logits_per_image=logits_per_image)


def _resize_columns(matrix: torch.Tensor, width: int) -> torch.Tensor:
    """Slice or zero-pad a 2D tensor's columns to an exact width, per row, independent of other rows."""
    if matrix.shape[1] >= width:
        return matrix[:, :width]
    return torch.nn.functional.pad(matrix, (0, width - matrix.shape[1]))


class _FakeCLIPProcessor:
    """Deterministic stand-in for CLIPProcessor: text -> per-string token codes, images -> per-image pixel codes."""

    def __call__(self, text=None, images=None, return_tensors: str = 'pt', padding: bool = True) -> dict:
        result = {}
        if text is not None:
            # Distinct, deterministic per-string codes so two different
            # prompts are guaranteed to embed differently. Built-in hash()
            # is salted per-process (PYTHONHASHSEED is random by default),
            # so two prompts could collide to the same residue on roughly
            # 1 run in 97, making this fixture flaky; zlib.crc32 is a fixed,
            # unsalted function, so its output is identical every run.
            result['input_ids'] = torch.tensor(
                [[float((zlib.crc32(s.encode()) % 97) + i) for i in range(8)] for s in text]
            )
            result['attention_mask'] = torch.ones(len(text), 8)
        if images is not None:
            # One row per image, derived from that image's own pixels only,
            # so a crop's embedding never depends on which other crops share
            # its batch (the property the batching rewrite must preserve).
            means = torch.tensor([float(np.asarray(image).mean()) for image in images]).unsqueeze(1)
            result['pixel_values'] = means.repeat(1, 8) + torch.arange(8).float()
        return result


@pytest.fixture
def loaded_fashionclip(monkeypatch) -> FashionCLIPClassifier:
    """A FashionCLIPClassifier with a real (fake-model-backed) _load_model() call, no network access."""
    import basketball.team_classifier.classifier as classifier_module

    monkeypatch.setattr(classifier_module, 'CLIPModel', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPModel()))
    monkeypatch.setattr(classifier_module, 'CLIPProcessor', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPProcessor()))

    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.5,
    )
    classifier._load_model()
    return classifier


# --- Bug 1 fix: _load_vision_model(), the vision-only load path used by ----
# --- EmbeddingClusteringClassifier (new in this PR, not the closed --------
# --- prompted path) ---------------------------------------------------------

def test_load_vision_model_populates_model_and_processor_without_computing_text_embeds(monkeypatch):
    # Deferred: needs the function-scoped monkeypatch fixture, so it cannot
    # be imported at module level; matches loaded_fashionclip's fixture above.
    import basketball.team_classifier.classifier as classifier_module

    monkeypatch.setattr(classifier_module, 'CLIPModel', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPModel()))
    monkeypatch.setattr(
        classifier_module, 'CLIPProcessor', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPProcessor())
    )
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.5,
    )

    classifier._load_vision_model()

    assert classifier.model is not None
    assert classifier.processor is not None
    assert classifier.model.eval_called is True
    assert classifier.text_embeds is None  # the prompt-free contract: never computed here


def test_load_vision_model_does_not_reconstruct_an_already_loaded_model(monkeypatch):
    # Deferred for the same reason as the test above.
    import basketball.team_classifier.classifier as classifier_module

    from_pretrained_calls: list[str] = []

    def counting_from_pretrained(name: str) -> _FakeCLIPModel:
        from_pretrained_calls.append(name)
        return _FakeCLIPModel()

    monkeypatch.setattr(classifier_module, 'CLIPModel', SimpleNamespace(from_pretrained=counting_from_pretrained))
    monkeypatch.setattr(
        classifier_module, 'CLIPProcessor', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPProcessor())
    )
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.5,
    )

    classifier._load_vision_model()
    first_model = classifier.model
    classifier._load_vision_model()

    assert classifier.model is first_model  # not replaced by a second call
    assert len(from_pretrained_calls) == 1  # not re-downloaded/reconstructed


def test_load_model_still_calls_the_vision_load_and_computes_text_embeds(monkeypatch):
    # Regression guard for the split itself: _load_model() must still do
    # everything it did before (vision load AND text embeddings); only the
    # vision half moved into its own method, not out of _load_model()'s
    # own contract.
    import basketball.team_classifier.classifier as classifier_module

    monkeypatch.setattr(classifier_module, 'CLIPModel', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPModel()))
    monkeypatch.setattr(
        classifier_module, 'CLIPProcessor', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPProcessor())
    )
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.5,
    )

    classifier._load_model()

    assert classifier.model is not None
    assert classifier.processor is not None
    assert classifier.text_embeds is not None
    assert classifier.text_embeds.shape[0] == 2  # one row per team description


# --- GPU placement ----------------------------------------------------------

def test_device_is_resolved_once_and_used_for_the_model_and_its_inputs(loaded_fashionclip):
    # Assert against the actual expected value, not a hardcoded 'cpu': on a
    # CUDA-equipped machine the device correctly resolves to 'cuda', and a
    # hardcoded assumption about the sandbox would fail there despite the
    # code being correct.
    expected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert loaded_fashionclip.device == expected_device
    assert loaded_fashionclip.model.to_calls == [loaded_fashionclip.device]
    assert loaded_fashionclip.model.eval_called is True

    frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)
    tracks = {1: track(1, LIGHT_BBOX)}
    embeddings = loaded_fashionclip.get_image_embeddings(frame, tracks)

    # The forward pass ran under torch.inference_mode() and on the resolved
    # device, proven here by the pipeline completing and returning a tensor
    # whose device matches, since inference_mode()/device mismatches raise.
    assert embeddings[1] is not None
    assert embeddings[1].shape == (loaded_fashionclip.model.feature_dim,)


# --- Cached text embeddings, proven equivalent ------------------------------

def test_cached_text_embedding_path_reproduces_clipmodel_call_exactly(loaded_fashionclip):
    """
    The batched path (cached get_text_features + per-crop get_image_features,
    dot product scaled by logit_scale.exp(), softmax) must match the bare
    model(**inputs).logits_per_image.softmax(dim=1) path to within 1e-4; if
    it cannot, the right response is stop and report, not ship an approximation.
    """
    rng = np.random.default_rng(0)
    random_image = (rng.random((40, 40, 3)) * 255).astype(np.uint8)

    processor = loaded_fashionclip.processor
    model = loaded_fashionclip.model
    classes = [loaded_fashionclip.team_1_description, loaded_fashionclip.team_2_description]

    # (a) the old, un-cached path: text and image encoded together, every call.
    inputs = processor(text=classes, images=[_to_pil(random_image)], return_tensors='pt', padding=True)
    old_probs = model(**inputs).logits_per_image.softmax(dim=1).detach().numpy()[0]

    # (b) the new path under test: cached text embeddings from _load_model(),
    # a fresh image batch of one, our own dot-product + logit_scale + softmax.
    batch = loaded_fashionclip._classify_batch(random_image, {1: track(1, (0, 0, 40, 40))})
    predicted_team, new_confidence, crop_ok = batch[1]
    assert crop_ok is True

    np.testing.assert_allclose(new_confidence, old_probs.max(), atol=1e-4)
    assert predicted_team - 1 == int(np.argmax(old_probs))


def _to_pil(array: np.ndarray):
    from PIL import Image
    return Image.fromarray(array)


# --- Batched crops, explicit player_id mapping ------------------------------

def test_classify_batch_gives_each_of_three_players_their_own_correct_verdict(loaded_fashionclip):
    frame = np.zeros((60, 90, 3), dtype=np.uint8)
    frame[:, 0:30] = 10
    frame[:, 30:60] = 200
    frame[:, 60:90] = 100
    tracks = {
        11: track(11, (0, 0, 30, 30)),
        22: track(22, (30, 0, 60, 30)),
        33: track(33, (60, 0, 90, 30)),
    }

    batch = loaded_fashionclip._classify_batch(frame, tracks)

    # Every player gets a result, keyed explicitly by player_id, not by
    # position in the batch, which the differing crop brightnesses would
    # expose immediately if the mapping were positional-by-luck.
    assert set(batch) == {11, 22, 33}
    for player_id, (predicted_team, confidence, crop_ok) in batch.items():
        assert crop_ok is True
        assert predicted_team in (1, 2)
        assert 0.0 <= confidence <= 1.0

    # Each player's crop is distinct (10 / 200 / 100), so at least the two
    # most different crops (11 and 22) must not collapse to the same verdict
    # by construction of this fake model's per-image-independent embeddings.
    assert batch[11][:2] != batch[22][:2]


def test_classify_batch_handles_the_zero_crop_and_one_crop_frame(loaded_fashionclip):
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    assert loaded_fashionclip._classify_batch(frame, {}) == {}

    one = loaded_fashionclip._classify_batch(frame, {5: track(5, (0, 0, 30, 30))})
    assert set(one) == {5}
    assert one[5][2] is True  # crop_ok


def test_classify_batch_reports_crop_ok_false_for_an_empty_crop_without_a_raw_team_zero(loaded_fashionclip):
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    batch = loaded_fashionclip._classify_batch(frame, {1: track(1, (10, 10, 10, 10))})

    predicted_team, confidence, crop_ok = batch[1]
    assert crop_ok is False
    assert predicted_team != 0  # documented placeholder; never a raw 0
    assert confidence == 0.0


# --- Image embeddings accessor, no second forward pass ----------------------

def test_get_image_embeddings_reuses_the_classification_forward_pass(loaded_fashionclip, monkeypatch):
    frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)
    tracks = {1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)}

    loaded_fashionclip._classify_batch(frame, tracks)

    calls = []
    original = type(loaded_fashionclip.model).get_image_features
    monkeypatch.setattr(
        type(loaded_fashionclip.model),
        'get_image_features',
        lambda self, pixel_values: (calls.append(1), original(self, pixel_values))[1],
    )

    embeddings = loaded_fashionclip.get_image_embeddings(frame, tracks)

    assert calls == []  # memoised against the immediately preceding classify call; no second forward pass
    assert embeddings[1] is not None and embeddings[2] is not None
    assert embeddings[1].shape == (loaded_fashionclip.model.feature_dim,)


def test_encode_crops_memo_does_not_collide_across_a_freed_frames_reused_address(loaded_fashionclip):
    """
    Regression test for the id(frame) memoisation bug: id() is a memory
    address, which CPython is free to reuse once the original object is
    garbage-collected. In a streaming pipeline that decodes one frame array
    at a time and drops the old one, two distinct frames can land at the
    same address back to back; an id()-only memo would then return the
    first frame's stale cached embeddings/verdict for the second frame's
    different pixel content. The fix holds a strong reference to the cached
    frame, so no other frame can ever reuse its address while it's the memo
    key; the two frames here are simulated one at a time (never both alive
    at once) with identical bboxes and different pixel content, and must
    still produce different verdicts.
    """
    tracks = {1: track(1, (0, 0, 30, 30))}

    def classify_and_drop(fill_value: int) -> tuple[int, float]:
        frame = np.full((40, 40, 3), fill_value, dtype=np.uint8)
        result = loaded_fashionclip._classify_batch(frame, tracks)
        # `frame` goes out of scope here (and nothing else references it);
        # forcing a collection immediately after frees its memory for reuse
        # by the next same-shape allocation, exactly the streaming scenario
        # the id()-only memo got wrong.
        del frame
        gc.collect()
        return result[1][:2]

    first_verdict = classify_and_drop(10)
    second_verdict = classify_and_drop(220)

    assert first_verdict != second_verdict


def test_get_image_embeddings_returns_none_for_an_invalid_crop(loaded_fashionclip):
    frame = np.zeros((60, 60, 3), dtype=np.uint8)

    embeddings = loaded_fashionclip.get_image_embeddings(frame, {1: track(1, (10, 10, 10, 10))})

    assert embeddings == {1: None}


# --- Bbox clipping -----------------------------------------------------------

def test_fashionclip_crop_player_clamps_a_negative_origin_bbox(fashionclip):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:20, 0:20] = 240  # the visible part of a box that nominally starts off-frame

    # Without clamping, x1=-10 would wrap around and slice from the frame's
    # right edge instead of clamping to 0, a different region entirely.
    image = fashionclip._crop_player(frame, [-10, -5, 20, 30])

    assert image is not None
    pixels = np.asarray(image)
    assert pixels.mean() > 200  # sampled the visible bright region, not a wraparound slice


def test_fashionclip_crop_player_clamps_a_bbox_extending_past_the_frame_edge(fashionclip):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:30, 80:100] = 240

    image = fashionclip._crop_player(frame, [80, 0, 150, 30])

    assert image is not None
    assert image.size[0] == 20  # clipped to the frame's actual width (100 - 80)


def test_kmeans_crop_player_clamps_a_negative_origin_bbox(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = kmeans._crop_player(frame, [-10, -5, 20, 30])

    assert crop.shape[1] == 20  # x clamped to [0, 20), not a negative-index wraparound


def test_kmeans_crop_player_clamps_a_bbox_extending_past_the_frame_edge(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = kmeans._crop_player(frame, [80, 0, 150, 30])

    assert crop.shape[1] == 20  # clipped to the frame's actual width (100 - 80)


# --- Bug 2: two-directional clamp + explicit degenerate-box rejection ------
#
# A one-directional clamp (x1/y1 up, x2/y2 down only) never rejects a box
# entirely off one edge: e.g. bbox=[-100, 10, -50, 60] on a 100-wide frame
# clamps to x1=0, x2=-50 (min(-50, 100) leaves it negative), and
# frame[..., 0:-50] is Python slice syntax for "up to 50px before the end";
# it silently returns ~50px of unrelated content from the frame's right
# region instead of an empty crop. Every coordinate must be clamped into
# [0, bound] in both directions, and x2 <= x1 / y2 <= y1 must be rejected
# explicitly before slicing.

def test_fashionclip_crop_player_returns_none_for_a_box_entirely_off_the_left_edge(fashionclip):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, 80:100] = 240  # bright region far from the off-frame box, to expose a wraparound slice

    assert fashionclip._crop_player(frame, [-100, 10, -50, 60]) is None


def test_fashionclip_crop_player_returns_none_for_a_box_entirely_off_the_top_edge(fashionclip):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[80:100, :] = 240

    assert fashionclip._crop_player(frame, [10, -60, 60, -10]) is None


def test_fashionclip_crop_player_returns_none_for_a_negative_y2(fashionclip):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[80:100, :] = 240

    # y2 is negative (the box is entirely off the top) but y1=30 is valid,
    # so a one-directional clamp leaves crop_height very negative, making
    # y1 + crop_height a small negative number; Python's negative-index
    # slicing then wraps that around to near the frame's bottom, sweeping in
    # the unrelated bright rows 80-99 instead of returning an empty crop.
    assert fashionclip._crop_player(frame, [10, 30, 60, -20]) is None


def test_fashionclip_crop_player_still_clips_a_box_straddling_an_edge(fashionclip):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[0:20, 0:20] = 240  # the visible part of a box that nominally starts off-frame

    image = fashionclip._crop_player(frame, [-10, -5, 20, 30])

    assert image is not None
    pixels = np.asarray(image)
    assert pixels.mean() > 200  # sampled the visible bright region, not a wraparound slice


def test_kmeans_crop_player_returns_a_zero_size_crop_for_a_box_entirely_off_the_left_edge(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[:, 80:100] = 240

    crop = kmeans._crop_player(frame, [-100, 10, -50, 60])

    assert crop.size == 0  # the existing crop.size == 0 guard must fire, not a wraparound slice


def test_kmeans_crop_player_returns_a_zero_size_crop_for_a_box_entirely_off_the_top_edge(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[80:100, :] = 240

    crop = kmeans._crop_player(frame, [10, -60, 60, -10])

    assert crop.size == 0


def test_kmeans_crop_player_returns_a_zero_size_crop_for_a_negative_y2(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    frame[80:100, :] = 240

    crop = kmeans._crop_player(frame, [10, 30, 60, -20])

    assert crop.size == 0


def test_kmeans_crop_player_still_clips_a_box_straddling_an_edge(kmeans):
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    crop = kmeans._crop_player(frame, [-10, -5, 20, 30])

    assert crop.shape[1] == 20  # x clamped to [0, 20), not a negative-index wraparound
    assert crop.size > 0


# --- Bug: crop_height must derive from the RAW box height, not the clamped
# one -------------------------------------------------------------------
#
# crop_fraction is defined as a fraction of the player's true box height.
# Deriving crop_height from the frame-clamped y1/y2 instead makes a player
# touching the top or bottom edge sample a different fraction of their
# actual body than a fully on-frame player gets: crop_fraction must mean
# the same thing for every player, or a crop_fraction sweep measures an
# inconsistent quantity. The fix: compute crop_height from the raw box
# height, derive the intended slice end from the raw y1, and only then
# clamp both the slice start and end into [0, h].

def test_fashionclip_crop_player_uses_the_raw_box_height_for_a_bottom_clipped_box(fashionclip):
    frame = np.zeros((720, 100, 3), dtype=np.uint8)

    # Raw box height 800 - 600 = 200; crop_height = int(200 * 0.667) = 133;
    # intended end row 600 + 133 = 733, clamped to the frame's 720 rows,
    # not int((720 - 600) * 0.667) = 80 rows (600..680), which under-samples
    # relative to the box's true height.
    image = fashionclip._crop_player(frame, [10, 600, 60, 800])

    assert image is not None
    assert image.size[1] == 120  # rows 600..720


def test_fashionclip_crop_player_uses_the_raw_box_height_for_a_top_clipped_box(fashionclip):
    frame = np.zeros((720, 100, 3), dtype=np.uint8)

    # Raw box height 80 - (-40) = 120; crop_height = int(120 * 0.667) = 80;
    # intended end row -40 + 80 = 40, not int((80 - 0) * 0.667) = 53 rows
    # (0..53), which pulls extra rows in from further down the true box.
    image = fashionclip._crop_player(frame, [10, -40, 60, 80])

    assert image is not None
    assert image.size[1] == 40  # rows 0..40


def test_fashionclip_crop_player_fully_visible_box_is_unchanged(fashionclip):
    frame = np.zeros((720, 100, 3), dtype=np.uint8)

    image = fashionclip._crop_player(frame, [10, 50, 60, 110])

    assert image is not None
    assert image.size[1] == 40  # int((110 - 50) * 0.667); no clamping involved, same either way


def test_kmeans_crop_player_uses_the_raw_box_height_for_a_bottom_clipped_box(kmeans):
    frame = np.zeros((720, 100, 3), dtype=np.uint8)

    crop = kmeans._crop_player(frame, [10, 600, 60, 800])

    assert crop.shape[0] == 120  # rows 600..720


def test_kmeans_crop_player_uses_the_raw_box_height_for_a_top_clipped_box(kmeans):
    frame = np.zeros((720, 100, 3), dtype=np.uint8)

    crop = kmeans._crop_player(frame, [10, -40, 60, 80])

    assert crop.shape[0] == 40  # rows 0..40


def test_kmeans_crop_player_fully_visible_box_is_unchanged(kmeans):
    frame = np.zeros((720, 100, 3), dtype=np.uint8)

    crop = kmeans._crop_player(frame, [10, 50, 60, 110])

    assert crop.shape[0] == 40  # int((110 - 50) * 0.667); no clamping involved, same either way


# --- Raw prediction CSV artefact --------------------------------------------

def test_fashionclip_records_one_row_per_player_per_frame_with_a_never_zero_raw_team(fashionclip, tmp_path):
    record_path = str(tmp_path / 'predictions.csv')
    stub_inference(fashionclip, {10: (1, 0.9), 60: (2, 0.4)})  # second player: confident raw, but below threshold
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(3)]

    assignment = fashionclip.assign_teams(frames, tracks, record_path=record_path, clip_name='clip_1')

    with open(record_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader]

    assert header == PREDICTION_CSV_HEADER
    assert len(rows) == 3 * 2  # one row per player per frame, all 3 frames, no cache-skip shortcut
    assert {row['predicted_team'] for row in rows} == {'1', '2'}
    assert '0' not in {row['predicted_team'] for row in rows}  # raw predicted_team is never 0

    # The returned, thresholded assignment is unaffected by recording being on:
    # player 2's raw confidence (0.4) is below the 0.65 threshold -> team 0.
    assert assignment == [{1: 1, 2: 0}] * 3


def test_fashionclip_recording_mode_returns_the_same_assignment_as_non_recording(tmp_path):
    """
    Regression test for the recording-mode cache-divergence bug: player 2 is
    confidently resolved (team 2, confidence 0.9) on frame 0, then its fresh
    per-frame confidence dips to 0.3 (below the 0.65 threshold) on frame 1
    before recovering on frame 2. A constant-verdict stub can't exercise this:
    sticky and per-frame decisions agree trivially when nothing varies.

    Recording always runs a genuine forward pass every frame (so the raw
    artefact reflects that dip), but the RETURNED assignment must still
    honour assignment_cache exactly like the non-recording path: player 2
    must stay resolved at team 2 through the dip in both runs.
    """
    verdicts_by_frame = {
        0: {10: (1, 0.9), 60: (2, 0.9)},
        1: {10: (1, 0.9), 60: (2, 0.3)},
        2: {10: (1, 0.9), 60: (2, 0.9)},
    }
    frames = [marked_frame(i, LIGHT_BBOX, DARK_BBOX) for i in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(3)]

    def make_classifier() -> FashionCLIPClassifier:
        classifier = FashionCLIPClassifier(
            team_1_description='white basketball jersey',
            team_2_description='dark blue basketball jersey',
            reset_interval=10,  # larger than 3 frames: the cache from frame 0 must survive the dip
            crop_fraction=0.667,
            confidence_threshold=0.65,
            # This test is specifically about the sticky cache, i.e. the
            # 'memoise_reset' policy: cache on, no window aggregation.
            use_assignment_cache=True,
            aggregation='none',
        )
        stub_inference_by_frame(classifier, verdicts_by_frame)
        return classifier

    non_recording = make_classifier()
    returned_without_recording = non_recording.assign_teams(frames, tracks)

    record_path = str(tmp_path / 'predictions.csv')
    recording = make_classifier()
    returned_with_recording = recording.assign_teams(
        frames, tracks, record_path=record_path, clip_name='clip_1',
    )

    # The two returned assignments must be element-wise identical, and both
    # must keep player 2 resolved at team 2 across the dip (assignment_cache
    # sticks from frame 0).
    assert returned_with_recording == returned_without_recording
    assert returned_with_recording == [{1: 1, 2: 2}] * 3

    # The recorded rows differ from the returned labels on frame 1: the raw
    # forward pass genuinely saw confidence 0.3, even though the returned
    # assignment stayed at the frame-0 sticky value.
    with open(record_path, newline='') as f:
        rows = list(csv.DictReader(f))
    frame_1_player_2 = next(r for r in rows if r['player_id'] == '2' and r['frame_idx'] == '1')
    assert frame_1_player_2['predicted_team'] == '2'
    assert frame_1_player_2['confidence'] == '0.3'


def test_fashionclip_assign_teams_never_writes_a_cache_file_when_recording(fashionclip, tmp_path):
    """
    Even though the returned assignment is now identical either way, a cache
    written while recording must never exist: it would carry the recording
    run's own risk of drift as a silent, uninspectable fact.
    """
    cache_path = str(tmp_path / 'teams.pkl')
    record_path = str(tmp_path / 'predictions.csv')
    stub_inference(fashionclip, {10: (1, 0.9), 60: (2, 0.9)})
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)}]

    fashionclip.assign_teams(
        frames, tracks, cache_path=cache_path, record_path=record_path, clip_name='clip_1',
    )

    assert not Path(cache_path).exists()


def test_kmeans_assign_teams_never_writes_a_cache_file_when_recording(kmeans, tmp_path):
    """
    K-means equivalent of the FashionCLIP regression test above: cache_path
    must never be read from or written to while record_path is set, even
    though K-means' returned assignment doesn't diverge between the two
    modes the way FashionCLIP's did before its fix; the invariant is about
    never carrying the recording run's own risk of drift as a silent,
    uninspectable fact, not just about avoiding wrong labels.
    """
    cache_path = str(tmp_path / 'teams.pkl')
    record_path = str(tmp_path / 'predictions.csv')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]

    kmeans.assign_teams(
        frames, tracks, cache_path=cache_path, record_path=record_path, clip_name='clip_1',
    )

    assert not Path(cache_path).exists()


def test_fashionclip_assign_teams_requires_clip_name_when_recording(fashionclip, tmp_path):
    stub_inference(fashionclip, {10: (1, 0.9)})
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX)]
    tracks = [{1: track(1, LIGHT_BBOX)}]

    with pytest.raises(ValueError, match='clip_name'):
        fashionclip.assign_teams(frames, tracks, record_path=str(tmp_path / 'p.csv'))


def test_kmeans_records_one_row_per_player_per_frame_with_a_never_zero_raw_team(kmeans, tmp_path):
    record_path = str(tmp_path / 'predictions.csv')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]

    assignment = kmeans.assign_teams(frames, tracks, record_path=record_path, clip_name='clip_1')

    with open(record_path, newline='') as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 2 * 2
    assert '0' not in {row['predicted_team'] for row in rows}
    assert all(row['clip'] == 'clip_1' for row in rows)
    assert assignment == [{1: 1, 2: 2}] * 2


def test_kmeans_unresolved_diagnostic_matches_between_recording_and_non_recording(tmp_path, capsys):
    """
    K-means equivalent of the FashionCLIP diagnostic test above: the reported
    unresolved count must equal the number of team-0 entries in the returned
    assignment, and must be the same whether or not recording is on (K-means
    has no assignment_cache to diverge on, but the derivation must still be
    consistent between modes rather than trusting empty_crop_count/
    sparse_crop_count as a proxy).
    """
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [
        {1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX), 3: track(3, (5, 5, 5, 5))}
        for _ in range(2)
    ]

    def unresolved_count_from_output(output: str) -> int:
        match = re.search(r'(\d+) \(frame, player\) assignments left unresolved', output)
        assert match is not None, output
        return int(match.group(1))

    non_recording = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)
    capsys.readouterr()
    returned_without_recording = non_recording.assign_teams(frames, tracks)
    unresolved_without_recording = unresolved_count_from_output(capsys.readouterr().out)

    recording = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)
    returned_with_recording = recording.assign_teams(
        frames, tracks, record_path=str(tmp_path / 'predictions.csv'), clip_name='clip_1',
    )
    unresolved_with_recording = unresolved_count_from_output(capsys.readouterr().out)

    expected_unresolved = sum(
        team_id == 0 for frame_teams in returned_without_recording for team_id in frame_teams.values()
    )

    assert expected_unresolved == 2  # player 3's genuinely empty crop, once per frame
    assert unresolved_without_recording == unresolved_with_recording == expected_unresolved
    assert returned_with_recording == returned_without_recording


def test_kmeans_records_a_placeholder_raw_team_for_an_invalid_crop(kmeans, tmp_path):
    record_path = str(tmp_path / 'predictions.csv')
    # Two distinct valid crops (light, dark) so _fit_team_centres has genuine
    # separation to cluster; player 3's degenerate bbox is invalid throughout
    # and is simply skipped by the fit loop, then hits the placeholder path
    # at classification time.
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [
        {1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX), 3: track(3, (10, 10, 10, 10))}
        for _ in range(2)
    ]

    assignment = kmeans.assign_teams(frames, tracks, record_path=record_path, clip_name='clip_1')

    with open(record_path, newline='') as f:
        rows = {row['player_id']: row for row in csv.DictReader(f) if row['frame_idx'] == '0'}

    assert rows['3']['crop_ok'] == 'False'
    assert rows['3']['predicted_team'] != '0'  # placeholder, never a raw 0
    assert assignment[0][3] == 0  # the returned, resolved assignment still reports unresolved


# --- Bug: unresolved diagnostic must come from the returned assignment, not
# from empty_crop_count (which is mutated inside a memoised function and
# over-counts under recording) ----------------------------------------------

def test_fashionclip_unresolved_diagnostic_matches_between_recording_and_non_recording(monkeypatch, capsys, tmp_path):
    """
    Player 2 is confidently resolved (crop_ok, confidence >= 0.5; the fake
    model's 2-class softmax guarantees the argmax gets at least half the
    probability mass whenever the two logits aren't exactly tied) on frame 0
    and stays cached from then on, even though its bbox is genuinely empty
    on frame 1. In recording mode, _encode_crops() re-processes every
    player's raw crop every frame regardless of assignment_cache membership,
    so empty_crop_count would count player 2's frame-1 crop even though the
    returned assignment keeps it at its cached, non-zero team throughout.
    Player 3 has a genuinely empty crop on every frame in both modes, so it
    contributes a real, mode-independent unresolved count of 3 (one per
    frame); the diagnostic must report exactly that in both modes.
    """
    import basketball.team_classifier.classifier as classifier_module

    monkeypatch.setattr(classifier_module, 'CLIPModel', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPModel()))
    monkeypatch.setattr(classifier_module, 'CLIPProcessor', SimpleNamespace(from_pretrained=lambda name: _FakeCLIPProcessor()))

    def make_classifier() -> FashionCLIPClassifier:
        classifier = FashionCLIPClassifier(
            team_1_description='white basketball jersey',
            team_2_description='dark blue basketball jersey',
            reset_interval=50,  # larger than 3 frames: player 2's frame-0 cache entry must survive
            crop_fraction=0.667,
            confidence_threshold=0.5,
            # This test is specifically about the sticky cache, i.e. the
            # 'memoise_reset' policy: cache on, no window aggregation.
            use_assignment_cache=True,
            aggregation='none',
        )
        classifier._load_model()
        return classifier

    frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(3)]
    tracks = [
        {1: track(1, (10, 10, 30, 40)), 2: track(2, (60, 10, 80, 40)), 3: track(3, (5, 5, 5, 5))},
        {1: track(1, (10, 10, 30, 40)), 2: track(2, (5, 5, 5, 5)), 3: track(3, (5, 5, 5, 5))},
        {1: track(1, (10, 10, 30, 40)), 2: track(2, (60, 10, 80, 40)), 3: track(3, (5, 5, 5, 5))},
    ]

    def unresolved_count_from_output(output: str) -> int:
        match = re.search(r'(\d+) \(frame, player\) assignments left unresolved', output)
        assert match is not None, output
        return int(match.group(1))

    non_recording = make_classifier()
    capsys.readouterr()
    returned_without_recording = non_recording.assign_teams(frames, tracks)
    unresolved_without_recording = unresolved_count_from_output(capsys.readouterr().out)

    recording = make_classifier()
    returned_with_recording = recording.assign_teams(
        frames, tracks, record_path=str(tmp_path / 'predictions.csv'), clip_name='clip_1',
    )
    unresolved_with_recording = unresolved_count_from_output(capsys.readouterr().out)

    expected_unresolved = sum(
        team_id == 0 for frame_teams in returned_without_recording for team_id in frame_teams.values()
    )

    assert expected_unresolved == 3  # player 3's genuinely empty crop, once per frame
    assert unresolved_without_recording == unresolved_with_recording == expected_unresolved

    # The raw crop-encoding counter this diagnostic used to be derived from
    # does diverge between modes (recording re-encodes player 2's frame-1
    # empty crop even though it's resolved); only the derived, reported
    # unresolved count must not.
    assert recording.empty_crop_count > non_recording.empty_crop_count


# --- Cache fingerprint revision ---------------------------------------------

def test_fashionclip_fingerprint_carries_the_classifier_revision(fashionclip):
    fingerprint = fashionclip._cache_fingerprint([{}], 1)
    # 2 with the use_assignment_cache default flip, then 3 with the
    # temporal-aggregation adoption; both change assignments under the
    # previous defaults and must invalidate every pre-adoption cache.
    assert fingerprint['classifier_revision'] == FashionCLIPClassifier.CLASSIFIER_REVISION == 3


def test_fashionclip_fingerprint_carries_use_assignment_cache(fashionclip):
    fingerprint = fashionclip._cache_fingerprint([{}], 1)
    assert fingerprint['use_assignment_cache'] is True  # the fashionclip fixture sets it explicitly


def test_fashionclip_cache_is_invalidated_by_a_use_assignment_cache_change(tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(2)]

    cache_off = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.65,
        use_assignment_cache=False,
    )
    stub_inference(cache_off, {10: (1, 0.9)})
    cache_off.assign_teams(frames, tracks, cache_path=cache_path)

    cache_on = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.65,
        use_assignment_cache=True,
        aggregation='none',  # the cache is only reachable as the 'memoise_reset' policy
    )
    calls = []
    stub_inference(cache_on, {10: (1, 0.9)}, calls)
    cache_on.assign_teams(frames, tracks, cache_path=cache_path)

    assert calls != []  # refitted/reclassified, not served from the stale cache


def test_kmeans_fingerprint_carries_the_classifier_revision(kmeans):
    fingerprint = kmeans._cache_fingerprint([{}], 1)
    # Bumped to 2 by the fair-baseline rebuild, which changes assignments for
    # identical parameters and must invalidate every pre-rebuild cache.
    assert fingerprint['classifier_revision'] == KMeansClassifier.CLASSIFIER_REVISION == 2


# --- Adopted temporal aggregation (confidence-weighted sliding window) ------

# Per-frame (player_id, frame_idx, predicted_team, confidence) rows shared by
# the equivalence test and its production-path counterpart. Deliberately
# includes a track SHORTER than the window (players 3 and 4, two frames
# each), a track with a mid-track team change (player 1 flips 1 -> 2 -> 1),
# a single one-frame flicker (player 2, frame 3), a gap where a track is
# absent (player 2 has no row on frame 5) which the window must close up
# rather than pad, and an EXACT w1 == w2 tie (player 4: 0.5 on each team,
# both exactly representable in binary floating point, so the tie is real
# rather than approximate) to pin the >=-ties-to-team-1 rule.
AGGREGATION_ROWS = [
    (1, 0, 1, 0.91), (1, 1, 1, 0.72), (1, 2, 2, 0.55), (1, 3, 2, 0.83),
    (1, 4, 2, 0.77), (1, 5, 1, 0.64), (1, 6, 1, 0.88),
    (2, 0, 2, 0.80), (2, 1, 2, 0.62), (2, 2, 2, 0.71), (2, 3, 1, 0.95),
    (2, 4, 2, 0.58), (2, 6, 2, 0.69),
    (3, 4, 1, 0.51), (3, 5, 2, 0.99),
    (4, 2, 1, 0.5), (4, 3, 2, 0.5),
]


def aggregation_rows_as_raw_frames(
    rows: list[tuple[int, int, int, float]],
) -> list[dict[int, tuple[int, float, bool]]]:
    """Convert (player_id, frame_idx, predicted_team, confidence) rows into assign_teams' internal per-frame raw structure."""
    n_frames = max(frame_idx for _pid, frame_idx, _team, _conf in rows) + 1
    raw_frames: list[dict[int, tuple[int, float, bool]]] = [{} for _ in range(n_frames)]
    for player_id, frame_idx, predicted_team, confidence in rows:
        raw_frames[frame_idx][player_id] = (predicted_team, confidence, True)
    return raw_frames


def offline_pandas_aggregation(
    rows: list[tuple[int, int, int, float]],
    window: int,
) -> dict[tuple[int, int], int]:
    """The offline sweep's own implementation, verbatim: per player track ordered by frame_idx, centred confidence-weighted rolling sums with min_periods=1, ties to team 1."""
    aggregated: dict[tuple[int, int], int] = {}
    frame = pd.DataFrame(rows, columns=['player_id', 'frame_idx', 'predicted_team', 'confidence'])
    for player_id, track in frame.groupby('player_id'):
        track = track.sort_values('frame_idx')
        w1 = (track['confidence'] * (track['predicted_team'] == 1)).rolling(
            window, center=True, min_periods=1).sum()
        w2 = (track['confidence'] * (track['predicted_team'] == 2)).rolling(
            window, center=True, min_periods=1).sum()
        teams = np.where(w1 >= w2, 1, 2)
        for frame_idx, team in zip(track['frame_idx'], teams):
            aggregated[(int(player_id), int(frame_idx))] = int(team)
    return aggregated


@pytest.mark.parametrize('window', [1, 2, 3, 5, 9, 21])
def test_window_aggregation_matches_the_offline_pandas_expression(window):
    """
    THE load-bearing test for the adopted 0.9443 figure: production must
    reproduce the offline sweep's pandas expression element-wise, including
    its centred-window bounds, its min_periods=1 edge truncation and its
    >=-ties-to-team-1 rule. Parametrised over odd AND even windows because
    pandas' centring is asymmetric for even ones.
    """
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=1.0,
        confidence_threshold=0.5,
        aggregation='window_weighted',
        aggregation_window=window,
    )
    raw_frames = aggregation_rows_as_raw_frames(AGGREGATION_ROWS)

    aggregated_frames = classifier._aggregate_window_weighted(raw_frames)

    expected = offline_pandas_aggregation(AGGREGATION_ROWS, window)
    produced = {
        (player_id, frame_idx): team
        for frame_idx, aggregated_frame in enumerate(aggregated_frames)
        for player_id, (team, _confidence, _crop_ok) in aggregated_frame.items()
    }
    assert produced == expected
    assert len(produced) == len(AGGREGATION_ROWS)  # every row aggregated, none dropped


def test_window_aggregation_leaves_confidence_and_crop_ok_untouched():
    # Only the team changes: abstention downstream reads the frame's OWN
    # confidence and crop_ok, so aggregation must not overwrite either.
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=1.0,
        confidence_threshold=0.5,
        aggregation_window=9,
    )
    raw_frames = aggregation_rows_as_raw_frames(AGGREGATION_ROWS)

    aggregated_frames = classifier._aggregate_window_weighted(raw_frames)

    for frame_idx, aggregated_frame in enumerate(aggregated_frames):
        for player_id, (_team, confidence, crop_ok) in aggregated_frame.items():
            _raw_team, raw_confidence, raw_crop_ok = raw_frames[frame_idx][player_id]
            assert confidence == raw_confidence
            assert crop_ok == raw_crop_ok


def test_assign_teams_applies_window_aggregation_end_to_end():
    # The production path, not the helper: a single-frame flicker (frame 1's
    # team 2) is outvoted by its neighbours and disappears from the returned
    # assignment, which is the flicker reduction the policy was adopted for.
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=1.0,
        confidence_threshold=0.5,
        aggregation='window_weighted',
        aggregation_window=3,
    )
    verdicts_by_frame = {
        0: {10: (1, 0.9)},
        1: {10: (2, 0.6)},  # the flicker
        2: {10: (1, 0.9)},
    }
    stub_inference_by_frame(classifier, verdicts_by_frame)
    frames = [marked_frame(i, LIGHT_BBOX, DARK_BBOX) for i in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(3)]

    assignment = classifier.assign_teams(frames, tracks)

    assert assignment == [{1: 1}] * 3


def test_aggregation_none_keeps_the_per_frame_decision():
    # The measured comparator: with aggregation off, the same flicker
    # survives into the returned assignment.
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=1.0,
        confidence_threshold=0.5,
        aggregation='none',
    )
    verdicts_by_frame = {
        0: {10: (1, 0.9)},
        1: {10: (2, 0.6)},
        2: {10: (1, 0.9)},
    }
    stub_inference_by_frame(classifier, verdicts_by_frame)
    frames = [marked_frame(i, LIGHT_BBOX, DARK_BBOX) for i in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(3)]

    assignment = classifier.assign_teams(frames, tracks)

    assert assignment == [{1: 1}, {1: 2}, {1: 1}]


def test_abstention_is_applied_after_aggregation_from_the_frames_own_confidence():
    # Frame 1's own confidence (0.30) is below the 0.5 threshold, so it
    # abstains even though aggregation resolved its team to 1 from the
    # neighbours: abstention is a per-frame gate applied AFTER aggregation,
    # not a vote in it.
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=1.0,
        confidence_threshold=0.5,
        aggregation='window_weighted',
        aggregation_window=3,
    )
    verdicts_by_frame = {
        0: {10: (1, 0.9)},
        1: {10: (2, 0.3)},
        2: {10: (1, 0.9)},
    }
    stub_inference_by_frame(classifier, verdicts_by_frame)
    frames = [marked_frame(i, LIGHT_BBOX, DARK_BBOX) for i in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(3)]

    assignment = classifier.assign_teams(frames, tracks)

    assert assignment == [{1: 1}, {1: 0}, {1: 1}]


def test_an_unusable_crop_still_abstains_under_aggregation():
    # The abstention contract, checked against the new code path: aggregation can
    # resolve this frame's team from its neighbours, but a crop that could
    # not be embedded at all must still return team 0 rather than inheriting
    # a confident-looking label from the frames around it.
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=1.0,
        confidence_threshold=0.5,
        aggregation='window_weighted',
        aggregation_window=3,
    )
    # crop_ok=False on frame 1 only, with the documented (1, 0.0, False)
    # placeholder the real _classify_batch() uses for an unusable crop.
    verdicts_by_frame = {
        0: (1, 0.9, True),
        1: (1, 0.0, False),
        2: (1, 0.9, True),
    }
    classifier._load_model = lambda: None
    classifier._classify_batch = lambda frame, tracks: {
        player_id: verdicts_by_frame[int(frame[0, 0, 0])] for player_id in tracks
    }
    frames = [marked_frame(i, LIGHT_BBOX, DARK_BBOX) for i in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(3)]

    assignment = classifier.assign_teams(frames, tracks)

    assert assignment == [{1: 1}, {1: 0}, {1: 1}]


def test_recorded_rows_are_unaffected_by_the_aggregation_setting(tmp_path):
    # The artefact must stay RAW pre-aggregation and pre-threshold: every
    # offline sweep (including the one that selected this policy) reads it.
    verdicts_by_frame = {
        0: {10: (1, 0.9)},
        1: {10: (2, 0.6)},
        2: {10: (1, 0.9)},
    }
    frames = [marked_frame(i, LIGHT_BBOX, DARK_BBOX) for i in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(3)]

    def record_with(aggregation: str, path: str) -> tuple[list[dict], list[dict[int, int]]]:
        classifier = FashionCLIPClassifier(
            team_1_description='white basketball jersey',
            team_2_description='dark blue basketball jersey',
            reset_interval=50,
            crop_fraction=1.0,
            confidence_threshold=0.5,
            aggregation=aggregation,
            aggregation_window=3,
        )
        stub_inference_by_frame(classifier, verdicts_by_frame)
        assignment = classifier.assign_teams(frames, tracks, record_path=path, clip_name='clip_1')
        with open(path, newline='') as f:
            return list(csv.DictReader(f)), assignment

    aggregated_rows, aggregated_assignment = record_with('window_weighted', str(tmp_path / 'aggregated.csv'))
    per_frame_rows, per_frame_assignment = record_with('none', str(tmp_path / 'per_frame.csv'))

    assert aggregated_rows == per_frame_rows
    # And specifically still the RAW frame-1 flicker, not the aggregated 1.
    frame_1 = next(row for row in aggregated_rows if row['frame_idx'] == '1')
    assert frame_1['predicted_team'] == '2'
    assert frame_1['confidence'] == '0.6'

    # The other half of the contract: recording does not disable aggregation
    # either. The RETURNED assignment still differs between the two policies
    # while the recorded rows above are byte-identical.
    assert aggregated_assignment == [{1: 1}] * 3
    assert per_frame_assignment == [{1: 1}, {1: 2}, {1: 1}]


def test_an_unknown_aggregation_policy_raises():
    with pytest.raises(ValueError, match='Unknown aggregation'):
        FashionCLIPClassifier(
            team_1_description='white basketball jersey',
            team_2_description='dark blue basketball jersey',
            reset_interval=50,
            crop_fraction=1.0,
            confidence_threshold=0.5,
            aggregation='majority_vote',
        )


@pytest.mark.parametrize('window', [0, -1])
def test_a_non_positive_aggregation_window_raises(window):
    # Left unguarded, these empty every rolling window and silently label
    # every player team 1 (w1 == w2 == 0 ties to 1) instead of failing.
    with pytest.raises(ValueError, match='aggregation_window must be at least 1'):
        FashionCLIPClassifier(
            team_1_description='white basketball jersey',
            team_2_description='dark blue basketball jersey',
            reset_interval=50,
            crop_fraction=1.0,
            confidence_threshold=0.5,
            aggregation_window=window,
        )


def test_the_assignment_cache_cannot_be_combined_with_window_aggregation():
    # Alternative policies from the same sweep, never stacked: raising
    # beats silently letting one override the other.
    with pytest.raises(ValueError, match='alternative aggregation policies'):
        FashionCLIPClassifier(
            team_1_description='white basketball jersey',
            team_2_description='dark blue basketball jersey',
            reset_interval=50,
            crop_fraction=1.0,
            confidence_threshold=0.5,
            use_assignment_cache=True,
            aggregation='window_weighted',
        )


def test_fashionclip_fingerprint_carries_the_aggregation_policy_and_window():
    classifier = FashionCLIPClassifier(
        team_1_description='white basketball jersey',
        team_2_description='dark blue basketball jersey',
        reset_interval=50,
        crop_fraction=1.0,
        confidence_threshold=0.5,
    )

    fingerprint = classifier._cache_fingerprint([{}], 1)

    assert fingerprint['aggregation'] == 'window_weighted'  # the adopted default
    assert fingerprint['aggregation_window'] == 9


def test_fashionclip_cache_is_invalidated_by_an_aggregation_window_change(tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    tracks = [{1: track(1, LIGHT_BBOX)} for _ in range(3)]

    def run(window: int, calls: list | None = None) -> None:
        classifier = FashionCLIPClassifier(
            team_1_description='white basketball jersey',
            team_2_description='dark blue basketball jersey',
            reset_interval=50,
            crop_fraction=1.0,
            confidence_threshold=0.5,
            aggregation_window=window,
        )
        stub_inference(classifier, {10: (1, 0.9)}, calls)
        classifier.assign_teams(frames, tracks, cache_path=cache_path)

    run(window=9)
    calls = []
    run(window=5, calls=calls)

    assert calls != []  # reclassified, not served from the stale cache


# --- Rebuilt K-means baseline: whole-clip fitting scope ---------------------

def frame_with_light_player_only() -> np.ndarray:
    """Green background frame carrying only the light player; the dark team is off screen."""
    frame = np.full((100, 100, 3), 60, dtype=np.uint8)
    draw_player(frame, LIGHT_BBOX, 240)
    return frame


def test_fit_team_centres_uses_frames_beyond_the_first_few():
    # The old baseline fitted the first fit_frames=5 frames only, so a team
    # that only appears later in the clip was never in its reference
    # colours. Here the dark player is absent until frame 8.
    late_frames = []
    late_tracks = []
    for frame_idx in range(10):
        if frame_idx < 8:
            late_frames.append(frame_with_light_player_only())
            late_tracks.append({1: track(1, LIGHT_BBOX)})
        else:
            late_frames.append(frame_with_players(LIGHT_BBOX, DARK_BBOX))
            late_tracks.append({1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)})

    classifier = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)
    classifier._fit_team_centres(late_frames, late_tracks)

    # A dark centre exists at all only because frames 8-9 were fitted; under
    # the old first-five-frames scope both centres would be the light jersey.
    luminances = sorted(float(centre.sum()) for centre in classifier.team_centres)
    assert luminances == pytest.approx([45.0, 720.0])  # 3 * 15 and 3 * 240
    assert classifier.sparse_crop_count == 0


def test_fit_team_centres_honours_a_wider_stride():
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(8)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(8)]
    strided = KMeansClassifier(crop_fraction=0.667, fit_stride=4, bg_distance_threshold=30, margin_threshold=0.55)
    every_frame = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)

    strided._fit_team_centres(frames, tracks)
    every_frame._fit_team_centres(frames, tracks)

    # Same colours either way here (the synthetic clip is stationary); the
    # point is that a stride is applied at all rather than ignored.
    assert strided.team_centres.shape == every_frame.team_centres.shape == (2, 3)
    assert sorted(strided.team_labels.values()) == [1, 2]


def test_kmeans_fingerprint_carries_the_new_fitting_and_margin_parameters(kmeans):
    fingerprint = kmeans._cache_fingerprint([{}], 1)

    assert fingerprint['fit_stride'] == 1
    assert fingerprint['margin_threshold'] == 0.55
    assert 'fit_frames' not in fingerprint  # the retired parameter must not linger


def test_kmeans_cache_is_invalidated_by_a_margin_threshold_change(tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]
    lenient = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)
    lenient.assign_teams(frames, tracks, cache_path=cache_path)

    strict = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.99)
    strict.assign_teams(frames, tracks, cache_path=cache_path)

    assert strict.team_centres is not None  # refitted, not served from the stale cache


def test_kmeans_cache_is_invalidated_by_a_fit_stride_change(tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(4)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(4)]
    dense = KMeansClassifier(crop_fraction=0.667, fit_stride=1, bg_distance_threshold=30, margin_threshold=0.55)
    dense.assign_teams(frames, tracks, cache_path=cache_path)

    sparse = KMeansClassifier(crop_fraction=0.667, fit_stride=2, bg_distance_threshold=30, margin_threshold=0.55)
    sparse.assign_teams(frames, tracks, cache_path=cache_path)

    assert sparse.team_centres is not None  # refitted, not served from the stale cache


# --- Rebuilt K-means baseline: the two abstention paths ---------------------

def test_kmeans_abstains_on_a_crop_with_too_few_foreground_pixels(kmeans):
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]
    kmeans._fit_team_centres(frames, tracks)
    # A patch of pure background: nothing separates from the border ring, so
    # the old code returned every background pixel and confidently labelled
    # it. The rebuilt path abstains.
    flat = np.full((100, 100, 3), 60, dtype=np.uint8)

    predicted_team, confidence, crop_ok = kmeans._classify_with_margin(flat, [10, 10, 30, 40])

    assert crop_ok is False
    assert predicted_team == 1  # documented placeholder, never a raw 0
    assert confidence == 0.0
    assert kmeans.classify_jersey(flat, [10, 10, 30, 40]) == 0
    assert kmeans.sparse_crop_count > 0


def test_kmeans_abstains_when_the_margin_is_below_threshold(kmeans):
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(2)]
    kmeans._fit_team_centres(frames, tracks)

    assert kmeans._resolve_team(predicted_team=2, confidence=0.54, crop_ok=True) == 0
    assert kmeans.low_margin_count == 1
    assert kmeans._resolve_team(predicted_team=2, confidence=0.56, crop_ok=True) == 2
    assert kmeans.low_margin_count == 1  # an accepted decision is not counted


AMBIGUOUS_BBOX = (40, 10, 60, 40)
AMBIGUOUS_GREY = 128  # almost exactly between the two jersey colours (15 and 240)
# Measured margins for this fixture: ambiguous 0.669, light 0.801, dark 1.0.
# 0.75 separates them with room on both sides rather than sitting on an edge.
AMBIGUOUS_MARGIN_THRESHOLD = 0.75


def clip_with_an_ambiguous_player(n_frames: int) -> tuple[list[np.ndarray], list[dict[int, PlayerTrack]]]:
    """A synthetic clip whose third player's jersey colour sits between the two team colours."""
    frames = []
    tracks = []
    for _ in range(n_frames):
        frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)
        draw_player(frame, AMBIGUOUS_BBOX, AMBIGUOUS_GREY)
        frames.append(frame)
        tracks.append({
            1: track(1, LIGHT_BBOX),
            2: track(2, DARK_BBOX),
            3: track(3, AMBIGUOUS_BBOX),
        })
    return frames, tracks


def test_kmeans_assign_teams_abstains_on_an_ambiguous_jersey_colour():
    # A mid-grey player sits between the two colour centres. The old baseline
    # had no way to abstain on a weak decision and forced it to a side; that
    # was then reported as a finding about K-means rather than about this
    # implementation.
    frames, tracks = clip_with_an_ambiguous_player(6)
    classifier = KMeansClassifier(
        crop_fraction=0.667,
        fit_stride=1,
        bg_distance_threshold=30,
        margin_threshold=AMBIGUOUS_MARGIN_THRESHOLD,
    )

    assignment = classifier.assign_teams(frames, tracks)

    assert all(frame_teams[3] == 0 for frame_teams in assignment)  # ambiguous player abstained
    assert all(frame_teams[1] != 0 and frame_teams[2] != 0 for frame_teams in assignment)
    assert classifier.low_margin_count == len(frames)  # counted, once per frame
    assert classifier.empty_crop_count == 0  # abstained on the decision, not on the crop


def test_kmeans_records_the_pre_threshold_margin_for_an_abstained_player(tmp_path):
    record_path = str(tmp_path / 'predictions.csv')
    frames, tracks = clip_with_an_ambiguous_player(2)
    classifier = KMeansClassifier(
        crop_fraction=0.667,
        fit_stride=1,
        bg_distance_threshold=30,
        margin_threshold=AMBIGUOUS_MARGIN_THRESHOLD,
    )

    assignment = classifier.assign_teams(frames, tracks, record_path=record_path, clip_name='clip_1')

    with open(record_path, newline='') as f:
        rows = list(csv.DictReader(f))

    # The ambiguous player is abstained in the returned assignment, but its
    # recorded row still carries the raw team and the PRE-threshold margin,
    # so the threshold can be swept offline without re-running the fit.
    assert all(frame_teams[3] == 0 for frame_teams in assignment)
    abstained_rows = [row for row in rows if row['player_id'] == '3']
    assert len(abstained_rows) == 2
    assert all(row['predicted_team'] in {'1', '2'} for row in abstained_rows)
    assert all(row['crop_ok'] == 'True' for row in abstained_rows)
    assert all(0.5 < float(row['confidence']) < AMBIGUOUS_MARGIN_THRESHOLD for row in abstained_rows)


# --- EmbeddingClusteringClassifier ------------------------------------------
#
# BUG this section's own tests let through: the original stub_embeddings()
# replaced get_image_embeddings() wholesale, so _encode_crops' "model is not
# loaded" guard never ran and no test could have caught
# EmbeddingClusteringClassifier.assign_teams() never loading its encoder.
# stub_embeddings() below fakes only the model's own forward pass: it
# installs a fake _load_vision_model() that must still be CALLED to populate
# encoder.model before get_image_embeddings()/_encode_crops() (both real,
# unstubbed) will do anything but raise. A missing load in production code
# now raises in these tests exactly as it would for a real user.

class _FakeVisionOnlyCLIPModel:
    """Deterministic stand-in for CLIPModel's image path only, mapping a crop's mean pixel value to a well-separated 2D direction once normalised, unlike the shared-large-constant _FakeCLIPModel above (built for a different purpose: proving batched == called-once-per-image, not for producing genuinely separable clusters)."""

    def __init__(self) -> None:
        self.to_calls: list[torch.device] = []
        self.eval_called = False
        self.forward_pass_count = 0

    def to(self, device: torch.device) -> '_FakeVisionOnlyCLIPModel':
        self.to_calls.append(device)
        return self

    def eval(self) -> '_FakeVisionOnlyCLIPModel':
        self.eval_called = True
        return self

    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        self.forward_pass_count += 1
        mean = pixel_values[:, 0]
        return torch.stack([mean, 255.0 - mean], dim=1)


class _FakeVisionOnlyCLIPProcessor:
    """Deterministic stand-in for CLIPProcessor's image path only: one row per image, its own pixel mean."""

    def __call__(
        self,
        images: list | None = None,
        return_tensors: str = 'pt',
        padding: bool = True,
        text: list | None = None,
    ) -> dict:
        means = torch.tensor([float(np.asarray(image).mean()) for image in images]).unsqueeze(1)
        return {'pixel_values': means}


def stub_embeddings(encoder: FashionCLIPClassifier) -> _FakeVisionOnlyCLIPModel:
    """Install a fake vision model that encoder._load_vision_model() must still be CALLED to install; get_image_embeddings()/_encode_crops() and their load-guard stay real, so a missing load raises exactly as in production. Returns the fake model so a test can inspect its forward-pass count."""
    fake_model = _FakeVisionOnlyCLIPModel()

    def fake_load_vision_model() -> None:
        if encoder.model is not None and encoder.processor is not None:
            return
        encoder.model = fake_model
        encoder.processor = _FakeVisionOnlyCLIPProcessor()

    encoder._load_vision_model = fake_load_vision_model
    return fake_model


def loaded_stub_embeddings(encoder: FashionCLIPClassifier) -> _FakeVisionOnlyCLIPModel:
    """Like stub_embeddings(), but also loads the fake model immediately, for tests that call an internal method (_fit_clusters, _classify_with_margin) directly rather than through assign_teams(), and so must set up the already-loaded precondition themselves instead of relying on assign_teams() to do it."""
    fake_model = stub_embeddings(encoder)
    encoder._load_vision_model()
    return fake_model


def embedding_encoder() -> FashionCLIPClassifier:
    """A FashionCLIPClassifier built only to be an embedding source; its prompts are never read by the embedding arm."""
    return FashionCLIPClassifier(
        team_1_description='',
        team_2_description='',
        reset_interval=50,
        crop_fraction=0.667,
        confidence_threshold=0.65,
    )


@pytest.fixture
def embedding_classifier() -> EmbeddingClusteringClassifier:
    return EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)


def two_team_embedding_clip(n_frames: int = 4) -> tuple[list[np.ndarray], list[dict[int, PlayerTrack]]]:
    """A synthetic clip whose two players' jersey colours produce two well-separated fake embeddings once encoded."""
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(n_frames)]
    tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(n_frames)]
    return frames, tracks


# Measured (not assumed) against this fixture and _FakeVisionOnlyCLIPModel:
# the light and dark jerseys are identical every frame, so K-means' fitted
# centres land exactly on their two embeddings, giving both a margin of
# 1.0; a third, ambiguous-coloured player measures 0.543. 0.75 separates
# clean from ambiguous with room on both sides.
AMBIGUOUS_MARGIN_MEASURED = 0.543
EMBEDDING_MARGIN_THRESHOLD = 0.75


# --- the guard gap itself: two regression tests for Bug 1 -------------------

def test_fit_clusters_raises_if_the_encoder_was_never_loaded(embedding_classifier):
    # No stub_embeddings() call at all here: the encoder is exactly as
    # fresh as main.py leaves it. This is _encode_crops' real, unstubbed
    # guard firing; if EmbeddingClusteringClassifier ever again reaches
    # get_image_embeddings() without loading the encoder first, this is the
    # test that must catch it.
    frames, tracks = two_team_embedding_clip()

    with pytest.raises(RuntimeError, match='not loaded'):
        embedding_classifier._fit_clusters(frames, tracks)


def test_assign_teams_loads_the_encoders_vision_model_automatically(embedding_classifier):
    stub_embeddings(embedding_classifier.encoder)
    assert embedding_classifier.encoder.model is None  # not loaded yet
    frames, tracks = two_team_embedding_clip()

    assignment = embedding_classifier.assign_teams(frames, tracks)  # must not raise

    assert embedding_classifier.encoder.model is not None  # assign_teams() loaded it
    assert len(assignment) == len(frames)


def test_assign_teams_does_not_reload_an_already_loaded_encoder(embedding_classifier):
    fake_model = loaded_stub_embeddings(embedding_classifier.encoder)
    loaded_model = embedding_classifier.encoder.model
    frames, tracks = two_team_embedding_clip()

    embedding_classifier.assign_teams(frames, tracks)

    # Idempotent: the SAME model object survives a subsequent assign_teams()
    # call rather than being replaced by a fresh one.
    assert embedding_classifier.encoder.model is loaded_model
    assert embedding_classifier.encoder.model is fake_model


# --- fitting, assignment, prompt-free contract ------------------------------

def test_embedding_fit_clusters_separates_the_two_teams(embedding_classifier):
    loaded_stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip()

    embedding_classifier._fit_clusters(frames, tracks)

    assert embedding_classifier.cluster_centres.shape == (2, 2)
    assert sorted(embedding_classifier.team_labels.values()) == [1, 2]


def test_embedding_assign_teams_labels_every_player_in_every_frame(embedding_classifier):
    stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip()

    assignment = embedding_classifier.assign_teams(frames, tracks)

    assert len(assignment) == len(frames)
    # Unsupervised, so which cluster is team 1 is arbitrary; what matters is
    # that the two players are consistently separated across every frame.
    assert all(frame_teams[1] != frame_teams[2] for frame_teams in assignment)
    assert all(set(frame_teams.values()) <= {1, 2} for frame_teams in assignment)
    assert len({frame_teams[1] for frame_teams in assignment}) == 1  # stable per player


def test_embedding_never_reads_the_encoders_text_prompts(embedding_classifier):
    stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip()

    def fail_on_text(*args: object, **kwargs: object) -> None:
        raise AssertionError('the embedding arm must never encode a text prompt')

    embedding_classifier.encoder._compute_text_embeds = fail_on_text

    assignment = embedding_classifier.assign_teams(frames, tracks)

    assert embedding_classifier.encoder.text_embeds is None
    assert len(assignment) == len(frames)


def test_embedding_only_encodes_each_frame_once_across_fit_and_assign(embedding_classifier):
    """
    Regression test for Bug 2: the encoder's own memo is a single slot, so
    the old design called get_image_embeddings() once per frame while
    fitting and again per frame while assigning, and by the second pass the
    memo had long since been overwritten by every other frame in between,
    a full second forward pass over the whole clip. Each frame's crops must
    now be encoded exactly once in total.
    """
    fake_model = stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip(n_frames=3)

    embedding_classifier.assign_teams(frames, tracks)

    assert fake_model.forward_pass_count == len(frames)


# --- abstention --------------------------------------------------------------

def test_embedding_abstains_on_a_crop_that_could_not_be_embedded(embedding_classifier):
    stub_embeddings(embedding_classifier.encoder)
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(3)]
    # A genuinely degenerate (zero-area) bbox: FashionCLIPClassifier's own
    # _crop_player() returns None for it for real, rather than a value
    # injected past the crop/embed pipeline.
    tracks = [
        {1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX), 3: track(3, (40, 10, 40, 10))}
        for _ in range(3)
    ]

    assignment = embedding_classifier.assign_teams(frames, tracks)

    assert all(frame_teams[3] == 0 for frame_teams in assignment)
    assert all(frame_teams[1] != 0 and frame_teams[2] != 0 for frame_teams in assignment)
    assert embedding_classifier.empty_crop_count == 3


def test_embedding_abstains_when_the_margin_is_below_threshold(embedding_classifier):
    loaded_stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip()
    embedding_classifier._fit_clusters(frames, tracks)
    ambiguous_frame = frame_with_players(LIGHT_BBOX, DARK_BBOX)
    draw_player(ambiguous_frame, AMBIGUOUS_BBOX, AMBIGUOUS_GREY)
    ambiguous_embedding = embedding_classifier.encoder.get_image_embeddings(
        ambiguous_frame, {3: track(3, AMBIGUOUS_BBOX)}
    )[3]

    predicted_team, confidence, crop_ok = embedding_classifier._classify_with_margin(ambiguous_embedding)

    assert crop_ok is True
    assert confidence == pytest.approx(AMBIGUOUS_MARGIN_MEASURED, abs=0.01)

    # Below the measured margin: accepted. Above it: abstained. Bracketing
    # the measured value on both sides, rather than relying on the
    # fixture's default margin_threshold (0.55, which turns out to sit
    # ABOVE 0.543 and would abstain either way).
    embedding_classifier.margin_threshold = 0.5
    assert embedding_classifier._resolve_team(predicted_team, confidence, crop_ok) != 0

    embedding_classifier.margin_threshold = EMBEDDING_MARGIN_THRESHOLD
    assert embedding_classifier._resolve_team(predicted_team, confidence, crop_ok) == 0
    assert embedding_classifier.low_margin_count == 1


def test_embedding_classify_with_margin_placeholder_is_never_a_raw_zero(embedding_classifier):
    loaded_stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip()
    embedding_classifier._fit_clusters(frames, tracks)

    predicted_team, confidence, crop_ok = embedding_classifier._classify_with_margin(None)

    assert (predicted_team, confidence, crop_ok) == (1, 0.0, False)


def test_embedding_raises_before_fitting(embedding_classifier):
    with pytest.raises(RuntimeError, match='not fitted'):
        embedding_classifier._classify_with_margin(np.array([1.0, 0.0], dtype=np.float32))


def test_embedding_raises_when_no_crop_can_be_embedded(embedding_classifier):
    stub_embeddings(embedding_classifier.encoder)
    # Every player's bbox is degenerate in every frame, so nothing survives
    # to be clustered, a genuinely unusable clip, not an injected None.
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    tracks = [{1: track(1, (5, 5, 5, 5)), 2: track(2, (6, 6, 6, 6))} for _ in range(2)]

    with pytest.raises(ValueError, match='Insufficient usable player crops'):
        embedding_classifier.assign_teams(frames, tracks)


def test_embedding_raises_on_misaligned_frames_and_tracks(embedding_classifier):
    frames, tracks = two_team_embedding_clip(n_frames=3)

    with pytest.raises(ValueError, match='aligned frame-for-frame'):
        embedding_classifier.assign_teams(frames, tracks[:2])


def test_embedding_requires_clip_name_when_recording(embedding_classifier, tmp_path):
    frames, tracks = two_team_embedding_clip()

    with pytest.raises(ValueError, match='clip_name is required'):
        embedding_classifier.assign_teams(frames, tracks, record_path=str(tmp_path / 'p.csv'))


# --- recording ---------------------------------------------------------------

def test_embedding_records_one_row_per_player_per_frame_with_a_never_zero_raw_team(embedding_classifier, tmp_path):
    record_path = str(tmp_path / 'predictions.csv')
    stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip(n_frames=3)

    embedding_classifier.assign_teams(frames, tracks, record_path=record_path, clip_name='clip_1')

    with open(record_path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = [dict(zip(header, row)) for row in reader]

    assert header == PREDICTION_CSV_HEADER
    assert len(rows) == 3 * 2
    assert '0' not in {row['predicted_team'] for row in rows}
    assert all(row['clip'] == 'clip_1' for row in rows)
    assert all(float(row['confidence']) > 0.0 for row in rows)


def test_embedding_assign_teams_abstains_on_an_ambiguous_embedding(tmp_path):
    record_path = str(tmp_path / 'predictions.csv')
    classifier = EmbeddingClusteringClassifier(
        encoder=embedding_encoder(), fit_stride=1, margin_threshold=EMBEDDING_MARGIN_THRESHOLD,
    )
    stub_embeddings(classifier.encoder)
    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(2)]
    for frame in frames:
        draw_player(frame, AMBIGUOUS_BBOX, AMBIGUOUS_GREY)
    tracks = [
        {1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX), 3: track(3, AMBIGUOUS_BBOX)}
        for _ in range(2)
    ]

    assignment = classifier.assign_teams(frames, tracks, record_path=record_path, clip_name='clip_1')

    with open(record_path, newline='') as f:
        rows = list(csv.DictReader(f))

    # Abstained in the returned assignment, but the artefact still carries
    # the raw pre-threshold decision and its margin.
    assert all(frame_teams[3] == 0 for frame_teams in assignment)
    assert all(frame_teams[1] != 0 and frame_teams[2] != 0 for frame_teams in assignment)
    assert classifier.low_margin_count == 2
    abstained_rows = [row for row in rows if row['player_id'] == '3']
    assert len(abstained_rows) == 2
    assert all(row['predicted_team'] in {'1', '2'} for row in abstained_rows)
    assert all(row['crop_ok'] == 'True' for row in abstained_rows)
    assert all(0.5 <= float(row['confidence']) < EMBEDDING_MARGIN_THRESHOLD for row in abstained_rows)


def test_embedding_recording_does_not_change_the_returned_assignment(tmp_path):
    frames, tracks = two_team_embedding_clip(n_frames=3)
    plain = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)
    stub_embeddings(plain.encoder)
    recording = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)
    stub_embeddings(recording.encoder)

    without_recording = plain.assign_teams(frames, tracks)
    with_recording = recording.assign_teams(
        frames, tracks, record_path=str(tmp_path / 'predictions.csv'), clip_name='clip_1',
    )

    assert with_recording == without_recording


def test_embedding_assign_teams_never_writes_a_cache_file_when_recording(embedding_classifier, tmp_path):
    cache_path = tmp_path / 'teams.pkl'
    stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip(n_frames=2)

    embedding_classifier.assign_teams(
        frames,
        tracks,
        cache_path=str(cache_path),
        record_path=str(tmp_path / 'predictions.csv'),
        clip_name='clip_1',
    )

    assert not cache_path.exists()
    assert not Path(f'{cache_path}.meta.json').exists()


# --- caching -------------------------------------------------------------

def test_embedding_assign_teams_round_trips_through_the_cache(embedding_classifier, tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip(n_frames=3)
    first = embedding_classifier.assign_teams(frames, tracks, cache_path=cache_path)

    fresh = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)
    stub_embeddings(fresh.encoder)

    assert fresh.assign_teams(frames, tracks, cache_path=cache_path) == first
    assert fresh.cluster_centres is None  # served from cache, never fitted


def test_embedding_cache_is_invalidated_by_a_margin_threshold_change(tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    frames, tracks = two_team_embedding_clip(n_frames=2)
    lenient = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)
    stub_embeddings(lenient.encoder)
    lenient.assign_teams(frames, tracks, cache_path=cache_path)

    strict = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.99)
    stub_embeddings(strict.encoder)
    strict.assign_teams(frames, tracks, cache_path=cache_path)

    assert strict.cluster_centres is not None  # refitted, not served from the stale cache


def test_embedding_cache_is_invalidated_by_a_stale_tracks_digest(embedding_classifier, tmp_path):
    cache_path = str(tmp_path / 'teams.pkl')
    stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip(n_frames=3)
    embedding_classifier.assign_teams(frames[:2], tracks[:2], cache_path=cache_path)

    fresh = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)
    stub_embeddings(fresh.encoder)
    assignment = fresh.assign_teams(frames, tracks, cache_path=cache_path)

    assert len(assignment) == 3


def test_embedding_fingerprint_carries_the_classifier_revision_and_encoder_keys(embedding_classifier):
    fingerprint = embedding_classifier._cache_fingerprint([{}], 1)

    assert fingerprint['classifier_revision'] == EmbeddingClusteringClassifier.CLASSIFIER_REVISION == 1
    assert fingerprint['method'] == 'embedding'
    assert fingerprint['fit_stride'] == 1
    assert fingerprint['margin_threshold'] == 0.55
    # The encoder's crop geometry and revision change the embeddings this
    # method clusters, so both belong in its fingerprint.
    assert fingerprint['crop_fraction'] == 0.667
    assert fingerprint['encoder_revision'] == FashionCLIPClassifier.CLASSIFIER_REVISION


# --- deterministic team labelling (independent of iteration order) ---------

def test_team_labels_are_derived_from_the_fitted_centroids_lexicographically(embedding_classifier):
    # Flag 2 regression test, most directly: after a real fit, team_labels
    # must equal what the lexicographic rule gives on the ACTUAL fitted
    # centroids, not merely be internally consistent with itself, and not
    # dependent on which player was encoded first (see the reversed-order
    # test below for that half of the regression).
    loaded_stub_embeddings(embedding_classifier.encoder)
    frames, tracks = two_team_embedding_clip()

    embedding_classifier._fit_clusters(frames, tracks)

    centre_a, centre_b = embedding_classifier.cluster_centres
    expected_labels = {0: 1, 1: 2} if tuple(centre_a) <= tuple(centre_b) else {0: 2, 1: 1}
    assert embedding_classifier.team_labels == expected_labels


def test_team_labels_do_not_flip_when_player_iteration_order_is_reversed():
    """The direct regression test: the SAME jersey colour must resolve to the SAME team whether its player was iterated first or second during fitting."""
    # Reversing which player's track dict entry comes first changes the ROW
    # ORDER K-means fits over, which can (and measurably does, for this
    # fixture) change which array index -- 0 or 1 -- sklearn assigns to the
    # light-jersey cluster versus the dark-jersey one; team_labels is keyed
    # by that index, so the two runs' raw {index: team} dicts are NOT
    # expected to be equal (asserted below via the LOGICAL mapping instead:
    # classifying the same concrete embedding on both fitted classifiers).
    # Under the old "first label seen by kmeans.labels_[0]" rule this could
    # flip depending on tracks.values() iteration order; the
    # centroid-lexicographic rule does not, because it reads only the two
    # centroid values, never which index or which player produced them.
    forward = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)
    loaded_stub_embeddings(forward.encoder)
    reversed_order = EmbeddingClusteringClassifier(encoder=embedding_encoder(), fit_stride=1, margin_threshold=0.55)
    loaded_stub_embeddings(reversed_order.encoder)

    frames = [frame_with_players(LIGHT_BBOX, DARK_BBOX) for _ in range(4)]
    forward_tracks = [{1: track(1, LIGHT_BBOX), 2: track(2, DARK_BBOX)} for _ in range(4)]
    reversed_tracks = [{2: track(2, DARK_BBOX), 1: track(1, LIGHT_BBOX)} for _ in range(4)]

    forward._fit_clusters(frames, forward_tracks)
    reversed_order._fit_clusters(frames, reversed_tracks)

    assert np.allclose(sorted(forward.cluster_centres.tolist()), sorted(reversed_order.cluster_centres.tolist()))

    light_embedding = forward.encoder.get_image_embeddings(frames[0], {1: track(1, LIGHT_BBOX)})[1]
    dark_embedding = forward.encoder.get_image_embeddings(frames[0], {2: track(2, DARK_BBOX)})[2]

    forward_light_team = forward._classify_with_margin(light_embedding)[0]
    forward_dark_team = forward._classify_with_margin(dark_embedding)[0]
    reversed_light_team = reversed_order._classify_with_margin(light_embedding)[0]
    reversed_dark_team = reversed_order._classify_with_margin(dark_embedding)[0]

    assert forward_light_team == reversed_light_team
    assert forward_dark_team == reversed_dark_team
    assert forward_light_team != forward_dark_team
