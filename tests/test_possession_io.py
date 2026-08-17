"""Unit tests for possession's shared correct-input helper: the adopted gate before interpolation, in main.py's order."""
from __future__ import annotations

from pathlib import Path

import basketball.possession.possession_io as possession_io
from basketball.detection.ball_detector import BallDetection
from basketball.detection.ball_interpolation import BallInterpolator
from basketball.possession.possession_io import BallInput, possession_ball_input


def ball_frame(x1: float, y1: float, confidence: float = 0.9) -> dict[int, BallDetection]:
    """One frame carrying a single ball detection, keyed 1 like the detector's own output."""
    return {1: BallDetection(bbox=[x1, y1, x1 + 10, y1 + 10], confidence=confidence)}


def straight_line_detections(n_frames: int, step: float = 5.0) -> list[dict[int, BallDetection]]:
    """A physically plausible left-to-right ball track the gate keeps in full, with no missing frames."""
    return [ball_frame(10 + step * i, 10 + step * i) for i in range(n_frames)]


def test_possession_ball_input_calls_the_gate_before_fill_missing(monkeypatch) -> None:
    # The whole point of the helper: every possession script written before it
    # called fill_missing(raw) and skipped the gate entirely, so the ORDER is
    # the behaviour under test, not an implementation detail.
    calls: list[str] = []
    real_gate = possession_io.BallInterpolator.filter_detections_global_trajectory
    real_fill = possession_io.BallInterpolator.fill_missing

    def spy_gate(
        self: BallInterpolator,
        ball_detections: list[dict[int, BallDetection]],
        *args: object,
        **kwargs: object,
    ) -> list[dict[int, BallDetection]]:
        calls.append('gate')
        return real_gate(self, ball_detections, *args, **kwargs)

    def spy_fill(
        self: BallInterpolator,
        ball_detections: list[dict[int, BallDetection]],
    ) -> list[dict[int, BallDetection]]:
        calls.append('fill')
        return real_fill(self, ball_detections)

    monkeypatch.setattr(
        possession_io.BallInterpolator, 'filter_detections_global_trajectory', spy_gate,
    )
    monkeypatch.setattr(possession_io.BallInterpolator, 'fill_missing', spy_fill)

    possession_ball_input(straight_line_detections(6))

    assert calls == ['gate', 'fill']


def test_possession_ball_input_fills_every_frame_but_leaves_gated_sparse() -> None:
    # The asymmetry IS the point: the labelling tool consumes gated (real
    # detections only, so it can never show a fabricated ball) while the
    # tracker consumes filled (a position on every frame).
    raw = straight_line_detections(6)
    raw[2] = {}  # a genuine detector gap in the middle of the track
    raw[3] = {}

    ball_input = possession_ball_input(raw)

    assert isinstance(ball_input, BallInput)
    assert all(1 in frame for frame in ball_input.filled)
    assert not all(1 in frame for frame in ball_input.gated)
    assert len(ball_input.gated) == len(ball_input.filled) == len(raw)


def test_possession_ball_input_gate_removes_a_teleporting_detection() -> None:
    # A detection far outside the plausible-speed envelope is exactly what the
    # adopted gate exists to drop, and what fill_missing(raw) alone would have
    # interpolated straight through.
    raw = straight_line_detections(8)
    raw[4] = ball_frame(4000, 4000)

    ball_input = possession_ball_input(raw)

    assert 1 not in ball_input.gated[4]
    # ...and interpolation then bridges that frame from its real neighbours,
    # rather than anchoring on the false position.
    assert 1 in ball_input.filled[4]
    assert ball_input.filled[4][1].bbox[0] < 1000


def test_possession_ball_input_does_not_mutate_the_raw_detections() -> None:
    raw = straight_line_detections(6)
    raw[3] = {}
    before = [dict(frame) for frame in raw]

    possession_ball_input(raw)

    assert raw == before


def test_generate_possession_script_uses_the_shared_helper_under_a_main_guard() -> None:
    # Guards both small fixes at once: the script must no longer be named
    # test_possession.py (pytest.ini is the only reason it was never
    # collected and executed), must not run its body at import, and must
    # build its ball input through the helper rather than fill_missing(raw).
    source = Path('scripts/generate_possession.py').read_text(encoding='utf-8')
    # Comments are stripped before the absence check: the script's own comment
    # names fill_missing(raw) to explain what it no longer does, and a
    # substring match would read that explanation as the call itself.
    code = '\n'.join(line for line in source.splitlines() if not line.strip().startswith('#'))

    assert not Path('scripts/test_possession.py').exists()
    assert 'def main() -> None:' in source
    assert "if __name__ == '__main__':" in source
    assert 'possession_ball_input(' in code
    assert 'fill_missing(' not in code
