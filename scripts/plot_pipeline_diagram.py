"""
plot_pipeline_diagram.py draws the pipeline's execution order as it actually runs,
not as the documented stage numbering reads. The three inference passes are
bracketed together at the left, and the remaining six stages run left to right in
the order main.py calls them, so the court keypoint stage appears inside the
opening group while carrying the number seven. Arrows are data dependencies,
never execution order: nothing in the figure connects two boxes merely because
one runs after the other.

The stage list below is transcribed from main.py rather than parsed out of it, so
the script verifies the transcription before it draws. Each stage declares the
call site it stands for, and the run aborts unless every call site is present
exactly once and their line numbers ascend in the drawn order. The five caching
stages declare their cache filenames, which are checked the same way. A diagram
that has drifted from the source it describes is worse than no diagram, so both
checks are fatal rather than warnings.

Three further checks cover the drawing rather than its content, because this
layout is laid out by hand at fixed coordinates and hand-set coordinates fail
silently: no text may fall below 6 pt, no text may be cropped by the figure edge,
and no stage name may overflow the box drawn around it. All three are fatal for
the same reason as the others.

Sized for the full width of an IEEE two-column A4 page at 18.0 cm, drawn wholly
in greys so it survives greyscale printing. Needs no caches, no checkpoints and
no GPU: it reads main.py and nothing else.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Selected before pyplot is imported. This script only ever writes files, and the
# environment the results were produced in is headless.
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
from matplotlib.patches import FancyArrow, Rectangle  # noqa: E402

MAIN_PATH = 'main.py'

OUTPUT_DIR = Path('data/outputs/figures')
OUTPUT_STEM = 'fig1_pipeline_execution'
# The PDF is the figure to include; the PNG exists for anything that cannot place
# a vector file. IEEE wants vector artwork, so the PDF is the deliverable.
OUTPUT_SUFFIXES = ('.pdf', '.png')


@dataclass(frozen=True)
class Stage:
    """One pipeline stage: its documented number, its call site in main.py, and its cache if it has one."""

    number: int
    name: str
    call_token: str
    cache: str | None


# In execution order, which is the point of the figure. The number column is the
# documented stage number from README.md and docs/pipeline_overview.md, so the
# court keypoint box reading 7 inside the opening group is the discrepancy being
# drawn. Line breaks are set by hand rather than wrapped: the six boxes in the
# row are 0.80 in wide, and the longest word in each name has to fit inside that
# once the stage number and the padding are taken out. verify_labels_fit() below
# is what holds this claim up.
STAGES = (
    Stage(1, 'Player detection\nand tracking', 'player_tracks = player_detector.run_tracking(', 'player_detections.pkl'),
    Stage(2, 'Ball detection', 'raw_ball_detections = ball_detector.run_detection(', 'ball_detections.pkl'),
    Stage(7, 'Court keypoints', 'keypoints_per_frame = court_keypoints.run_detection(', 'keypoints.pkl'),
    Stage(3, 'Ball gate and\ninterpolation', 'ball_input = possession_ball_input(raw_ball_detections)', None),
    Stage(4, 'Team\nclassification', 'team_assignment = team_classifier.assign_teams(', 'team_assignment_'),
    Stage(5, 'Possession', 'possession = possession_tracker.assign_possession(', 'possession.pkl'),
    Stage(6, 'Passes and\ninterceptions', 'passes = event_detector.identify_passes(', None),
    Stage(8, 'Homography\nand court\nmapping', 'court_positions, mapping_report = court_mapper.map_to_court(', None),
    Stage(9, 'Speed and\ndistance', 'distances, speeds, metrics_report = player_metrics.compute(', None),
)

# How many of the leading stages are the consecutive inference passes. Nothing but
# print statements separates them in main.py, which is what the bracket asserts.
INFERENCE_PASS_COUNT = 3

# Data dependencies as (producing stage number, consuming stage number, label).
# Raw frames are deliberately absent: they are the run's input rather than any
# stage's output, and drawing them would put an arrow into four boxes that says
# nothing about the ordering this figure exists to show.
DEPENDENCIES = (
    (1, 4, 'player_tracks'),
    (1, 5, 'player_tracks'),
    (1, 8, 'player_tracks'),
    (2, 3, 'raw ball'),
    (3, 5, 'ball_input.filled'),
    (4, 6, 'team_assignment'),
    (5, 6, 'possession'),
    (7, 8, 'keypoints_per_frame'),
    (8, 9, 'court_positions'),
)

# player_tracks feeds three stages, so it is drawn once as a horizontal trunk
# along the top with three branches dropping out of it, rather than as three
# near-parallel lines competing for the same band.
BUS_SOURCE = 1

# keypoints_per_frame runs the length of the figure underneath the row, from the
# bottom box of the opening group to stage 8. It gets the outermost lower lane to
# itself so it never shares a line with the short hops between adjacent stages.
UNDER_TRUNK_EDGE = (7, 8)

# The one dependency short enough to draw straight across, from the middle box of
# the opening group into the first box of the row, at the row's own height.
DIRECT_EDGE = (2, 3)

# Geometry, in inches, so the figure is laid out at its printed size and the point
# sizes below mean what they say.
FIGURE_WIDTH_IN = 18.0 / 2.54

# Tall enough for the three lines stage 8's name needs at 6.5 pt, which is what
# sets it: two lines would need a box 0.86 in wide, and six of those do not fit
# across 18 cm alongside the opening group.
BOX_HEIGHT_IN = 0.40
GROUP_BOX_LEFT_IN = 0.30
GROUP_BOX_WIDTH_IN = 0.98
# The opening group's boxes are spaced far enough apart that the dependency lanes
# can run between their heights to the right of the group, which is what keeps the
# figure to one band of boxes rather than a band plus a separate lane region.
GROUP_GAP_IN = 0.24

# Leaves a 0.38 in gap after the opening group, which is what the 'raw ball' label
# on the one straight-across dependency needs.
ROW_LEFT_IN = 1.66
ROW_BOX_WIDTH_IN = 0.82
RIGHT_MARGIN_IN = 0.035
TOP_MARGIN_IN = 0.13

# Below the boxes: the note explaining the stage 7 position, then the footnote.
NOTE_GAP_IN = 0.05
NOTE_HEIGHT_IN = 0.19
FOOTNOTE_GAP_IN = 0.05
FOOTNOTE_HEIGHT_IN = 0.22
BOTTOM_MARGIN_IN = 0.04

# Lanes, measured from the row's own height. One lane above carries the two
# forward hops, which do not overlap and so share it; two below carry the hops
# that do overlap. The pitch is set by the labels rather than the lines.
LANE_PITCH_IN = 0.14
ABOVE_LANE_ZERO_IN = 0.38
BELOW_LANE_ZERO_IN = 0.30

# How far from a box's centre a dependency enters or leaves it. Producers use one
# side and consumers the other, so a trunk branch dropping into a box never lands
# on the segment another edge is rising out of.
PORT_OFFSET_IN = 0.13

BRACKET_X_IN = 0.145
# The rotated bracket label is the leftmost thing in the figure. At 0.052 it sat
# 0.7 pt from the paper's edge, which survives the clipping check but leaves
# nothing for a trimmed margin.
BRACKET_LABEL_X_IN = 0.068
BRACKET_TICK_IN = 0.05

# Greys only. The cached fill has to read as distinctly shaded without darkening
# the text it sits behind, and the two fills must stay apart when a reader prints
# at 600 dpi in black and white.
CACHED_FILL = '0.86'
PLAIN_FILL = 'white'
EDGE_COLOUR = '0.15'
LINE_COLOUR = '0.35'
BOX_LINE_WIDTH = 0.7
HIGHLIGHT_LINE_WIDTH = 1.6
DEPENDENCY_LINE_WIDTH = 0.6

# No text in this figure is set below 6 pt, which is the floor most IEEE
# templates state for figure text. The figure is drawn at its final printed
# size, so these point sizes are the sizes that reach the page: nothing here is
# scaled down by the LaTeX \includegraphics that places it.
STAGE_FONT_SIZE = 6.5
NUMBER_FONT_SIZE = 6.5
LABEL_FONT_SIZE = 6.0
NOTE_FONT_SIZE = 6.0
LEGEND_FONT_SIZE = 6.0
MINIMUM_FONT_SIZE_PT = 6.0

# The rotated edge labels sit in the lanes the dependency elbows run through, and
# the trunk branches cross those lanes where they pass. An opaque mask behind
# each label breaks the line rather than the word: the elbow is still readable
# either side, and the label is never struck through.
LABEL_MASK = {'facecolor': 'white', 'edgecolor': 'none', 'pad': 0.8}

ARROW_HEAD_LENGTH_IN = 0.055
ARROW_HEAD_WIDTH_IN = 0.042

# Clearance a stage name must keep inside its own box, so a name that only just
# fits is reported as a failure rather than printed touching the border.
LABEL_PADDING_IN = 0.03

FIGURE_DPI = 300

FOOTNOTE_LINES = (
    'The number outside each box is its execution position; the number inside is the documented stage number.',
    "Raw frames feed stages 1, 2, 7 and 4 and are not drawn, being the run's input rather than any stage's output.",
)


def require_main_source() -> list[str]:
    """Return main.py's lines, raising a clear error when the file this figure describes is absent."""
    if not Path(MAIN_PATH).exists():
        raise FileNotFoundError(
            f'{MAIN_PATH} not found. This figure is a drawing of that file\'s execution order, '
            f'so it cannot be produced without it. main.py is committed, which makes an absent '
            f'one an incomplete checkout rather than a missing generated artefact.'
        )
    return Path(MAIN_PATH).read_text(encoding='utf-8').splitlines()


def call_site_lines(lines: list[str]) -> list[int]:
    """Return each stage's call-site line number, raising unless every one appears exactly once."""
    found = []
    for stage in STAGES:
        hits = [number for number, line in enumerate(lines, 1) if stage.call_token in line]
        if len(hits) != 1:
            raise ValueError(
                f'Stage {stage.number} declares the call site {stage.call_token!r}, which appears '
                f'{len(hits)} times in {MAIN_PATH} rather than once. The stage list in this script '
                f'no longer describes the pipeline it draws.'
            )
        found.append(hits[0])
    return found


def verify_execution_order(lines: list[str]) -> list[int]:
    """Confirm the drawn order matches main.py's call order and that the inference passes are consecutive."""
    found = call_site_lines(lines)

    out_of_order = [
        (STAGES[i].number, STAGES[i + 1].number)
        for i in range(len(found) - 1) if found[i] >= found[i + 1]
    ]
    if out_of_order:
        raise ValueError(
            f'{MAIN_PATH} calls these stage pairs in the opposite order to the one drawn: '
            f'{out_of_order}. The figure would assert an execution order the source contradicts.'
        )

    # Consecutive means no other drawn stage is called between them, which is what
    # the bracket in the figure claims. Print statements and comments in between
    # are not stages and do not count.
    block_end = found[INFERENCE_PASS_COUNT - 1]
    intervening = [
        stage.number for stage, line in zip(STAGES, found)
        if stage not in STAGES[:INFERENCE_PASS_COUNT] and found[0] < line < block_end
    ]
    if intervening:
        raise ValueError(
            f'Stage(s) {intervening} are called between the three inference passes, so they are not '
            f'consecutive and the bracket in the figure would be wrong.'
        )
    return found


def verify_caches(lines: list[str]) -> int:
    """Confirm every declared cache appears in main.py and that no undeclared cache_path exists, returning the count."""
    source = '\n'.join(lines)
    declared = [stage for stage in STAGES if stage.cache is not None]
    for stage in declared:
        if stage.cache not in source:
            raise ValueError(
                f'Stage {stage.number} is drawn as caching to {stage.cache!r}, which does not appear '
                f'in {MAIN_PATH}.'
            )

    cache_lines = sum(1 for line in lines if 'cache_path=' in line)
    if cache_lines != len(declared):
        raise ValueError(
            f'{MAIN_PATH} passes cache_path on {cache_lines} lines but {len(declared)} stages are '
            f'shaded as caching. The shading would misstate which stages cache.'
        )
    return len(declared)


def verify_font_sizes() -> float:
    """Confirm no text is set below the 6 pt floor, returning the smallest size used."""
    sizes = {
        'stage name': STAGE_FONT_SIZE,
        'stage number': NUMBER_FONT_SIZE,
        'edge label and footnote': LABEL_FONT_SIZE,
        'note': NOTE_FONT_SIZE,
        'legend': LEGEND_FONT_SIZE,
    }
    below = {name: size for name, size in sizes.items() if size < MINIMUM_FONT_SIZE_PT}
    if below:
        raise ValueError(
            f'These text elements are set below the {MINIMUM_FONT_SIZE_PT:.0f} pt floor: '
            f'{", ".join(f"{name} at {size} pt" for name, size in below.items())}. '
            f'The figure is placed at its drawn size, so a size set here is the size printed.'
        )
    return min(sizes.values())


def figure_height() -> float:
    """Return the figure's height in inches, driven by the three stacked boxes of the opening group."""
    group = INFERENCE_PASS_COUNT * BOX_HEIGHT_IN + (INFERENCE_PASS_COUNT - 1) * GROUP_GAP_IN
    below = NOTE_GAP_IN + NOTE_HEIGHT_IN + FOOTNOTE_GAP_IN + FOOTNOTE_HEIGHT_IN + BOTTOM_MARGIN_IN
    return TOP_MARGIN_IN + group + below


def box_rectangles() -> dict[int, tuple[float, float, float, float]]:
    """Return every stage's box as (left, centre height, width, height), keyed by documented stage number."""
    height = figure_height()
    boxes: dict[int, tuple[float, float, float, float]] = {}

    # The opening group, stacked downward from the top margin.
    centre = height - TOP_MARGIN_IN - BOX_HEIGHT_IN / 2.0
    for stage in STAGES[:INFERENCE_PASS_COUNT]:
        boxes[stage.number] = (GROUP_BOX_LEFT_IN, centre, GROUP_BOX_WIDTH_IN, BOX_HEIGHT_IN)
        centre -= BOX_HEIGHT_IN + GROUP_GAP_IN

    # The row, at the height of the group's middle box, so the one dependency
    # between the group and the row is a straight line rather than a step, and so
    # the gaps above and below the group's middle box become the lane bands.
    row_centre = boxes[STAGES[1].number][1]
    span = FIGURE_WIDTH_IN - RIGHT_MARGIN_IN - ROW_LEFT_IN
    row = STAGES[INFERENCE_PASS_COUNT:]
    gap = (span - len(row) * ROW_BOX_WIDTH_IN) / (len(row) - 1)
    for index, stage in enumerate(row):
        left = ROW_LEFT_IN + index * (ROW_BOX_WIDTH_IN + gap)
        boxes[stage.number] = (left, row_centre, ROW_BOX_WIDTH_IN, BOX_HEIGHT_IN)
    return boxes


def assign_lanes(spans: list[tuple[float, float]]) -> list[int]:
    """Return a lane per horizontal span, giving overlapping spans different lanes and reusing a lane once it is clear."""
    lane_last_end: list[float] = []
    assigned: dict[int, int] = {}
    # Longest first, so the widest spans take the outer lanes and shorter ones
    # nest inside them rather than crossing.
    order = sorted(range(len(spans)), key=lambda i: spans[i][0] - spans[i][1])
    for index in order:
        start, end = spans[index]
        for lane, last in enumerate(lane_last_end):
            if last < start:
                lane_last_end[lane] = end
                assigned[index] = lane
                break
        else:
            lane_last_end.append(end)
            assigned[index] = len(lane_last_end) - 1
    return [assigned[index] for index in range(len(spans))]


def draw_arrow_head(axes, x: float, y: float, dx: float, dy: float) -> None:
    """Draw a solid arrow head at (x, y) pointing along the unit direction (dx, dy)."""
    axes.add_patch(FancyArrow(
        x - dx * ARROW_HEAD_LENGTH_IN, y - dy * ARROW_HEAD_LENGTH_IN,
        dx * ARROW_HEAD_LENGTH_IN, dy * ARROW_HEAD_LENGTH_IN,
        width=0, head_width=ARROW_HEAD_WIDTH_IN, head_length=ARROW_HEAD_LENGTH_IN,
        length_includes_head=True, color=LINE_COLOUR, linewidth=0, zorder=3,
    ))


def draw_line(axes, xs: list[float], ys: list[float]) -> None:
    """Draw one dependency segment in the common line style."""
    axes.plot(xs, ys, color=LINE_COLOUR, linewidth=DEPENDENCY_LINE_WIDTH,
              solid_capstyle='round', zorder=1)


def draw_lane_label(axes, x: float, y: float, text: str, above: bool):
    """Place a lane's label clear of its own line, masked so crossing lines do not strike through it."""
    return axes.text(
        x, y + (0.055 if above else -0.055), text,
        fontsize=LABEL_FONT_SIZE, color=LINE_COLOUR, ha='center',
        va='bottom' if above else 'top', zorder=2, bbox=LABEL_MASK,
    )


def draw_boxes(axes, boxes: dict[int, tuple[float, float, float, float]]) -> list:
    """Draw every stage box with its shading, its documented number inside and its execution position outside."""
    texts = []
    for position, stage in enumerate(STAGES, 1):
        left, centre, width, height = boxes[stage.number]
        cached = stage.cache is not None
        # The court keypoint box carries a heavier border as well as its shading:
        # it is the one box whose position in the figure contradicts its number,
        # so it is marked in a way that survives greyscale and a photocopier.
        highlight = stage.number == 7
        axes.add_patch(Rectangle(
            (left, centre - height / 2.0), width, height,
            facecolor=CACHED_FILL if cached else PLAIN_FILL,
            edgecolor=EDGE_COLOUR,
            linewidth=HIGHLIGHT_LINE_WIDTH if highlight else BOX_LINE_WIDTH,
            zorder=2,
        ))
        # Execution position, outside the box, above its left corner. Set clear of
        # the box's centre line so the trunk branches, which drop in at the
        # centre, never run into it.
        texts.append(('execution position', axes.text(
            left + 0.02, centre + height / 2.0 + 0.018, str(position),
            fontsize=NUMBER_FONT_SIZE, ha='left', va='bottom', color=LINE_COLOUR, zorder=3,
        )))
        # Documented stage number, inside the box at its left edge.
        texts.append(('stage number', axes.text(
            left + 0.045, centre, str(stage.number),
            fontsize=NUMBER_FONT_SIZE, ha='left', va='center', color=EDGE_COLOUR, zorder=3,
        )))
        texts.append((f'stage {stage.number} name', axes.text(
            text_centre(left, width), centre, stage.name,
            fontsize=STAGE_FONT_SIZE, ha='center', va='center', color='black', zorder=3,
            linespacing=1.15,
        )))
    return texts


# The strip at a box's left that its documented stage number occupies. The name is
# centred in what remains, so the clearance it keeps is equal on both sides rather
# than being tight against the number and loose against the right border.
NUMBER_COLUMN_IN = 0.10


def text_centre(left: float, width: float) -> float:
    """Return the x a stage name is centred on, which is the box less the column its number occupies."""
    return left + NUMBER_COLUMN_IN + (width - NUMBER_COLUMN_IN) / 2.0


def draw_dependencies(axes, boxes: dict[int, tuple[float, float, float, float]]) -> list:
    """Draw every data dependency, returning the label text objects."""
    texts = []
    row_centre = boxes[STAGES[1].number][1]

    def box_edges(number: int) -> tuple[float, float, float, float]:
        left, centre, width, height = boxes[number]
        return left, left + width, centre - height / 2.0, centre + height / 2.0

    # The straight hop from the opening group into the first box of the row.
    producer, consumer, label = next(
        edge for edge in DEPENDENCIES if (edge[0], edge[1]) == DIRECT_EDGE
    )
    _, from_right, _, _ = box_edges(producer)
    to_left, _, _, _ = box_edges(consumer)
    draw_line(axes, [from_right, to_left], [row_centre, row_centre])
    draw_arrow_head(axes, to_left, row_centre, 1, 0)
    texts.append((f'{label} label', draw_lane_label(
        axes, (from_right + to_left) / 2.0, row_centre, label, above=True)))

    # The player_tracks trunk: one horizontal line out of the top box of the
    # group, with a branch dropping into each of the three stages that consume it.
    consumers = [edge[1] for edge in DEPENDENCIES if edge[0] == BUS_SOURCE]
    _, trunk_left, _, _ = box_edges(BUS_SOURCE)
    trunk_y = boxes[BUS_SOURCE][1]
    drops = [boxes[number][0] + boxes[number][2] / 2.0 - PORT_OFFSET_IN for number in consumers]
    draw_line(axes, [trunk_left, max(drops)], [trunk_y, trunk_y])
    for number, drop_x in zip(consumers, drops):
        _, _, _, box_top = box_edges(number)
        draw_line(axes, [drop_x, drop_x], [trunk_y, box_top])
        draw_arrow_head(axes, drop_x, box_top, 0, -1)
    texts.append(('player_tracks label', axes.text(
        trunk_left + 0.06, trunk_y + 0.055, 'player_tracks',
        fontsize=LABEL_FONT_SIZE, color=LINE_COLOUR, ha='left', va='bottom',
        zorder=2, bbox=LABEL_MASK,
    )))

    # keypoints_per_frame, along the bottom from the group's last box to stage 8.
    producer, consumer = UNDER_TRUNK_EDGE
    label = next(edge[2] for edge in DEPENDENCIES if (edge[0], edge[1]) == UNDER_TRUNK_EDGE)
    _, from_right, _, _ = box_edges(producer)
    under_y = boxes[producer][1]
    rise_x = boxes[consumer][0] + boxes[consumer][2] / 2.0 - PORT_OFFSET_IN
    _, _, box_bottom, _ = box_edges(consumer)
    draw_line(axes, [from_right, rise_x], [under_y, under_y])
    draw_line(axes, [rise_x, rise_x], [under_y, box_bottom])
    draw_arrow_head(axes, rise_x, box_bottom, 0, 1)
    texts.append((f'{label} label', axes.text(
        from_right + 0.06, under_y - 0.055, label,
        fontsize=LABEL_FONT_SIZE, color=LINE_COLOUR, ha='left', va='top',
        zorder=2, bbox=LABEL_MASK,
    )))

    # Everything else is a hop between two boxes of the row, routed through a lane
    # above or below it. Forward hops that skip no box would still need a lane,
    # because the gap between boxes is too narrow to carry a label.
    handled = {DIRECT_EDGE, UNDER_TRUNK_EDGE}
    hops = [
        edge for edge in DEPENDENCIES
        if edge[0] != BUS_SOURCE and (edge[0], edge[1]) not in handled
    ]
    above_hops = [edge for edge in hops if edge[0] in (4, 8)]
    below_hops = [edge for edge in hops if edge not in above_hops]

    for hops_set, above in ((above_hops, True), (below_hops, False)):
        spans = []
        for producer, consumer, _ in hops_set:
            start = boxes[producer][0] + boxes[producer][2] / 2.0 + PORT_OFFSET_IN
            end = boxes[consumer][0] + boxes[consumer][2] / 2.0 - PORT_OFFSET_IN
            spans.append((start, end))
        for (producer, consumer, label), (start, end), lane in zip(
                hops_set, spans, assign_lanes(spans)):
            zero = ABOVE_LANE_ZERO_IN if above else -BELOW_LANE_ZERO_IN
            lane_y = row_centre + zero + (lane * LANE_PITCH_IN * (1 if above else -1))
            _, _, from_bottom, from_top = box_edges(producer)
            _, _, to_bottom, to_top = box_edges(consumer)
            exit_y = from_top if above else from_bottom
            enter_y = to_top if above else to_bottom
            draw_line(axes, [start, start], [exit_y, lane_y])
            draw_line(axes, [start, end], [lane_y, lane_y])
            draw_line(axes, [end, end], [lane_y, enter_y])
            draw_arrow_head(axes, end, enter_y, 0, -1 if above else 1)
            texts.append((f'{label} label', draw_lane_label(
                axes, (start + end) / 2.0, lane_y, label, above=above)))
    return texts


def draw_bracket(axes, boxes: dict[int, tuple[float, float, float, float]]) -> list:
    """Draw the bracket grouping the three inference passes, with its rotated label."""
    numbers = [stage.number for stage in STAGES[:INFERENCE_PASS_COUNT]]
    top = boxes[numbers[0]][1] + BOX_HEIGHT_IN / 2.0
    bottom = boxes[numbers[-1]][1] - BOX_HEIGHT_IN / 2.0
    draw_line(axes, [BRACKET_X_IN, BRACKET_X_IN], [bottom, top])
    for y in (top, bottom):
        draw_line(axes, [BRACKET_X_IN, BRACKET_X_IN + BRACKET_TICK_IN], [y, y])
    return [('bracket label', axes.text(
        BRACKET_LABEL_X_IN, (top + bottom) / 2.0, 'three inference passes, consecutive',
        fontsize=LABEL_FONT_SIZE, color=EDGE_COLOUR, rotation=90,
        ha='center', va='center', rotation_mode='anchor',
    ))]


def draw_note_and_legend(axes, boxes: dict[int, tuple[float, float, float, float]]) -> list:
    """Draw the stage 7 note beneath the opening group, the key beside it, and the footnote below both."""
    texts = []
    group_bottom = boxes[STAGES[INFERENCE_PASS_COUNT - 1].number][1] - BOX_HEIGHT_IN / 2.0
    note_top = group_bottom - NOTE_GAP_IN

    texts.append(('stage 7 note', axes.text(
        GROUP_BOX_LEFT_IN + GROUP_BOX_WIDTH_IN / 2.0, note_top,
        'numbered stage 7,\nexecutes third',
        fontsize=NOTE_FONT_SIZE, ha='center', va='top', color=EDGE_COLOUR,
        style='italic', linespacing=1.25,
    )))

    # The key, on the same band as the note and to the right of it, where the row
    # of boxes has ended and the space is otherwise empty.
    swatch = 0.10
    y = note_top - NOTE_HEIGHT_IN / 2.0
    x = ROW_LEFT_IN
    for fill, caption in ((CACHED_FILL, 'caches its output'), (PLAIN_FILL, 'recomputed each run')):
        axes.add_patch(Rectangle(
            (x, y - swatch / 2.0), swatch, swatch,
            facecolor=fill, edgecolor=EDGE_COLOUR, linewidth=BOX_LINE_WIDTH, zorder=2,
        ))
        texts.append((f'key {caption!r}', axes.text(
            x + swatch + 0.035, y, caption,
            fontsize=LEGEND_FONT_SIZE, ha='left', va='center', color=EDGE_COLOUR,
        )))
        x += swatch + 0.035 + text_width(caption, LEGEND_FONT_SIZE) + 0.16

    draw_line(axes, [x, x + 0.13], [y, y])
    draw_arrow_head(axes, x + 0.13, y, 1, 0)
    texts.append(('key arrow caption', axes.text(
        x + 0.13 + 0.035, y, 'data dependency, not execution order',
        fontsize=LEGEND_FONT_SIZE, ha='left', va='center', color=EDGE_COLOUR,
    )))

    footnote_top = note_top - NOTE_HEIGHT_IN - FOOTNOTE_GAP_IN
    texts.append(('footnote', axes.text(
        GROUP_BOX_LEFT_IN, footnote_top, '\n'.join(FOOTNOTE_LINES),
        fontsize=LABEL_FONT_SIZE, ha='left', va='top', color=EDGE_COLOUR, linespacing=1.3,
    )))
    return texts


_WIDTH_CACHE: dict[tuple[str, float], float] = {}


def text_width(text: str, size: float) -> float:
    """Return a string's rendered width in inches, measured rather than estimated."""
    key = (text, size)
    if key not in _WIDTH_CACHE:
        figure = plt.figure(figsize=(1, 1), dpi=FIGURE_DPI)
        artist = figure.text(0.5, 0.5, text, fontsize=size)
        figure.canvas.draw()
        width = artist.get_window_extent(renderer=figure.canvas.get_renderer()).width / FIGURE_DPI
        plt.close(figure)
        _WIDTH_CACHE[key] = width
    return _WIDTH_CACHE[key]


def verify_labels_fit(figure, axes, boxes: dict[int, tuple[float, float, float, float]],
                      name_texts: dict[int, object]) -> float:
    """Confirm every stage name fits inside its own box, returning the tightest clearance in inches."""
    # The line breaks in STAGES are set by hand, and a name one word too long for
    # its box would otherwise be drawn straddling the border rather than reported.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    inverse = axes.transData.inverted()

    tightest = float('inf')
    overflowing = []
    for stage in STAGES:
        left, centre, width, height = boxes[stage.number]
        box = name_texts[stage.number].get_window_extent(renderer=renderer)
        (x0, y0) = inverse.transform((box.x0, box.y0))
        (x1, y1) = inverse.transform((box.x1, box.y1))
        # Measured against the column the name actually has, which is the box less
        # the width its stage number occupies on the left.
        available_left = left + NUMBER_COLUMN_IN
        clearance = min(
            x0 - available_left, (left + width) - x1,
            y0 - (centre - height / 2.0), (centre + height / 2.0) - y1,
        )
        tightest = min(tightest, clearance)
        if clearance < LABEL_PADDING_IN:
            overflowing.append(
                f'stage {stage.number} ({stage.name.replace(chr(10), " / ")!r}) clears its box by '
                f'only {clearance:.3f} in'
            )
    if overflowing:
        raise ValueError(
            'These stage names do not fit the box drawn around them: ' + '; '.join(overflowing) +
            f'. Either break the name across more lines or widen ROW_BOX_WIDTH_IN, which cannot '
            f'be done without narrowing something else across the {FIGURE_WIDTH_IN * 2.54:.1f} cm.'
        )
    return tightest


def verify_nothing_clipped(figure, texts: list) -> float:
    """Confirm every text element falls inside the figure, returning the smallest margin to an edge in points."""
    # The layout is set by hand at fixed coordinates rather than by tight_layout,
    # which is what keeps the figure at precisely 18.0 cm. The cost is that too
    # small a margin crops a label at the paper's edge instead of resizing to fit,
    # so every text element's rendered box is checked against the canvas.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()

    smallest = float('inf')
    clipped = []
    for name, artist in texts:
        box = artist.get_window_extent(renderer=renderer)
        margin = min(box.x0, box.y0, width - box.x1, height - box.y1)
        smallest = min(smallest, margin)
        if margin < 0:
            clipped.append(f'{name} overruns the figure edge by {abs(margin) / FIGURE_DPI * 72:.1f} pt')
    if clipped:
        raise ValueError(
            'These elements are cut off by the figure edge: ' + '; '.join(clipped) +
            '. Widen the corresponding margin.'
        )
    return smallest / FIGURE_DPI * 72


def draw(boxes: dict[int, tuple[float, float, float, float]], height: float) -> list[Path]:
    """Draw the whole figure and write it, returning the paths written."""
    figure = plt.figure(figsize=(FIGURE_WIDTH_IN, height), dpi=FIGURE_DPI)
    axes = figure.add_axes([0, 0, 1, 1])
    axes.set_xlim(0, FIGURE_WIDTH_IN)
    axes.set_ylim(0, height)
    axes.axis('off')

    texts = draw_boxes(axes, boxes)
    name_texts = {
        stage.number: artist for (name, artist), stage in zip(
            [t for t in texts if t[0].endswith('name')], STAGES)
    }
    texts += draw_dependencies(axes, boxes)
    texts += draw_bracket(axes, boxes)
    texts += draw_note_and_legend(axes, boxes)

    tightest = verify_labels_fit(figure, axes, boxes, name_texts)
    print(f'Every stage name fits its box, the tightest by {tightest:.3f} in.')
    margin = verify_nothing_clipped(figure, texts)
    print(f'Every text element is inside the figure, the tightest by {margin:.1f} pt.')

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in OUTPUT_SUFFIXES:
        path = OUTPUT_DIR / f'{OUTPUT_STEM}{suffix}'
        figure.savefig(path, dpi=FIGURE_DPI)
        written.append(path)
    plt.close(figure)
    return written


def main() -> int:
    """Verify the stage list against main.py, draw the figure and report what was written."""
    lines = require_main_source()

    call_lines = verify_execution_order(lines)
    print(f'{MAIN_PATH} call order verified for {len(STAGES)} stages:')
    for stage, line in zip(STAGES, call_lines):
        marker = ' [cached]' if stage.cache else ''
        print(f'  line {line:>4}  stage {stage.number}{marker}')

    cached = verify_caches(lines)
    print(f'{cached} stages cache, matching the {cached} cache_path arguments in {MAIN_PATH}.')

    smallest = verify_font_sizes()
    print(f'Smallest text is {smallest:.1f} pt, at or above the {MINIMUM_FONT_SIZE_PT:.0f} pt floor.')

    boxes = box_rectangles()
    height = figure_height()
    print(f'Figure {FIGURE_WIDTH_IN * 2.54:.1f} cm wide by {height * 2.54:.1f} cm tall, spanning both columns.')

    print()
    for path in draw(boxes, height):
        print(f'Wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
