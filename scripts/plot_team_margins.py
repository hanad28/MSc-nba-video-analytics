"""
plot_team_margins.py renders the team classifier's margin distributions from the
nine committed margin CSVs, so the figure the dissertation prints is generated
from the recorded evidence rather than redrawn by hand. Each arm is plotted as
an empirical cumulative distribution: every one of the recorded confidences
appears in its curve exactly where it was measured, with no bin width and no
kernel bandwidth to defend.

The figure's whole claim is that the three arms were measured under one common
setting, so the script verifies that before it draws anything. It checks that
the three arms cover an identical set of crops, and that each arm's recorded
confidences and team labels reproduce its crop 0.667 file in the committed
inference grid. Both checks are fatal: a figure whose caption claims a
controlled comparison must not be produced from files that no longer support
the claim.

Like measure_camera_motion.py this script needs no caches, no checkpoints and no
GPU. Everything it reads is committed, so it runs from a fresh clone.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

# Selected before pyplot is imported. This script only ever writes files, and
# the environment the results were produced in is headless, so asking for an
# interactive backend would fail there for no gain.
matplotlib.use('Agg')

import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

CLIPS = ('clip_1', 'clip_2', 'clip_3')
CONFIG_PATH = 'config/default.yaml'
MARGIN_CSV_TEMPLATE = 'results/team_classification/margin_measurement/{arm}_{clip}.csv'
GRID_CSV_TEMPLATE = 'results/team_classification/inference_grid/{stem}_{clip}.csv'

# The margin CSVs record a confidence per crop and nothing about the settings
# they were measured under, so provenance is established by matching them
# against the inference grid, whose filenames carry the prompt variant and the
# crop fraction. All three arms match their crop 0.667 file exactly, which is
# what makes this a controlled comparison: the crop is held constant and only
# the decision mechanism varies. Note that this is NOT the deployed setting for
# every arm (config deploys FashionCLIP at crop 1.0 and embedding at 0.5), and
# the FashionCLIP arm shown here is the domain prompt variant rather than the
# adopted NBA one, so the figure understates the deployed classifier.
GRID_STEMS = {
    'fashionclip': 'fashionclip_B_domain_crop0.667',
    'kmeans': 'kmeans_crop0.667',
    'embedding': 'embedding_crop0.667',
}

# The crop identity above is the figure's caption in machine-readable form.
COMMON_CROP_FRACTION = 0.667

# How far a recorded confidence may sit from its grid counterpart and still count
# as the same measurement. The classifiers compute in float32 and the two CSVs
# were written from separately widened copies, so identical measurements differ
# in the eighth significant digit: the largest disagreement across the six
# affected files is 1.8e-07, under two float32 epsilons. The nearest alternative
# setting is nothing like that close. Swapping the crop fraction moves the
# median confidence by 0.02 to 0.05 and individual crops by up to 0.44, so this
# tolerance separates arithmetic noise from a settings difference by four orders
# of magnitude rather than splitting a fine distinction.
MAX_PROVENANCE_DIFFERENCE = 1e-6

# Every margin CSV carries these columns; a file missing one is a different
# artefact from the one this script was written against.
REQUIRED_COLUMNS = ('clip', 'frame_idx', 'player_id', 'predicted_team', 'confidence', 'crop_ok')

# The key each arm's deployed abstention floor is read from. Read from config
# rather than written in here so a re-tuned threshold cannot silently leave this
# script reporting the old one.
ABSTENTION_THRESHOLD_KEYS = {
    'fashionclip': 'fashionclip_confidence_threshold',
    'kmeans': 'kmeans_margin_threshold',
    'embedding': 'embedding_margin_threshold',
}

# A confidence equal to another arm's is not the same quantity: each is that
# arm's own d_other/(d_near + d_other) margin, and none of the three is a
# calibrated probability. What the figure compares is how decisively each method
# separates two teams on identical crops, not how well any of them is calibrated.
X_AXIS_LABEL = 'Decision margin (each arm on its own scale)'
Y_AXIS_LABEL = 'Cumulative fraction of crops'

# 0.5 is an exact tie in all three arms' margin form, so no curve can begin
# below it and the axis starts there rather than at 0.
X_AXIS_LIMITS = (0.5, 1.0)


@dataclass(frozen=True)
class ArmStyle:
    """One arm's display label and line style."""

    label: str
    colour: str
    linestyle: tuple
    linewidth: float


# The figure is drawn in greyscale rather than in colour that degrades to
# greyscale. An earlier revision used a blue and a dark red whose luminances
# were 67 and 61 of 255, six levels apart: indistinguishable the moment the
# dissertation is printed or photocopied in black and white, leaving the dash
# pattern doing all the work while appearing to be backed up by colour. Here
# three things separate the arms independently, and any one of them suffices:
# a grey level (0, 58, 112 of 255), a dash pattern, and a line weight. Nothing
# is lost by dropping hue, because hue was carrying nothing.
ARM_STYLES = {
    'fashionclip': ArmStyle('FashionCLIP (prompted)', '#000000', (0, ()), 1.9),
    'kmeans': ArmStyle('K-means (colour baseline)', '#707070', (0, (5.5, 2.2)), 1.5),
    'embedding': ArmStyle('CLIP embeddings (unprompted)', '#3a3a3a', (0, (1.1, 1.9)), 1.2),
}
ARMS = tuple(ARM_STYLES)

# Full width of an IEEE two-column A4 page, so the figure spans both columns.
# The previous 16.26 cm was the matplotlib default aspect left unchanged, which
# fitted nothing in particular: dropped into a single 8.8 cm column it would
# have been scaled to 54%, taking the annotation down to 4.3 pt.
FIGURE_WIDTH_CM = 18.0
FIGURE_HEIGHT_CM = 7.0
CM_PER_IN = 2.54
FIGURE_SIZE_IN = (FIGURE_WIDTH_CM / CM_PER_IN, FIGURE_HEIGHT_CM / CM_PER_IN)
FIGURE_DPI = 300

# The figure is placed at its drawn size, so these are the sizes that reach the
# page. 6 pt is the floor most IEEE templates state for figure text.
MINIMUM_FONT_SIZE_PT = 6.0
TITLE_FONT_SIZE = 8.5
AXIS_LABEL_FONT_SIZE = 8.0
TICK_FONT_SIZE = 7.5
LEGEND_FONT_SIZE = 8.0
ANNOTATION_FONT_SIZE = 7.0

# Margins are set explicitly rather than by tight_layout, which would leave the
# legend drawn outside the axes unaccounted for, and rather than by
# bbox_inches='tight' on save, which would silently change the figure's final
# width away from the 18.0 cm this is specified at.
# Wide enough on the left for the rotated y-axis label and its tick labels, and
# inset enough on the right for the half of the '1.0' tick label that overhangs
# the axis. At 0.052 and 0.995 both were cut off by the figure edge.
AXES_LEFT = 0.076
AXES_RIGHT = 0.982
AXES_TOP = 0.895
AXES_BOTTOM = 0.265

# The legend sits below the axes, spread along one row. Three entries across an
# 18 cm figure fit comfortably, and outside the axes it cannot collide with a
# curve or with the annotation however the measurement moves.
LEGEND_Y = 0.012
LEGEND_COLUMNS = 3

# The annotated margin is placed to the right of its own line, in the band above
# the K-means curve and below the embedding arm's flat top, which is the one
# region of this plot no curve enters.
ANNOTATION_X_OFFSET = 0.007
ANNOTATION_Y = 0.86

# Written as both a raster and a vector file: the PDF is what a LaTeX build
# should include, the PNG is for anything that cannot place one.
OUTPUT_DIR = Path('data/outputs/figures')
OUTPUT_STEM = 'fig2_team_margins'
OUTPUT_SUFFIXES = ('.pdf', '.png')


def require_file(path: str, what: str, hint: str) -> str:
    """Return path, raising a message naming what is missing and where it comes from."""
    if not Path(path).exists():
        raise FileNotFoundError(f'{what} not found at {path}. {hint}')
    return path


def load_config(path: str) -> dict:
    """Load and return the YAML pipeline configuration from disk."""
    with open(path, 'r') as handle:
        return yaml.safe_load(handle)


def load_arm(arm: str) -> pd.DataFrame:
    """Return one arm's recorded confidences across all three clips, in clip order."""
    frames = []
    for clip in CLIPS:
        path = require_file(
            MARGIN_CSV_TEMPLATE.format(arm=arm, clip=clip),
            f"{arm}'s recorded margins for {clip}",
            'The margin CSVs are committed under results/team_classification/margin_measurement/, '
            'so an absent one means an incomplete checkout rather than a missing generated artefact.',
        )
        frame = pd.read_csv(path)
        missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f'{path} is missing the column(s) {missing}, so it is not the artefact this script reads.')
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def crop_keys(frame: pd.DataFrame) -> set[tuple[str, int, int]]:
    """Return the set of crops one arm was measured on."""
    return set(zip(frame['clip'], frame['frame_idx'], frame['player_id']))


def verify_common_crops(arms: dict[str, pd.DataFrame]) -> int:
    """Confirm every arm was measured on an identical set of crops, returning that count."""
    key_sets = {arm: crop_keys(frame) for arm, frame in arms.items()}
    reference_arm = ARMS[0]
    reference = key_sets[reference_arm]
    for arm, keys in key_sets.items():
        if keys != reference:
            raise ValueError(
                f'{arm} was measured on a different set of crops from {reference_arm} '
                f'({len(keys)} against {len(reference)}, {len(keys ^ reference)} not shared). '
                f'The figure compares decision mechanisms on identical crops, which these files no longer support.'
            )
    return len(reference)


def verify_crop_provenance(arm: str, frame: pd.DataFrame) -> float:
    """Confirm one arm's confidences and labels reproduce its crop 0.667 grid files, returning the largest difference found."""
    largest_difference = 0.0
    for clip in CLIPS:
        path = require_file(
            GRID_CSV_TEMPLATE.format(stem=GRID_STEMS[arm], clip=clip),
            f"the inference grid file {arm}'s margins are checked against for {clip}",
            'The grid is committed under results/team_classification/inference_grid/, '
            'so an absent file means an incomplete checkout.',
        )
        grid = pd.read_csv(path)
        merged = frame[frame['clip'] == clip].merge(
            grid, on=['frame_idx', 'player_id'], suffixes=('_margin', '_grid'),
        )
        if len(merged) != int((frame['clip'] == clip).sum()):
            raise ValueError(
                f'{arm} on {clip} does not align crop for crop with {path} '
                f'({len(merged)} of {int((frame["clip"] == clip).sum())} rows matched).'
            )
        difference = float(np.abs(merged['confidence_margin'] - merged['confidence_grid']).max())
        largest_difference = max(largest_difference, difference)

        # Labels are compared exactly, with no tolerance to hide behind: a
        # confidence may land an epsilon away and still be the same
        # measurement, but a crop assigned to a different team is a different
        # measurement whatever the margin did.
        disagreements = int((merged['predicted_team_margin'] != merged['predicted_team_grid']).sum())
        if disagreements:
            raise ValueError(
                f'{arm} on {clip} assigns a different team from {GRID_STEMS[arm]} on {disagreements} crop(s), '
                f'so the two files are not the same measurement.'
            )
    if largest_difference > MAX_PROVENANCE_DIFFERENCE:
        raise ValueError(
            f"{arm}'s recorded margins differ from {GRID_STEMS[arm]} by up to {largest_difference:.3e}, "
            f'above the {MAX_PROVENANCE_DIFFERENCE:.0e} tolerance for float32 noise. They were not measured at '
            f'crop {COMMON_CROP_FRACTION}, and the figure would misstate its own settings.'
        )
    return largest_difference


def ecdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the empirical cumulative distribution of a sample as (sorted values, cumulative fraction)."""
    ordered = np.sort(values)
    return ordered, np.arange(1, len(ordered) + 1) / len(ordered)


def print_table(title: str, headers: list[str], rows: list[list[str]]) -> None:
    """Print one aligned results table."""
    widths = (
        [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)] if rows
        else [len(header) for header in headers]
    )
    print(f'\n{title}')
    print('  '.join(header.ljust(width) for header, width in zip(headers, widths)))
    for row in rows:
        print('  '.join(str(cell).ljust(width) for cell, width in zip(row, widths)))


def summarise(arms: dict[str, pd.DataFrame], thresholds: dict[str, float]) -> None:
    """Print the distribution summary and the per-clip medians the caption is written from."""
    print_table(
        'Recorded margins, pooled over the three clips',
        ['arm', 'n', 'min', 'p5', 'median', 'p95', 'max', 'floor', 'would abstain'],
        [
            [
                ARM_STYLES[arm].label,
                str(len(frame)),
                f'{frame["confidence"].min():.4f}',
                f'{frame["confidence"].quantile(0.05):.4f}',
                f'{frame["confidence"].median():.4f}',
                f'{frame["confidence"].quantile(0.95):.4f}',
                f'{frame["confidence"].max():.4f}',
                f'{thresholds[arm]:.2f}',
                f'{(frame["confidence"] < thresholds[arm]).mean() * 100:.1f}%',
            ]
            for arm, frame in arms.items()
        ],
    )
    print_table(
        'Median margin per clip',
        ['arm', *CLIPS],
        [
            [
                ARM_STYLES[arm].label,
                *[f'{frame.loc[frame["clip"] == clip, "confidence"].median():.4f}' for clip in CLIPS],
            ]
            for arm, frame in arms.items()
        ],
    )

    # The single most quotable separation in the measurement, computed here so
    # the caption states a figure this script printed rather than one read off
    # the curves by eye.
    unprompted_maximum = float(arms['embedding']['confidence'].max())
    above = (arms['fashionclip']['confidence'] > unprompted_maximum).mean() * 100
    print(
        f'\nThe unprompted arm never exceeds {unprompted_maximum:.4f}; '
        f'{above:.1f}% of prompted crops are more confident than that.'
    )


def verify_font_sizes() -> float:
    """Confirm no text is set below the 6 pt floor, returning the smallest size used."""
    sizes = {
        'title': TITLE_FONT_SIZE,
        'axis label': AXIS_LABEL_FONT_SIZE,
        'tick label': TICK_FONT_SIZE,
        'legend': LEGEND_FONT_SIZE,
        'annotation': ANNOTATION_FONT_SIZE,
    }
    below = {name: size for name, size in sizes.items() if size < MINIMUM_FONT_SIZE_PT}
    if below:
        raise ValueError(
            f'These text elements are set below the {MINIMUM_FONT_SIZE_PT:.0f} pt floor: '
            f'{", ".join(f"{name} at {size} pt" for name, size in below.items())}. '
            f'The figure is placed at its drawn size, so a size set here is the size printed.'
        )
    return min(sizes.values())


def ecdf_at(values: np.ndarray, x: float) -> float:
    """Return an ECDF's height at x, which for a step drawn post is the fraction of values at or below x."""
    return float(np.searchsorted(values, x, side='right')) / len(values)


def verify_nothing_clipped(figure, axes) -> float:
    """Confirm every text element falls inside the figure, returning the smallest margin to an edge in points."""
    # The margins are set by hand to exact fractions rather than by
    # tight_layout, which is what keeps the figure at precisely 18.0 cm. The
    # cost of setting them by hand is that too small a margin silently crops a
    # label at the paper's edge instead of resizing to fit, which is how the
    # y-axis label and the last x tick were lost at 0.052 and 0.995. So every
    # text element's rendered box is checked against the canvas.
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    width, height = figure.canvas.get_width_height()

    elements = [
        ('title', axes.title),
        ('x-axis label', axes.xaxis.label),
        ('y-axis label', axes.yaxis.label),
    ]
    elements += [(f'x tick {text.get_text()!r}', text) for text in axes.get_xticklabels()]
    elements += [(f'y tick {text.get_text()!r}', text) for text in axes.get_yticklabels()]
    for legend in figure.legends:
        elements += [(f'legend entry {text.get_text()!r}', text) for text in legend.get_texts()]

    smallest = float('inf')
    clipped = []
    for name, element in elements:
        if not element.get_text():
            continue
        box = element.get_window_extent(renderer=renderer)
        margin = min(box.x0, box.y0, width - box.x1, height - box.y1)
        smallest = min(smallest, margin)
        if margin < 0:
            clipped.append(f'{name} overruns the figure edge by {abs(margin) / figure.dpi * 72:.1f} pt')
    if clipped:
        raise ValueError(
            'These elements are cut off by the figure edge: ' + '; '.join(clipped) +
            '. Widen the corresponding margin in AXES_LEFT, AXES_RIGHT, AXES_TOP or AXES_BOTTOM.'
        )
    return smallest / figure.dpi * 72


def verify_annotation_clearance(figure, axes, annotation, arms: dict[str, pd.DataFrame]) -> float:
    """Confirm no curve passes through the annotation's rendered box, returning the smallest vertical clearance."""
    # The annotation is placed by hand at a height chosen from the measured
    # curves, so it is only clear for as long as the curves keep their shape.
    # Rather than trust that, the box the text actually occupies is measured
    # from the renderer and converted into data coordinates, then each arm's
    # ECDF is evaluated across exactly that span. Checking the whole width to
    # the right of the line instead would be wrong in the other direction: the
    # K-means curve does cross this height, but far past where the text ends.
    figure.canvas.draw()
    box = annotation.get_window_extent(renderer=figure.canvas.get_renderer())
    inverse = axes.transData.inverted()
    x_low, y_low = inverse.transform((box.x0, box.y0))
    x_high, y_high = inverse.transform((box.x1, box.y1))

    clearance = 1.0
    for arm, frame in arms.items():
        values, _ = ecdf(frame['confidence'].to_numpy(dtype=float))
        # An ECDF is monotone, so over the span the text covers it runs between
        # its height at each end and cannot stray outside that range.
        low = ecdf_at(values, x_low)
        high = ecdf_at(values, x_high)
        if low <= y_high and high >= y_low:
            raise ValueError(
                f"the {arm} curve runs from {low:.3f} to {high:.3f} across the annotation's span "
                f'({x_low:.3f} to {x_high:.3f}), which overlaps the text box '
                f'({y_low:.3f} to {y_high:.3f}). Move ANNOTATION_Y to a clear band.'
            )
        clearance = min(clearance, y_low - high if high < y_low else low - y_high)
    return clearance


def render(arms: dict[str, pd.DataFrame], crops: int) -> list[Path]:
    """Draw the three cumulative distributions and write the figure, returning the paths written."""
    figure, axes = plt.subplots(figsize=FIGURE_SIZE_IN)
    figure.subplots_adjust(left=AXES_LEFT, right=AXES_RIGHT, top=AXES_TOP, bottom=AXES_BOTTOM)

    for arm, frame in arms.items():
        style = ARM_STYLES[arm]
        values, cumulative = ecdf(frame['confidence'].to_numpy(dtype=float))
        # A step rather than a line: the distribution is a sample, and drawing
        # it as a continuous curve would imply interpolation between crops that
        # were never measured.
        axes.step(
            values, cumulative, where='post',
            label=style.label, color=style.colour, linestyle=style.linestyle, linewidth=style.linewidth,
        )

    # The one annotated value on the figure, set horizontally to the right of
    # its own line. An earlier revision set it vertically, which fitted a taller
    # figure but would span almost the whole height of this one; and the legend
    # it was dodging is now outside the axes entirely, which frees the band this
    # sits in. The panel behind it keeps it legible if a curve does move across.
    unprompted_maximum = float(arms['embedding']['confidence'].max())
    axes.axvline(unprompted_maximum, color='#3a3a3a', linewidth=0.8, linestyle=(0, (1, 2.5)))
    annotation = axes.text(
        unprompted_maximum + ANNOTATION_X_OFFSET, ANNOTATION_Y,
        f'unprompted maximum {unprompted_maximum:.3f}',
        fontsize=ANNOTATION_FONT_SIZE, color='#3a3a3a', ha='left', va='center',
        bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': 0.9, 'pad': 1.5},
    )

    axes.set_xlim(*X_AXIS_LIMITS)
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel(X_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    axes.set_ylabel(Y_AXIS_LABEL, fontsize=AXIS_LABEL_FONT_SIZE)
    axes.tick_params(labelsize=TICK_FONT_SIZE, length=2.5, width=0.6, pad=2)
    for spine in axes.spines.values():
        spine.set_linewidth(0.7)
    axes.grid(True, linewidth=0.35, alpha=0.35)
    axes.set_axisbelow(True)
    axes.set_title(
        f'Team assignment margins, {crops} crops per arm at crop fraction {COMMON_CROP_FRACTION}',
        fontsize=TITLE_FONT_SIZE, pad=4,
    )

    # Attached to the figure rather than the axes, in one row beneath the plot.
    handles, labels = axes.get_legend_handles_labels()
    figure.legend(
        handles, labels, loc='lower center', bbox_to_anchor=(0.5, LEGEND_Y),
        ncol=LEGEND_COLUMNS, frameon=False, fontsize=LEGEND_FONT_SIZE,
        handlelength=3.4, columnspacing=2.6, handletextpad=0.7,
    )

    # Checked here rather than before drawing: the annotation's extent is only
    # knowable once the text exists and the axes limits it is measured against
    # are set.
    clearance = verify_annotation_clearance(figure, axes, annotation, arms)
    print(f'Nearest curve clears the annotation by {clearance:.3f} in cumulative fraction.')
    margin = verify_nothing_clipped(figure, axes)
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
    """Verify the three arms' provenance, print the summary and write the figure."""
    config = load_config(require_file(
        CONFIG_PATH, 'the pipeline configuration', 'It is committed, so an absent one means an incomplete checkout.',
    ))
    team_config = config['team_classifier']
    thresholds = {arm: float(team_config[key]) for arm, key in ABSTENTION_THRESHOLD_KEYS.items()}

    arms = {arm: load_arm(arm) for arm in ARMS}

    crops = verify_common_crops(arms)
    print(f'All three arms measured on an identical set of {crops} crops.')
    for arm in ARMS:
        difference = verify_crop_provenance(arm, arms[arm])
        print(f'{ARM_STYLES[arm].label}: reproduces {GRID_STEMS[arm]} (largest confidence difference {difference:.1e}).')

    summarise(arms, thresholds)

    smallest = verify_font_sizes()
    print(f'\nSmallest text is {smallest:.1f} pt, at or above the {MINIMUM_FONT_SIZE_PT:.0f} pt floor.')
    print(f'Figure {FIGURE_WIDTH_CM:.1f} cm wide by {FIGURE_HEIGHT_CM:.1f} cm tall, spanning both columns.')

    print()
    for path in render(arms, crops):
        print(f'Wrote {path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
