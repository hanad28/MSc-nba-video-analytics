"""
split_court_keypoints.py

Rebuilds the court-keypoint dataset with a source-video grouped train/valid/test
split, replacing Roboflow's random image-level split, and writes a corrected
flip_idx. Run from the repository root.
"""
from __future__ import annotations

import collections
import os
import re
import shutil

import yaml

SOURCE_DIR = 'training/court-keypoints'
TARGET_DIR = 'training/court-keypoints-grouped'
SPLITS = {'train': 0.70, 'valid': 0.15, 'test': 0.15}

# Established by visual identification of every landmark on rendered frames:
# baselines run 0-5 (left) and 10-15 (right), lane corners 8/9 and 16/17,
# centre line 6/7. Roboflow ships the identity mapping, which renumbers
# nothing and so teaches each index both of its mirror-symmetric positions.
FLIP_IDX = [15, 14, 13, 12, 11, 10, 6, 7, 16, 17, 5, 4, 3, 2, 1, 0, 8, 9]


def source_video(stem: str) -> str:
    """Return the source video a frame came from, using a per-family rule."""
    if '_frame_' in stem:
        return stem.split('_frame_')[0]
    if stem.startswith('camcourt'):
        # camcourt1 and camcourt2 share timestamps: two synchronised cameras on
        # the same instant, which is a near-duplicate across viewpoints.
        return f'camcourt_{stem.split("_")[1]}'
    if stem.startswith('faasdasd'):
        return 'faasdasd'
    # basketball_N_M_K is game N, clip M, frame K; group at game level, since
    # one game is one court and one camera position.
    return '_'.join(stem.split('_')[:2])


def clamp_label(line: str) -> str:
    """Clamp a label line's normalised coordinates into [0, 1], leaving the class index and visibility flags untouched."""
    parts = line.split()
    values = [float(v) for v in parts[1:]]
    box = [min(1.0, max(0.0, v)) for v in values[:4]]
    keypoints = []
    for i in range(0, len(values[4:]), 3):
        x, y, visibility = values[4 + i], values[5 + i], values[6 + i]
        keypoints += [min(1.0, max(0.0, x)), min(1.0, max(0.0, y)), visibility]
    return ' '.join([parts[0]] + [f'{v:.10g}' for v in box + keypoints])


def collect() -> dict[str, list[tuple[str, str]]]:
    """Map each source video to its (image_path, label_path) pairs across all original splits."""
    groups: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
    for split in ('train', 'valid', 'test'):
        image_dir = f'{SOURCE_DIR}/{split}/images'
        if not os.path.isdir(image_dir):
            continue
        for image_name in sorted(os.listdir(image_dir)):
            base = os.path.splitext(image_name)[0]
            label = f'{SOURCE_DIR}/{split}/labels/{base}.txt'
            if not os.path.exists(label):
                raise FileNotFoundError(f'Image {image_name} has no label at {label}')
            stem = re.sub(r'_(jpg|png|jpeg)$', '', base.split('.rf.')[0])
            groups[source_video(stem)].append((f'{image_dir}/{image_name}', label))
    return groups


def assign(groups: dict[str, list]) -> dict[str, str]:
    """Assign whole source videos to splits, largest first, each to the split furthest below target."""
    total = sum(len(v) for v in groups.values())
    targets = {s: total * f for s, f in SPLITS.items()}
    counts = {s: 0 for s in SPLITS}
    placement: dict[str, str] = {}

    # Deterministic by construction: no seed, so the split is a fixed function
    # of the dataset and reproduces without recording one.
    for name in sorted(groups, key=lambda k: (-len(groups[k]), k)):
        split = max(SPLITS, key=lambda s: targets[s] - counts[s])
        placement[name] = split
        counts[split] += len(groups[name])
    return placement


def main() -> None:
    groups = collect()
    placement = assign(groups)

    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)
    for split in SPLITS:
        os.makedirs(f'{TARGET_DIR}/{split}/images')
        os.makedirs(f'{TARGET_DIR}/{split}/labels')

    for name, pairs in groups.items():
        split = placement[name]
        for image_path, label_path in pairs:
            shutil.copy2(image_path, f'{TARGET_DIR}/{split}/images/')
            # Roboflow exports off-frame padding coordinates for absent (v=0)
            # keypoints, up to 0.48 outside [0, 1]. Ultralytics discards the
            # WHOLE image for any coordinate out of range, and does so silently:
            # one warning per file, no summary. Measured from run A's own
            # labels.cache, that cost 259 of 1,028 training images (25.2%) and
            # 38 of 220 validation images (17.3%), and the discarded frames are
            # exactly those with off-frame landmarks -- the harder cases.
            # Clamping loses nothing: v=0 keypoints carry no pose loss, and the
            # single v=2 case sits 0.9 px past the frame edge.
            target = f'{TARGET_DIR}/{split}/labels/{os.path.basename(label_path)}'
            with open(label_path) as source, open(target, 'w') as sink:
                for line in source:
                    if line.strip():
                        sink.write(clamp_label(line) + '\n')

    config = yaml.safe_load(open(f'{SOURCE_DIR}/data.yaml'))
    if config['kpt_shape'] != [18, 3]:
        raise ValueError(f'Expected kpt_shape [18, 3], got {config["kpt_shape"]}')
    if not all(FLIP_IDX[FLIP_IDX[i]] == i for i in range(18)):
        raise ValueError('FLIP_IDX is not an involution, so it is not a mirror mapping')
    print('shipped flip_idx:', config['flip_idx'])
    config['flip_idx'] = FLIP_IDX
    config.update({'train': '../train/images', 'val': '../valid/images', 'test': '../test/images'})
    yaml.safe_dump(config, open(f'{TARGET_DIR}/data.yaml', 'w'), sort_keys=False)
    print('written flip_idx:', config['flip_idx'])

    total = sum(len(v) for v in groups.values())
    print(f'\n{len(groups)} source videos, {total} images')
    for split in SPLITS:
        names = [n for n, s in placement.items() if s == split]
        n_img = len(os.listdir(f'{TARGET_DIR}/{split}/images'))
        n_lbl = len(os.listdir(f'{TARGET_DIR}/{split}/labels'))
        if n_img != n_lbl:
            raise ValueError(f'{split}: {n_img} images against {n_lbl} labels')
        print(f'  {split:5s} {len(names):4d} videos  {n_img:5d} images  '
              f'({100*n_img/total:4.1f}%, target {100*SPLITS[split]:.0f}%)')

    seen: dict[str, str] = {}
    for name, split in placement.items():
        if seen.setdefault(name, split) != split:
            raise ValueError(f'Source {name} spans two splits')
    print('\nno source video appears in more than one split')


if __name__ == '__main__':
    main()
