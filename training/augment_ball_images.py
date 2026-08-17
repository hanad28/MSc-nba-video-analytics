"""Generates targeted offline augmentation for ball-labelled training images.

Filters the YOLO training split to images containing at least one ball
annotation, applies a randomised augmentation pipeline (rotation,
brightness/contrast, motion blur, scale jitter) to produce a configurable
number of augmented copies per image, and writes the new image/label pairs
into the same training split. Validation and test splits are left
untouched, since augmenting evaluation data would invalidate the
before/after comparison this script supports.
"""

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import yaml


@dataclass
class DatasetBalance:
    """Ball annotation count in a training split, before or after augmentation."""

    ball_annotations: int


def load_class_names(dataset_dir: Path) -> list[str]:
    """Reads class names from the dataset's data.yaml file."""
    data_yaml_path = dataset_dir / 'data.yaml'
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    return data_config['names']


def load_yolo_label(label_path: Path) -> tuple[list[int], list[list[float]]]:
    """Reads a YOLO-format label file, returning class ids and normalised bboxes."""
    class_ids: list[int] = []
    boxes: list[list[float]] = []
    if not label_path.exists():
        return class_ids, boxes
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            class_ids.append(int(parts[0]))
            boxes.append([float(value) for value in parts[1:5]])
    return class_ids, boxes


def save_yolo_label(label_path: Path, class_ids: list[int], boxes: list[list[float]]) -> None:
    """Writes a YOLO-format label file from class ids and normalised bboxes."""
    lines = [
        f'{int(class_id)} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}'
        for class_id, box in zip(class_ids, boxes)
    ]
    with open(label_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


def contains_ball(class_ids: list[int], ball_class_index: int) -> bool:
    """Checks whether a list of class ids includes the ball class."""
    return ball_class_index in class_ids


def find_ball_images(images_dir: Path, labels_dir: Path, ball_class_index: int) -> list[Path]:
    """Finds all training images whose label file contains at least one ball annotation."""
    ball_images: list[Path] = []
    for image_path in sorted(images_dir.iterdir()):
        if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
            continue
        label_path = labels_dir / f'{image_path.stem}.txt'
        class_ids, _ = load_yolo_label(label_path)
        if contains_ball(class_ids, ball_class_index):
            ball_images.append(image_path)
    return ball_images


def build_augmentation_pipeline() -> A.Compose:
    """Builds the targeted augmentation pipeline for ball-labelled images.

    Each transform is applied independently at random, so augmented copies
    of the same source image vary in which transforms they receive rather
    than always combining all four. Horizontal flip is deliberately
    excluded, since Ultralytics already applies it by default to every
    training image regardless of class.
    """
    return A.Compose(
        [
            A.Rotate(limit=15, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.MotionBlur(blur_limit=7, p=0.4),
            A.RandomScale(scale_limit=0.2, p=0.4),
        ],
        bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], min_visibility=0.3),
    )


def augment_image(
    image_path: Path,
    label_path: Path,
    pipeline: A.Compose,
    output_images_dir: Path,
    output_labels_dir: Path,
    copy_index: int,
    ball_class_index: int,
) -> int:
    """Generates one augmented copy of an image/label pair and writes it to disk.

    Returns the number of ball annotations present in the augmented copy,
    used to track the resulting dataset balance.
    """
    image = cv2.imread(str(image_path))
    class_ids, boxes = load_yolo_label(label_path)

    augmented = pipeline(image=image, bboxes=boxes, class_labels=class_ids)

    output_stem = f'{image_path.stem}_aug{copy_index}'
    output_image_path = output_images_dir / f'{output_stem}{image_path.suffix}'
    output_label_path = output_labels_dir / f'{output_stem}.txt'

    cv2.imwrite(str(output_image_path), augmented['image'])
    save_yolo_label(output_label_path, augmented['class_labels'], augmented['bboxes'])

    return augmented['class_labels'].count(ball_class_index)


def measure_balance(labels_dir: Path, ball_class_index: int) -> DatasetBalance:
    """Counts ball annotations across all label files in a directory."""
    ball_count = 0
    for label_path in labels_dir.glob('*.txt'):
        class_ids, _ = load_yolo_label(label_path)
        ball_count += class_ids.count(ball_class_index)
    return DatasetBalance(ball_annotations=ball_count)


def count_images(images_dir: Path) -> int:
    """Counts image files in a directory, ignoring label and other non-image files."""
    return sum(
        1 for image_path in images_dir.iterdir()
        if image_path.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )


def main() -> None:
    """Runs targeted augmentation on ball-labelled training images and reports the before/after balance."""
    parser = argparse.ArgumentParser(description='Targeted augmentation for ball-labelled training images.')
    parser.add_argument('--dataset-dir', type=Path, required=True, help='Root of the downloaded YOLO dataset.')
    parser.add_argument('--num-copies', type=int, default=2, help='Augmented copies to generate per ball image.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility.')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    class_names = load_class_names(args.dataset_dir)
    ball_class_index = class_names.index('ball')

    images_dir = args.dataset_dir / 'train' / 'images'
    labels_dir = args.dataset_dir / 'train' / 'labels'

    images_before = count_images(images_dir)
    balance_before = measure_balance(labels_dir, ball_class_index)
    print(f'Before augmentation: {images_before} images, {balance_before.ball_annotations} ball annotations.')

    ball_images = find_ball_images(images_dir, labels_dir, ball_class_index)
    print(f'Found {len(ball_images)} training images containing the ball class.')

    pipeline = build_augmentation_pipeline()
    for image_path in ball_images:
        label_path = labels_dir / f'{image_path.stem}.txt'
        for copy_index in range(args.num_copies):
            augment_image(
                image_path, label_path, pipeline, images_dir, labels_dir, copy_index, ball_class_index
            )

    images_after = count_images(images_dir)
    balance_after = measure_balance(labels_dir, ball_class_index)
    print(f'After augmentation: {images_after} images, {balance_after.ball_annotations} ball annotations.')
    print(
        f'Train split expanded from {images_before} to {images_after} images '
        f'(x{images_after / images_before:.1f}) and from {balance_before.ball_annotations} '
        f'to {balance_after.ball_annotations} ball annotations '
        f'(x{balance_after.ball_annotations / balance_before.ball_annotations:.1f}).'
    )


if __name__ == '__main__':
    main()
