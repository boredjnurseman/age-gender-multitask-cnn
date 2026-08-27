"""Data utilities for repeatable portfolio runs.

Paths are sorted before shuffling so new runs are reproducible across filesystems.
This cannot reconstruct the historical notebook's unsorted directory iteration;
the committed metrics therefore remain reference evidence for the original run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Sequence

import cv2
import numpy as np


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class FaceLabel:
    age: int
    gender: int


@dataclass(frozen=True)
class DatasetSplit:
    train: tuple[Path, ...]
    validation: tuple[Path, ...]


@dataclass(frozen=True)
class DatasetArrays:
    images: np.ndarray
    ages: np.ndarray
    genders: np.ndarray


def parse_utkface_filename(filename: str) -> FaceLabel:
    """Parse and validate age and binary gender fields in a UTKFace filename."""
    parts = Path(filename).stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"Expected age_gender_race fields in {filename!r}")
    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError as exc:
        raise ValueError(f"Non-numeric age or gender field in {filename!r}") from exc
    if not 0 <= age <= 116:
        raise ValueError(f"Age outside UTKFace range in {filename!r}")
    if gender not in (0, 1):
        raise ValueError(f"Gender label must be 0 or 1 in {filename!r}")
    return FaceLabel(age=age, gender=gender)


def list_image_paths(data_dir: Path) -> list[Path]:
    """Return supported image paths in stable filename order."""
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Training directory not found: {data_dir}")
    paths = sorted(
        path for path in data_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise ValueError(f"No supported images found in {data_dir}")
    for path in paths:
        parse_utkface_filename(path.name)
    return paths


def split_paths(
    paths: Sequence[Path], train_fraction: float = 0.8, seed: int = 0
) -> DatasetSplit:
    """Shuffle a supplied stable path sequence and return train/validation partitions."""
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie strictly between 0 and 1")
    shuffled = list(paths)
    random.Random(seed).shuffle(shuffled)
    boundary = int(len(shuffled) * train_fraction)
    if boundary == 0 or boundary == len(shuffled):
        raise ValueError("Both partitions must contain at least one image")
    return DatasetSplit(tuple(shuffled[:boundary]), tuple(shuffled[boundary:]))


def load_rgb_image(path: Path) -> np.ndarray:
    """Load one 128x128 training image as RGB float32 in [0, 1]."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode {path}")
    if bgr.shape != (128, 128, 3):
        raise ValueError(
            f"Expected 128x128x3 training image, got {bgr.shape} for {path.name}"
        )
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0


def load_dataset(paths: Sequence[Path]) -> DatasetArrays:
    """Load RGB arrays and align age/gender labels with path order."""
    images, ages, genders = [], [], []
    for path in paths:
        label = parse_utkface_filename(path.name)
        images.append(load_rgb_image(path))
        ages.append(float(label.age))
        genders.append(float(label.gender))
    return DatasetArrays(
        images=np.asarray(images, dtype=np.float32),
        ages=np.asarray(ages, dtype=np.float32),
        genders=np.asarray(genders, dtype=np.float32),
    )


def build_augmentation_layers(keras_module: Any | None = None) -> Any:
    """Build the conservative augmentation sequence used by both models."""
    if keras_module is None:
        import keras as keras_module
    layers = keras_module.layers
    return keras_module.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.03),
            layers.RandomZoom(0.05),
            layers.RandomTranslation(0.05, 0.05),
        ],
        name="conservative_augmentation",
    )
