from pathlib import Path

import numpy as np
import pytest

from age_gender_cnn.data import (
    build_augmentation_layers,
    list_image_paths,
    load_dataset,
    load_rgb_image,
    parse_utkface_filename,
    split_paths,
)


def test_parse_utkface_filename_returns_age_and_binary_label() -> None:
    label = parse_utkface_filename("42_1_3_201701.jpg")
    assert label.age == 42
    assert label.gender == 1


@pytest.mark.parametrize("filename", ["broken.jpg", "117_0_x.jpg", "21_2_x.jpg"])
def test_parse_utkface_filename_rejects_invalid_labels(filename: str) -> None:
    with pytest.raises(ValueError):
        parse_utkface_filename(filename)


def test_list_image_paths_filters_and_sorts_supported_files(tmp_path: Path) -> None:
    for filename in ("42_1_0_b.png", "20_0_0_a.jpg", "notes.txt"):
        (tmp_path / filename).write_bytes(b"fixture")
    paths = list_image_paths(tmp_path)
    assert [path.name for path in paths] == ["20_0_0_a.jpg", "42_1_0_b.png"]


def test_split_paths_is_seeded_and_non_overlapping() -> None:
    paths = [Path(f"{index}_0_0_x.jpg") for index in range(10)]
    split = split_paths(paths, train_fraction=0.8, seed=0)
    assert [path.name for path in split.train] == [
        "7_0_0_x.jpg",
        "8_0_0_x.jpg",
        "1_0_0_x.jpg",
        "5_0_0_x.jpg",
        "3_0_0_x.jpg",
        "4_0_0_x.jpg",
        "2_0_0_x.jpg",
        "0_0_0_x.jpg",
    ]
    assert [path.name for path in split.validation] == ["9_0_0_x.jpg", "6_0_0_x.jpg"]
    assert set(split.train).isdisjoint(split.validation)


def test_load_rgb_image_converts_bgr_and_scales(tmp_path: Path) -> None:
    import cv2

    path = tmp_path / "30_0_0_x.png"
    bgr = np.zeros((128, 128, 3), dtype=np.uint8)
    bgr[..., 0] = 255
    assert cv2.imwrite(str(path), bgr)
    rgb = load_rgb_image(path)
    assert rgb.shape == (128, 128, 3)
    assert rgb.dtype == np.float32
    assert np.allclose(rgb[0, 0], [0.0, 0.0, 1.0])


def test_load_dataset_aligns_images_and_filename_labels(tmp_path: Path) -> None:
    import cv2

    paths = []
    for filename in ("20_0_0_x.jpg", "35_1_0_x.jpg"):
        path = tmp_path / filename
        cv2.imwrite(str(path), np.zeros((128, 128, 3), dtype=np.uint8))
        paths.append(path)
    dataset = load_dataset(paths)
    assert dataset.images.shape == (2, 128, 128, 3)
    assert dataset.ages.tolist() == [20.0, 35.0]
    assert dataset.genders.tolist() == [0.0, 1.0]


def test_augmentation_builder_preserves_the_conservative_sequence() -> None:
    class RecordingLayers:
        @staticmethod
        def RandomFlip(mode):
            return ("flip", mode)

        @staticmethod
        def RandomRotation(amount):
            return ("rotation", amount)

        @staticmethod
        def RandomZoom(amount):
            return ("zoom", amount)

        @staticmethod
        def RandomTranslation(height, width):
            return ("translation", height, width)

    class RecordingKeras:
        layers = RecordingLayers()

        @staticmethod
        def Sequential(layers, name):
            return {"layers": layers, "name": name}

    sequence = build_augmentation_layers(RecordingKeras())
    assert sequence == {
        "layers": [
            ("flip", "horizontal"),
            ("rotation", 0.03),
            ("zoom", 0.05),
            ("translation", 0.05, 0.05),
        ],
        "name": "conservative_augmentation",
    }
