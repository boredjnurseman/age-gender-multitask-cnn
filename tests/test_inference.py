from pathlib import Path

import numpy as np
import pytest

from age_gender_cnn import inference
from age_gender_cnn.inference import (
    ModelSpec,
    detect_and_crop_face,
    download_models,
    map_named_outputs,
    predict_photo,
    prepare_model_batch,
)


class FakeDetector:
    def __init__(self, boxes):
        self.boxes = np.asarray(boxes, dtype=np.int32)

    def detectMultiScale(self, *_args, **_kwargs):
        return self.boxes


def test_detect_and_crop_face_selects_largest_box() -> None:
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    crop = detect_and_crop_face(
        image,
        detector=FakeDetector([(10, 10, 50, 50), (100, 50, 100, 120)]),
    )
    assert crop.method == "largest_detected_face"
    assert crop.detected_count == 2
    assert crop.rgb.size > 0


def test_detect_and_crop_face_uses_visible_centre_fallback() -> None:
    image = np.zeros((200, 300, 3), dtype=np.uint8)
    crop = detect_and_crop_face(image, detector=FakeDetector([]))
    assert crop.method == "centre_crop_fallback"
    assert crop.rgb.shape == (200, 200, 3)


def test_prepare_model_batch_matches_training_contract() -> None:
    batch = prepare_model_batch(np.full((80, 60, 3), 255, dtype=np.uint8))
    assert batch.shape == (1, 128, 128, 3)
    assert batch.dtype == np.float32
    assert np.allclose(batch, 1.0)


def test_map_named_outputs_does_not_assume_list_order() -> None:
    mapped = map_named_outputs(
        [np.asarray([[41.5]]), np.asarray([[0.8]])],
        output_names=["age_output", "gender_output"],
    )
    assert mapped == {"age_output": 41.5, "gender_output": 0.8}


def test_download_models_reuses_valid_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []
    tiny_specs = {
        "model_a": ModelSpec("Model A", "a.keras", "a-drive-id", 4),
        "model_b": ModelSpec("Model B", "b.keras", "b-drive-id", 4),
    }
    monkeypatch.setattr(inference, "MODEL_SPECS", tiny_specs)
    for spec in tiny_specs.values():
        (tmp_path / spec.filename).write_bytes(b"PK00")
    paths = download_models(
        tmp_path, downloader=lambda **kwargs: calls.append(kwargs)
    )
    assert not calls
    assert set(paths) == {"model_a", "model_b"}


def test_output_mapping_rejects_missing_task() -> None:
    with pytest.raises(ValueError, match="age_output"):
        map_named_outputs(
            [np.asarray([[0.6]])], output_names=["gender_output"]
        )


def test_predict_photo_returns_both_models_for_the_same_crop(tmp_path: Path) -> None:
    import cv2

    class FixedModel:
        output_names = ["gender_output", "age_output"]

        def __init__(self, score: float, age: float) -> None:
            self.outputs = [
                np.asarray([[score]], dtype=np.float32),
                np.asarray([[age]], dtype=np.float32),
            ]

        def predict(self, batch, verbose=0):
            assert batch.shape == (1, 128, 128, 3)
            assert verbose == 0
            return self.outputs

    image_path = tmp_path / "permitted-photo.png"
    bgr = np.zeros((180, 240, 3), dtype=np.uint8)
    bgr[..., 2] = 255
    assert cv2.imwrite(str(image_path), bgr)

    result = predict_photo(
        {
            "model_a": FixedModel(0.75, 31.25),
            "model_b": FixedModel(0.25, 34.5),
        },
        image_path,
        detector=FakeDetector([]),
    )

    assert result.crop.method == "centre_crop_fallback"
    assert [prediction.model_key for prediction in result.predictions] == [
        "model_a",
        "model_b",
    ]
    assert [prediction.dataset_label for prediction in result.predictions] == [
        "female",
        "male",
    ]
    assert [prediction.age_years for prediction in result.predictions] == [
        pytest.approx(31.25),
        pytest.approx(34.5),
    ]
