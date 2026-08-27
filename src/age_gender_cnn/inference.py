from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Literal, Mapping

import cv2
import numpy as np


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    filename: str
    drive_file_id: str
    minimum_bytes: int


MODEL_SPECS = {
    "model_a": ModelSpec(
        "Model A - custom residual CNN",
        "age_gender_A.keras",
        "1AfnCchElx08FN0jGGLz91tl19YKWVP09",
        25_000_000,
    ),
    "model_b": ModelSpec(
        "Model B - ResNet50V2",
        "age_gender_B.keras",
        "1TdKyD8Bo7tByfuJxQbJUbnuak4tERydN",
        200_000_000,
    ),
}


@dataclass(frozen=True)
class FaceCrop:
    rgb: np.ndarray
    method: Literal["largest_detected_face", "centre_crop_fallback"]
    detected_count: int


@dataclass(frozen=True)
class ModelPrediction:
    model_key: str
    age_years: float
    gender_score: float
    dataset_label: Literal["male", "female"]


@dataclass(frozen=True)
class InferenceResult:
    original_rgb: np.ndarray
    crop: FaceCrop
    predictions: tuple[ModelPrediction, ...]


def _default_detector() -> Any:
    cascade = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(cascade))
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade}")
    return detector


def detect_and_crop_face(
    rgb: np.ndarray,
    *,
    detector: Any | None = None,
    padding_fraction: float = 0.30,
) -> FaceCrop:
    """Use the largest detected face or a square centre-crop fallback."""
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with three channels, got {rgb.shape}")
    detector = detector or _default_detector()
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    boxes = detector.detectMultiScale(
        grey,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
    )
    if len(boxes) == 0:
        height, width = rgb.shape[:2]
        side = min(height, width)
        top, left = (height - side) // 2, (width - side) // 2
        return FaceCrop(
            rgb[top : top + side, left : left + side],
            "centre_crop_fallback",
            0,
        )
    x, y, width, height = max(
        boxes, key=lambda box: int(box[2]) * int(box[3])
    )
    padding = int(padding_fraction * max(width, height))
    top, left = max(0, y - padding), max(0, x - padding)
    bottom = min(rgb.shape[0], y + height + padding)
    right = min(rgb.shape[1], x + width + padding)
    return FaceCrop(
        rgb[top:bottom, left:right],
        "largest_detected_face",
        len(boxes),
    )


def prepare_model_batch(crop_rgb: np.ndarray) -> np.ndarray:
    """Resize an RGB uint8 crop to the models' float32 input contract."""
    resized = cv2.resize(crop_rgb, (128, 128), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)[None, ...]


def _valid_keras_archive(path: Path, minimum_bytes: int) -> bool:
    if not path.is_file() or path.stat().st_size < minimum_bytes:
        return False
    with path.open("rb") as stream:
        return stream.read(2) == b"PK"


def download_models(
    cache_dir: Path, *, downloader: Any | None = None
) -> dict[str, Path]:
    """Download only the two final models and reuse validated cache files."""
    if downloader is None:
        import gdown

        downloader = gdown.download
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for key, spec in MODEL_SPECS.items():
        target = cache_dir / spec.filename
        if not _valid_keras_archive(target, spec.minimum_bytes):
            downloader(
                id=spec.drive_file_id,
                output=str(target),
                quiet=False,
            )
        if not _valid_keras_archive(target, spec.minimum_bytes):
            raise RuntimeError(
                f"Downloaded {spec.filename} is missing, truncated or not a .keras archive"
            )
        paths[key] = target
    return paths


def register_upload_bytes(upload_bytes: bytes, seen_digests: set[str]) -> str:
    """Record an upload digest and reject bytes already seen in this session."""
    digest = hashlib.sha256(upload_bytes).hexdigest()
    if digest in seen_digests:
        raise ValueError("duplicate upload bytes are not accepted")
    seen_digests.add(digest)
    return digest


def map_named_outputs(
    raw_outputs: Any, *, output_names: list[str]
) -> dict[str, float]:
    """Map Keras prediction arrays by output name and require both tasks."""
    if isinstance(raw_outputs, Mapping):
        mapped = {
            name: float(np.asarray(value).reshape(-1)[0])
            for name, value in raw_outputs.items()
        }
    else:
        mapped = {
            name: float(np.asarray(value).reshape(-1)[0])
            for name, value in zip(output_names, raw_outputs, strict=True)
        }
    missing = {"gender_output", "age_output"} - set(mapped)
    if missing:
        raise ValueError(f"Model outputs missing required tasks: {sorted(missing)}")
    return mapped


def _read_rgb(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"OpenCV could not decode uploaded image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def predict_photo(
    models: Mapping[str, Any],
    image_path: Path,
    *,
    detector: Any | None = None,
) -> InferenceResult:
    """Run both supplied models on the same visible face crop."""
    original = _read_rgb(image_path)
    crop = detect_and_crop_face(original, detector=detector)
    batch = prepare_model_batch(crop.rgb)
    predictions: list[ModelPrediction] = []
    for key in ("model_a", "model_b"):
        if key not in models:
            raise KeyError(f"Required model missing: {key}")
        model = models[key]
        mapped = map_named_outputs(
            model.predict(batch, verbose=0),
            output_names=list(model.output_names),
        )
        score = mapped["gender_output"]
        predictions.append(
            ModelPrediction(
                key,
                mapped["age_output"],
                score,
                "female" if score > 0.5 else "male",
            )
        )
    return InferenceResult(original, crop, tuple(predictions))
