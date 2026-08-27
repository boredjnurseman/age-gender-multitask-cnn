"""Small plotting helpers that keep the notebook focused on interpretation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from .inference import InferenceResult


MODEL_COLOURS = {"model_a": "#315c8c", "model_b": "#bd5b35"}


def plot_augmentation_grid(
    image: np.ndarray,
    *,
    augment: Callable[[np.ndarray], np.ndarray],
    variants: int = 8,
    base_title: str = "Original CC BY example",
) -> Any:
    """Show one licensed base image beside repeated outputs from the supplied augmenter."""
    figure, axes = plt.subplots(3, 3, figsize=(9, 9))
    axes = axes.ravel()
    axes[0].imshow(np.clip(image, 0, 1))
    axes[0].set_title(base_title)
    for index in range(variants):
        axes[index + 1].imshow(np.clip(augment(image), 0, 1))
        axes[index + 1].set_title(f"Augmented {index + 1}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure

def plot_selected_metrics(metrics: Mapping[str, Mapping[str, float]]) -> Any:
    """Compare age MAE and binary label accuracy without reducing multitask results to one winner."""
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    keys = ["model_a", "model_b"]
    labels = ["Model A\ncustom residual", "Model B\nResNet50V2"]
    colours = [MODEL_COLOURS[key] for key in keys]
    axes[0].bar(
        labels,
        [metrics[key]["age_output_mae"] for key in keys],
        color=colours,
    )
    axes[0].set_ylabel("Validation MAE (years; lower is better)")
    axes[1].bar(
        labels,
        [metrics[key]["gender_output_accuracy"] for key in keys],
        color=colours,
    )
    axes[1].set_ylabel("Validation accuracy (higher is better)")
    figure.tight_layout()
    return figure


def plot_reference_curves(curve_dir: Path, model_key: str) -> Any:
    """Montage four frozen reference curves without altering their extracted pixel content or labels."""
    suffixes = ("gender_loss", "gender_accuracy", "age_loss", "age_mae")
    titles = (
        "Gender-label loss",
        "Gender-label accuracy",
        "Age loss",
        "Age MAE",
    )
    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    for axis, suffix, title in zip(
        axes.ravel(), suffixes, titles, strict=True
    ):
        path = curve_dir / f"{model_key}_{suffix}.png"
        if not path.is_file():
            raise FileNotFoundError(path)
        axis.imshow(plt.imread(path))
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle(
        f"{model_key.replace('_', ' ').title()} reference training curves"
    )
    figure.tight_layout()
    return figure


def plot_inference_result(result: InferenceResult) -> Any:
    """Show the submitted image, shared crop, and both models' task outputs together."""
    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    axes[0].imshow(result.original_rgb)
    axes[0].set_title("Submitted photograph")
    axes[1].imshow(result.crop.rgb)
    axes[1].set_title(
        f"Model crop: {result.crop.method.replace('_', ' ')}"
    )
    lines = []
    for prediction in result.predictions:
        label = "Model A" if prediction.model_key == "model_a" else "Model B"
        lines.extend(
            (
                label,
                f"Age estimate: {prediction.age_years:.1f}",
                (
                    f"Dataset-coded {prediction.dataset_label} score: "
                    f"{prediction.gender_score:.3f}"
                ),
                "",
            )
        )
    axes[2].text(0.0, 1.0, "\n".join(lines), va="top", fontsize=11)
    axes[2].set_title("Side-by-side outputs")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    return figure
