from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from age_gender_cnn.inference import FaceCrop, InferenceResult, ModelPrediction
from age_gender_cnn.plots import (
    plot_architecture_comparison,
    plot_augmentation_grid,
    plot_inference_result,
    plot_reference_curves,
    plot_selected_metrics,
)


ROOT = Path(__file__).parents[1]


def test_example_asset_has_complete_attribution() -> None:
    text = (ROOT / "assets" / "README.md").read_text()
    assert "The Beaverton" in text
    assert "CC BY 3.0" in text
    assert (
        "https://commons.wikimedia.org/wiki/File:Simu_Liu_on_The_Beaverton.jpg"
        in text
    )
    assert "cropping and augmentation" in text


def test_augmentation_grid_contains_original_and_eight_variants() -> None:
    image = np.zeros((128, 128, 3), dtype=np.float32)
    figure = plot_augmentation_grid(image, augment=lambda value: value, variants=8)
    assert len(figure.axes) == 9
    plt.close(figure)


def test_selected_metric_figure_has_both_models() -> None:
    figure = plot_selected_metrics(
        {
            "model_a": {
                "age_output_mae": 6.5747,
                "gender_output_accuracy": 0.901,
            },
            "model_b": {
                "age_output_mae": 7.2968,
                "gender_output_accuracy": 0.889,
            },
        }
    )
    assert len(figure.axes) == 2
    plt.close(figure)


def test_reference_curve_montage_uses_four_images(tmp_path: Path) -> None:
    for suffix in ("gender_loss", "gender_accuracy", "age_loss", "age_mae"):
        plt.imsave(
            tmp_path / f"model_a_{suffix}.png",
            np.zeros((10, 10, 3)),
        )
    figure = plot_reference_curves(tmp_path, "model_a")
    assert len(figure.axes) == 4
    plt.close(figure)


def test_inference_and_architecture_figures_have_stable_panels() -> None:
    rgb = np.zeros((32, 32, 3), dtype=np.uint8)
    result = InferenceResult(
        rgb,
        FaceCrop(rgb, "centre_crop_fallback", 0),
        (
            ModelPrediction("model_a", 31.2, 0.7, "female"),
            ModelPrediction("model_b", 29.8, 0.6, "female"),
        ),
    )
    inference_figure = plot_inference_result(result)
    architecture_figure = plot_architecture_comparison()
    assert len(inference_figure.axes) == 3
    assert len(architecture_figure.axes) == 2
    plt.close(inference_figure)
    plt.close(architecture_figure)
