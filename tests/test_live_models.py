import os
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from age_gender_cnn.inference import download_models, predict_photo
from age_gender_cnn.plots import plot_inference_result


ROOT = Path(__file__).parents[1]
EXAMPLE = ROOT / "assets" / "examples" / "simu-liu-beaverton.jpg"


@pytest.mark.integration
def test_public_models_are_cached_and_predict_on_licensed_example() -> None:
    tensorflow = pytest.importorskip("tensorflow")
    import keras

    del tensorflow
    cache_dir = Path(os.environ.get("AGE_GENDER_MODEL_CACHE", "models"))
    paths = download_models(cache_dir)

    def fail_if_downloaded_again(**_kwargs):
        raise AssertionError("cached model was downloaded again")

    cached_paths = download_models(cache_dir, downloader=fail_if_downloaded_again)
    assert cached_paths == paths

    models = {
        key: keras.models.load_model(path, compile=False)
        for key, path in paths.items()
    }
    for model in models.values():
        assert model.input_shape == (None, 128, 128, 3)
        assert model.output_names == ["gender_output", "age_output"]
    result = predict_photo(models, EXAMPLE)

    assert result.original_rgb.ndim == 3
    assert result.crop.rgb.ndim == 3
    assert [prediction.model_key for prediction in result.predictions] == [
        "model_a",
        "model_b",
    ]
    for prediction in result.predictions:
        assert isinstance(prediction.age_years, float)
        assert isinstance(prediction.gender_score, float)
        assert 0.0 <= prediction.gender_score <= 1.0

    figure = plot_inference_result(result)
    assert len(figure.axes) == 3
    plt.close(figure)
