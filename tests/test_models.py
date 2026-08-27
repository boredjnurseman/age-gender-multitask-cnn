import pytest

tf = pytest.importorskip("tensorflow")

from age_gender_cnn.models import build_model_a, build_model_b, set_model_b_fine_tuning


@pytest.mark.integration
def test_model_a_has_required_input_and_named_scalar_outputs() -> None:
    model = build_model_a()
    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_names == ["gender_output", "age_output"]
    assert [shape[-1] for shape in model.output_shape] == [1, 1]
    assert not any(isinstance(layer, tf.keras.layers.Lambda) for layer in model.layers)
    assert model.get_layer("block_3_pool").output.shape[1:3] == (8, 8)


@pytest.mark.integration
def test_model_b_uses_resnet50v2_and_freezes_batch_normalisation() -> None:
    model = build_model_b(weights=None)
    backbone = model.get_layer("resnet50v2")
    assert model.input_shape == (None, 128, 128, 3)
    assert model.output_names == ["gender_output", "age_output"]
    assert backbone.trainable is False
    set_model_b_fine_tuning(model, trainable_tail=30)
    assert backbone.trainable is True
    assert all(
        not layer.trainable
        for layer in backbone.layers
        if isinstance(layer, tf.keras.layers.BatchNormalization)
    )
