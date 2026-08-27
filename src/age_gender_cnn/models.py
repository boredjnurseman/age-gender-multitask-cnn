"""Keras model builders for the shared-trunk, two-head prediction experiment."""

from __future__ import annotations

from typing import Any

from .data import build_augmentation_layers


def _keras(keras_module: Any | None) -> Any:
    """Use an injected Keras module in tests, or import the project runtime lazily."""
    if keras_module is not None:
        return keras_module
    import keras

    return keras


def _residual_block(x: Any, filters: int, dropout: float, name: str, keras: Any) -> Any:
    """Apply two convolutions, a projected shortcut when needed, pooling, and regularising dropout."""
    layers = keras.layers
    shortcut = x
    residual = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv_1")(x)
    residual = layers.BatchNormalization(name=f"{name}_bn_1")(residual)
    residual = layers.Activation("gelu", name=f"{name}_gelu_1")(residual)
    residual = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv_2")(
        residual
    )
    residual = layers.BatchNormalization(name=f"{name}_bn_2")(residual)
    if int(shortcut.shape[-1]) != filters:
        shortcut = layers.Conv2D(
            filters, 1, padding="same", name=f"{name}_projection"
        )(shortcut)
        shortcut = layers.BatchNormalization(name=f"{name}_projection_bn")(shortcut)
    x = layers.Add(name=f"{name}_add")([residual, shortcut])
    x = layers.Activation("gelu", name=f"{name}_gelu_2")(x)
    x = layers.Dropout(dropout, name=f"{name}_dropout")(x)
    return layers.MaxPooling2D(name=f"{name}_pool")(x)


def build_model_a(*, keras_module: Any | None = None) -> Any:
    """Build Model A: a task-specific residual CNN with shared features and two outputs."""
    keras = _keras(keras_module)
    layers, regularizers = keras.layers, keras.regularizers
    inputs = keras.Input((128, 128, 3), name="image")
    # Keep augmentation in the graph: fit() samples variants, while predict() is deterministic.
    x = build_augmentation_layers(keras)(inputs)
    x = layers.Conv2D(32, 3, padding="same", name="stem_conv")(x)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.Activation("gelu", name="stem_gelu")(x)
    x = layers.MaxPooling2D(name="stem_pool")(x)
    x = _residual_block(x, 32, 0.05, "block_1", keras)
    x = _residual_block(x, 64, 0.10, "block_2", keras)
    x = _residual_block(x, 128, 0.15, "block_3", keras)
    x = layers.Flatten(name="shared_flatten")(x)
    x = layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="shared_dense",
    )(x)
    x = layers.BatchNormalization(name="shared_bn")(x)
    x = layers.Dropout(0.20, name="shared_dropout")(x)
    # A shared trunk supports both tasks; separate heads preserve task-specific capacity.
    gender = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="gender_hidden",
    )(x)
    age = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="age_hidden",
    )(x)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender_output")(
        gender
    )
    age_output = layers.Dense(1, activation="linear", name="age_output")(age)
    return keras.Model(
        inputs, [gender_output, age_output], name="model_a_residual"
    )


def build_model_b(
    *, weights: str | None = "imagenet", keras_module: Any | None = None
) -> Any:
    """Build Model B with a frozen ResNet50V2 backbone and shared multitask prediction heads."""
    keras = _keras(keras_module)
    layers, regularizers = keras.layers, keras.regularizers
    backbone = keras.applications.ResNet50V2(
        include_top=False,
        weights=weights,
        input_shape=(128, 128, 3),
    )
    backbone.trainable = False
    inputs = keras.Input((128, 128, 3), name="image")
    # Apply the same input-space augmentation before ResNet's [-1, 1] preprocessing.
    x = build_augmentation_layers(keras)(inputs)
    x = layers.Rescaling(scale=2.0, offset=-1.0, name="resnet_preprocessing")(x)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="shared_pool")(x)
    x = layers.Dense(
        256,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="shared_dense",
    )(x)
    x = layers.BatchNormalization(name="shared_bn")(x)
    x = layers.Dropout(0.30, name="shared_dropout")(x)
    gender = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="gender_hidden",
    )(x)
    age = layers.Dense(
        64,
        activation="gelu",
        kernel_regularizer=regularizers.l2(1e-4),
        name="age_hidden",
    )(x)
    gender_output = layers.Dense(1, activation="sigmoid", name="gender_output")(
        gender
    )
    age_output = layers.Dense(1, activation="linear", name="age_output")(age)
    return keras.Model(
        inputs, [gender_output, age_output], name="model_b_resnet50v2"
    )


def set_model_b_fine_tuning(model: Any, trainable_tail: int = 30) -> None:
    """Unfreeze only the ResNet tail while keeping Batch Normalisation statistics fixed during tuning."""
    import keras

    backbone = model.get_layer("resnet50v2")
    backbone.trainable = True
    for layer in backbone.layers[:-trainable_tail]:
        layer.trainable = False
    for layer in backbone.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
