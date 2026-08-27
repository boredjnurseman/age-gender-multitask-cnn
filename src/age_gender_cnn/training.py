"""Compilation, checkpointing, and optional two-phase training for both models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .data import DatasetArrays
from .models import build_model_a, build_model_b, set_model_b_fine_tuning


@dataclass(frozen=True)
class CompileConfig:
    """Record optimizer, age-loss, and task-weight choices for one reproducible multitask training phase."""

    learning_rate: float
    age_huber_delta: float
    loss_weights: dict[str, float]


MODEL_A_COMPILE = CompileConfig(
    1e-3, 6.5, {"gender_output": 3.0, "age_output": 0.30}
)
MODEL_B_COMPILE = CompileConfig(
    1e-3, 1.0, {"gender_output": 3.0, "age_output": 0.30}
)


def combine_histories(
    *histories: Mapping[str, Sequence[float]],
) -> dict[str, list[float]]:
    """Concatenate metrics shared by every phase while preserving their original training order."""
    if not histories:
        return {}
    shared = set.intersection(*(set(history) for history in histories))
    return {
        key: [value for history in histories for value in history[key]]
        for key in sorted(shared)
    }


def select_checkpoint(
    records: Mapping[str, dict[str, float]], selected_name: str
) -> dict[str, float]:
    """Return the named checkpoint record after requiring an explicit documented selection policy."""
    if selected_name not in records:
        raise KeyError(f"Checkpoint {selected_name!r} was not evaluated")
    return records[selected_name]


def compile_multitask(model: Any, config: CompileConfig) -> None:
    """Compile both heads with binary cross-entropy, Huber age loss, and configured weights."""
    import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            "gender_output": keras.losses.BinaryCrossentropy(),
            "age_output": keras.losses.Huber(delta=config.age_huber_delta),
        },
        metrics={
            "gender_output": [keras.metrics.BinaryAccuracy(name="accuracy")],
            "age_output": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
        loss_weights=config.loss_weights,
    )


def checkpoint_callbacks(
    output_dir: Path,
    prefix: str,
    *,
    early_stopping_patience: int,
    reduce_lr_patience: int,
    min_lr: float,
) -> list[Any]:
    """Create checkpoint, early-stopping, and learning-rate callbacks for one monitored model training phase."""
    import keras

    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            output_dir / f"{prefix}_best_val_loss.keras",
            monitor="val_loss",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.ModelCheckpoint(
            output_dir / f"{prefix}_best_age.keras",
            monitor="val_age_output_mae",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.ModelCheckpoint(
            output_dir / f"{prefix}_best_gender.keras",
            monitor="val_gender_output_accuracy",
            mode="max",
            save_best_only=True,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=early_stopping_patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=reduce_lr_patience,
            min_lr=min_lr,
        ),
    ]


def evaluate_checkpoints(
    paths: Mapping[str, Path],
    *,
    load_model: Callable[[Path], Any],
    validation_images: Any,
    validation_targets: Mapping[str, Any],
) -> dict[str, dict[str, float]]:
    """Evaluate every named checkpoint on the same validation partition for fair comparison."""
    records: dict[str, dict[str, float]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        raw = load_model(path).evaluate(
            validation_images,
            validation_targets,
            verbose=0,
            return_dict=True,
        )
        records[name] = {key: float(value) for key, value in raw.items()}
    return records


def _targets(dataset: DatasetArrays) -> dict[str, Any]:
    """Map aligned dataset arrays to the output names expected by the Keras model."""
    return {"gender_output": dataset.genders, "age_output": dataset.ages}


def train_model_a(
    train: DatasetArrays,
    validation: DatasetArrays,
    output_dir: Path,
    *,
    epochs: int = 100,
    batch_size: int = 16,
) -> tuple[Any, dict[str, list[float]]]:
    """Train Model A, restore its balanced checkpoint, and return serialisable training history."""
    import keras

    model = build_model_a()
    compile_multitask(model, MODEL_A_COMPILE)
    callbacks = checkpoint_callbacks(
        output_dir,
        "model_a",
        early_stopping_patience=12,
        reduce_lr_patience=5,
        min_lr=1e-6,
    )
    history = model.fit(
        train.images,
        _targets(train),
        validation_data=(validation.images, _targets(validation)),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )
    selected = keras.models.load_model(output_dir / "model_a_best_val_loss.keras")
    return selected, {key: list(values) for key, values in history.history.items()}


def train_model_b(
    train: DatasetArrays,
    validation: DatasetArrays,
    output_dir: Path,
    *,
    frozen_epochs: int = 70,
    tuned_epochs: int = 30,
    batch_size: int = 16,
) -> tuple[Any, dict[str, list[float]], int]:
    """Train frozen and fine-tuned Model B phases, returning checkpoint, history, and boundary."""
    import keras

    model = build_model_b()
    compile_multitask(model, MODEL_B_COMPILE)
    base_callbacks = checkpoint_callbacks(
        output_dir,
        "model_b_base",
        early_stopping_patience=8,
        reduce_lr_patience=3,
        min_lr=1e-6,
    )
    base = model.fit(
        train.images,
        _targets(train),
        validation_data=(validation.images, _targets(validation)),
        epochs=frozen_epochs,
        batch_size=batch_size,
        callbacks=base_callbacks,
    )
    set_model_b_fine_tuning(model, trainable_tail=30)
    # Recompile after changing trainability so the optimizer tracks the new variable set.
    compile_multitask(
        model,
        CompileConfig(1e-5, 1.0, MODEL_B_COMPILE.loss_weights),
    )
    tuned_callbacks = checkpoint_callbacks(
        output_dir,
        "model_b_tuned",
        early_stopping_patience=6,
        reduce_lr_patience=2,
        min_lr=1e-7,
    )
    tuned = model.fit(
        train.images,
        _targets(train),
        validation_data=(validation.images, _targets(validation)),
        epochs=tuned_epochs,
        batch_size=batch_size,
        callbacks=tuned_callbacks,
    )
    combined = combine_histories(base.history, tuned.history)
    selected = keras.models.load_model(
        output_dir / "model_b_tuned_best_val_loss.keras"
    )
    return selected, combined, len(base.history["loss"])
