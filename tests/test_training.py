from pathlib import Path

import pytest

from age_gender_cnn import training
from age_gender_cnn.data import DatasetArrays
from age_gender_cnn.training import (
    MODEL_A_COMPILE,
    MODEL_B_COMPILE,
    checkpoint_callbacks,
    compile_multitask,
    combine_histories,
    evaluate_checkpoints,
    select_checkpoint,
    train_model_a,
    train_model_b,
)


def test_compile_settings_preserve_final_coursework_choices() -> None:
    assert MODEL_A_COMPILE.age_huber_delta == 6.5
    assert MODEL_B_COMPILE.age_huber_delta == 1.0
    assert MODEL_A_COMPILE.loss_weights == {
        "gender_output": 3.0,
        "age_output": 0.30,
    }
    assert MODEL_B_COMPILE.loss_weights == MODEL_A_COMPILE.loss_weights


def test_combine_histories_concatenates_matching_series() -> None:
    combined = combine_histories(
        {"loss": [3.0, 2.0], "accuracy": [0.6, 0.7]},
        {"loss": [1.5], "accuracy": [0.8], "new_key": [4.0]},
    )
    assert combined == {
        "loss": [3.0, 2.0, 1.5],
        "accuracy": [0.6, 0.7, 0.8],
    }


def test_select_checkpoint_uses_explicit_name_not_best_looking_metric() -> None:
    records = {
        "tuned_best_val_loss": {"loss": 2.9620, "gender_output_accuracy": 0.889},
        "tuned_best_gender": {"loss": 3.0143, "gender_output_accuracy": 0.896},
    }
    assert select_checkpoint(records, "tuned_best_val_loss") == records[
        "tuned_best_val_loss"
    ]


class FakeModel:
    def __init__(self, score: float) -> None:
        self.score = score

    def evaluate(self, *_args, **_kwargs):
        return {"loss": self.score, "age_output_mae": self.score + 1.0}


def test_evaluate_checkpoints_returns_named_records(tmp_path: Path) -> None:
    paths = {
        "best_loss": tmp_path / "best.keras",
        "best_age": tmp_path / "age.keras",
    }
    for path in paths.values():
        path.write_bytes(b"fixture")
    scores = iter((2.0, 3.0))
    records = evaluate_checkpoints(
        paths,
        load_model=lambda _path: FakeModel(next(scores)),
        validation_images=object(),
        validation_targets={"gender_output": object(), "age_output": object()},
    )
    assert records["best_loss"]["loss"] == 2.0
    assert records["best_age"]["age_output_mae"] == 4.0


@pytest.mark.integration
def test_compile_multitask_applies_loss_weight_and_huber_behaviour() -> None:
    import keras
    import numpy as np

    inputs = keras.Input((1,))
    gender = keras.layers.Dense(1, name="gender_output")(inputs)
    age = keras.layers.Dense(1, name="age_output")(inputs)
    model = keras.Model(inputs, [gender, age])

    compile_multitask(model, MODEL_A_COMPILE)

    compile_config = model.get_compile_config()
    assert compile_config["loss_weights"] == {
        "gender_output": 3.0,
        "age_output": 0.30,
    }
    age_loss = model.loss["age_output"](
        np.asarray([0.0], dtype=np.float32),
        np.asarray([10.0], dtype=np.float32),
    )
    assert float(age_loss) == pytest.approx(43.875)


@pytest.mark.integration
def test_callbacks_accept_the_final_phase_specific_schedule(tmp_path: Path) -> None:
    callbacks = checkpoint_callbacks(
        tmp_path,
        "model_a",
        early_stopping_patience=12,
        reduce_lr_patience=5,
        min_lr=1e-6,
    )
    early_stopping, reduce_lr = callbacks[-2:]
    assert early_stopping.patience == 12
    assert reduce_lr.patience == 5
    assert reduce_lr.min_lr == 1e-6


def _tiny_multitask_model():
    import keras

    inputs = keras.Input((1,), name="image")
    shared = keras.layers.Dense(2, activation="relu")(inputs)
    gender = keras.layers.Dense(1, activation="sigmoid", name="gender_output")(
        shared
    )
    age = keras.layers.Dense(1, name="age_output")(shared)
    return keras.Model(inputs, [gender, age])


def _tiny_dataset() -> DatasetArrays:
    import numpy as np

    return DatasetArrays(
        images=np.asarray([[0.0], [0.3], [0.6], [1.0]], dtype=np.float32),
        ages=np.asarray([20.0, 30.0, 40.0, 50.0], dtype=np.float32),
        genders=np.asarray([0.0, 0.0, 1.0, 1.0], dtype=np.float32),
    )


@pytest.mark.integration
def test_train_model_a_returns_its_selected_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "build_model_a", _tiny_multitask_model)
    dataset = _tiny_dataset()

    selected, history = train_model_a(
        dataset, dataset, tmp_path, epochs=1, batch_size=2
    )

    assert selected.output_names == ["gender_output", "age_output"]
    assert len(history["loss"]) == 1
    assert (tmp_path / "model_a_best_val_loss.keras").is_file()


@pytest.mark.integration
def test_train_model_b_combines_both_phases_and_selects_tuned_loss(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(training, "build_model_b", _tiny_multitask_model)
    monkeypatch.setattr(training, "set_model_b_fine_tuning", lambda *_args, **_kwargs: None)
    dataset = _tiny_dataset()

    selected, history, fine_tune_start = train_model_b(
        dataset,
        dataset,
        tmp_path,
        frozen_epochs=1,
        tuned_epochs=1,
        batch_size=2,
    )

    assert selected.output_names == ["gender_output", "age_output"]
    assert len(history["loss"]) == 2
    assert fine_tune_start == 1
    assert (tmp_path / "model_b_tuned_best_val_loss.keras").is_file()
