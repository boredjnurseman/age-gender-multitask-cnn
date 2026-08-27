from pathlib import Path


ROOT = Path(__file__).parents[1]
README = ROOT / "README.md"
MODEL_CARD = ROOT / "MODEL_CARD.md"
DATA_README = ROOT / "data" / "README.md"


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_public_documents_exist() -> None:
    assert README.is_file()
    assert MODEL_CARD.is_file()
    assert DATA_README.is_file()


def test_readme_explains_results_installation_and_execution_modes() -> None:
    text = _text(README)
    for required in (
        "Model A performed better on the recorded evaluation",
        "6.5747",
        "0.901",
        "2.8543",
        "7.2968",
        "0.889",
        "2.962",
        "pip install",
        "RUN_TRAINING_PIPELINE = False",
        "RUN_TRAINING_PIPELINE = True",
        "predict_photo",
        "1AfnCchElx08FN0jGGLz91tl19YKWVP09",
        "1TdKyD8Bo7tByfuJxQbJUbnuak4tERydN",
        "notebooks/",
        "src/age_gender_cnn/",
        "Gradio",
        "Streamlit",
        "not currently implemented",
    ):
        assert required in text


def test_model_card_records_scope_policy_metrics_and_risks() -> None:
    text = _text(MODEL_CARD)
    for required in (
        "Intended use",
        "Out-of-scope use",
        "custom residual CNN",
        "ResNet50V2",
        "frozen",
        "last 30 layers",
        "Huber(delta=6.5)",
        "Huber(delta=1.0)",
        "6.5747",
        "0.901",
        "2.8543",
        "7.2968",
        "0.889",
        "2.962",
        "binary gender label",
        "fairness",
        "Privacy",
        "1AfnCchElx08FN0jGGLz91tl19YKWVP09",
        "1TdKyD8Bo7tByfuJxQbJUbnuak4tERydN",
        "29 MB",
        "217 MB",
    ):
        assert required in text


def test_data_readme_documents_source_terms_and_private_training_data() -> None:
    text = _text(DATA_README)
    for required in (
        "UTKFace",
        "[age]_[gender]_[race]_[date&time].jpg",
        "non-commercial research",
        "private",
        "train_val",
        "must not be redistributed",
        "https://susanqq.github.io/UTKFace/",
        "Simu Liu on The Beaverton",
        "CC BY 3.0",
    ):
        assert required in text


def test_public_documents_do_not_expose_local_paths_or_assignment_scaffolding() -> None:
    text = "\n".join(_text(path) for path in (README, MODEL_CARD, DATA_README))
    for banned in (
        "/Users/",
        "/content/",
        "Your Tasks",
        "DON'T use any other name",
        "MobileNetV2",
    ):
        assert banned not in text
