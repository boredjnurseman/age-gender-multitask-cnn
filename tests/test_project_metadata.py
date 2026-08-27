from pathlib import Path
import tomllib


ROOT = Path(__file__).parents[1]


def test_project_metadata_names_the_import_package() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text())
    assert metadata["project"]["name"] == "age-gender-multitask-cnn"
    assert metadata["project"]["requires-python"] == ">=3.10,<3.14"


def test_licence_excludes_third_party_assets() -> None:
    licence = (ROOT / "LICENSE").read_text()
    assert "MIT License" in licence
    assert "does not cover UTKFace" in licence
    assert "trained model artifacts" in licence
    assert "CC BY 3.0 example photograph" in licence


def test_gitignore_blocks_private_and_heavy_artifacts() -> None:
    ignored = (ROOT / ".gitignore").read_text().splitlines()
    for pattern in ("train_val/", "*.keras", "models/", "uploads/", ".DS_Store"):
        assert pattern in ignored
