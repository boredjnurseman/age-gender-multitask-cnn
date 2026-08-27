from pathlib import Path

import nbformat


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "Age_Gender_Multitask_CNN.ipynb"


def _notebook():
    return nbformat.read(NOTEBOOK, as_version=4)


def test_notebook_uses_public_narrative_and_safe_default() -> None:
    notebook = _notebook()
    markdown = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    code = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    for heading in (
        "# Custom residual CNN versus transfer learning",
        "## Project question",
        "## Data and multi-task formulation",
        "## Model A: a task-specific residual CNN",
        "## Model B: ResNet50V2 transfer learning",
        "## What the validation comparison showed",
        "## Discussion",
        "## Try both models on one photograph",
        "## Limitations",
    ):
        assert heading in markdown
    for phrase in (
        "### Implementation map",
        "src/age_gender_cnn/data.py",
        "src/age_gender_cnn/models.py",
        "src/age_gender_cnn/training.py",
        "src/age_gender_cnn/inference.py",
        "TensorFlow/Keras",
        "residual connection",
        "backbone",
        "empirical",
        "\\mathrm{MAE}",
        "\\mathcal{L}",
        "Caruana (1997)",
        "He et al. (2016)",
        "Shorten and Khoshgoftaar (2019)",
        "Yosinski et al. (2014)",
        "0.7221 years",
        "1.2 percentage points",
        "learning curves",
        "## References",
    ):
        assert phrase in markdown
    assert "RUN_TRAINING_PIPELINE = False" in code
    assert "MobileNetV2" not in code + markdown


def test_checkpoint_selection_is_kept_inside_discussion() -> None:
    notebook = _notebook()
    discussion_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "markdown" and cell.source.startswith("## Discussion")
    ]
    assert len(discussion_cells) == 1
    assert "### Checkpoint selection" in discussion_cells[0].source
    assert not any(
        cell.cell_type == "markdown"
        and cell.source.startswith("### Checkpoint selection")
        for cell in notebook.cells
    )


def test_notebook_contains_no_assignment_or_private_scaffolding() -> None:
    text = NOTEBOOK.read_text()
    for banned in (
        "Your Tasks",
        "DON'T use any other name",
        "my test code",
        "/Users/",
        "/content/drive/MyDrive/CM500335",
        "Visualize a few photos",
    ):
        assert banned not in text


def test_upload_and_training_cells_are_explicitly_tagged() -> None:
    notebook = _notebook()
    tags = [set(cell.metadata.get("tags", [])) for cell in notebook.cells]
    assert any("interactive" in cell_tags for cell_tags in tags)
    assert any("training" in cell_tags for cell_tags in tags)


def test_upload_cell_shows_consent_warning_and_rejects_duplicate_bytes() -> None:
    notebook = _notebook()
    markdown = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    ).lower()
    interactive = next(
        cell
        for cell in notebook.cells
        if "interactive" in cell.metadata.get("tags", [])
    )
    source = "".join(interactive.source).lower()
    assert "consent" in markdown
    assert "register_upload_bytes" in source
    assert "duplicate upload" in source


def test_augmentation_demo_uses_the_training_layer_on_a_face_centred_example() -> None:
    notebook = _notebook()
    markdown = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    ).lower()
    code = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "code"
    )
    assert "crop the face tightly" in markdown
    assert "assets/examples/simu-liu-beaverton.jpg" in code
    assert "build_augmentation_layers" in code
    assert "training=True" in code


def test_executed_figures_are_embedded_once_without_tagged_outputs() -> None:
    notebook = _notebook()
    metrics_cell = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "plot_selected_metrics" in cell.source
    )
    assert len(metrics_cell.outputs) == 4
    markdown = "\n".join(
        "".join(cell.source)
        for cell in notebook.cells
        if cell.cell_type == "markdown"
    )
    assert "<table>" in markdown
    assert "Model A" in markdown and "Model B" in markdown
    for cell in notebook.cells:
        if {"interactive", "training"} & set(cell.metadata.get("tags", [])):
            assert cell.execution_count is None
            assert cell.outputs == []
