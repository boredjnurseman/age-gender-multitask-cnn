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
        "## Try both models on one photograph",
        "## Limitations",
    ):
        assert heading in markdown
    assert "RUN_TRAINING_PIPELINE = False" in code
    assert "MobileNetV2" not in code + markdown


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


def test_executed_figures_are_embedded_once_without_tagged_outputs() -> None:
    notebook = _notebook()
    architecture_cell = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "plot_architecture_comparison" in cell.source
    )
    metrics_cell = next(
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and "plot_selected_metrics" in cell.source
    )
    assert len(architecture_cell.outputs) == 1
    assert len(metrics_cell.outputs) == 4
    for cell in notebook.cells:
        if {"interactive", "training"} & set(cell.metadata.get("tags", [])):
            assert cell.execution_count is None
            assert cell.outputs == []
