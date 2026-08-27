from copy import deepcopy
from pathlib import Path

from nbclient import NotebookClient
import nbformat


ROOT = Path(__file__).parents[1]
NOTEBOOK = ROOT / "notebooks" / "Age_Gender_Multitask_CNN.ipynb"


def test_reference_mode_executes_without_data_models_or_interaction() -> None:
    notebook = deepcopy(nbformat.read(NOTEBOOK, as_version=4))
    notebook.cells = [
        cell
        for cell in notebook.cells
        if not (
            {"interactive", "training"}
            & set(cell.metadata.get("tags", []))
        )
    ]
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(ROOT)}},
    )
    executed = client.execute()
    errors = [
        output
        for cell in executed.cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
