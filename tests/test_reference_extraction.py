from base64 import b64encode
from datetime import date
from hashlib import sha256
import json
from pathlib import Path

import nbformat
import pytest

from tools.extract_reference_results import (
    extract_reference_results,
    extract_png,
    parse_checkpoint_metrics,
    sha256_file,
)


def test_sha256_file_hashes_exact_notebook_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.ipynb"
    source.write_bytes(b"executed notebook")
    assert sha256_file(source) == sha256(b"executed notebook").hexdigest()


def test_parse_checkpoint_metrics_reads_named_blocks() -> None:
    text = """
 best_val_loss
age_output_mae: 6.5747
gender_output_accuracy: 0.9010
loss: 2.8543

 best_gender
age_output_mae: 7.1144
gender_output_accuracy: 0.9120
loss: 3.0397
"""
    parsed = parse_checkpoint_metrics(text)
    assert parsed["best_val_loss"]["age_output_mae"] == 6.5747
    assert parsed["best_gender"]["gender_output_accuracy"] == 0.912


def test_extract_png_decodes_display_output(tmp_path: Path) -> None:
    payload = b"\x89PNG\r\n\x1a\nfixture"
    cell = nbformat.v4.new_code_cell()
    cell.outputs = [
        nbformat.v4.new_output(
            "display_data",
            data={"image/png": b64encode(payload).decode("ascii")},
        )
    ]
    target = tmp_path / "figure.png"
    extract_png(cell, target)
    assert target.read_bytes() == payload


SOURCE = next(
    (
        ancestor / "age_gender_submit.ipynb"
        for ancestor in Path(__file__).parents
        if (ancestor / "age_gender_submit.ipynb").exists()
    ),
    Path("/missing/age_gender_submit.ipynb"),
)


@pytest.mark.skipif(not SOURCE.exists(), reason="canonical notebook is outside the public repo")
def test_canonical_extraction_preserves_selected_results(tmp_path: Path) -> None:
    outputs = extract_reference_results(SOURCE, tmp_path)
    metrics = json.loads(outputs["metrics"].read_text())
    manifest = json.loads(outputs["manifest"].read_text())

    assert metrics["model_a"]["selected"]["age_output_mae"] == 6.5747
    assert metrics["model_a"]["selected"]["gender_output_accuracy"] == 0.901
    assert metrics["model_b"]["selected"]["age_output_mae"] == 7.2968
    assert metrics["model_b"]["selected"]["gender_output_accuracy"] == 0.889
    assert manifest["source_sha256"] == (
        "dc420bdfa421e5d10ba60de8b0392e864642a75e29fcb971d8082e590c30b698"
    )
    assert date.fromisoformat(manifest["extraction_date"]) <= date.today()
    assert len(outputs["curve_paths"]) == 8
