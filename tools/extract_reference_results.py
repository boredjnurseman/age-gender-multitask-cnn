from __future__ import annotations

import argparse
from base64 import b64decode
from datetime import date
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

import nbformat


ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
CHECKPOINT_NAME = re.compile(r"^[ ]*([a-z][a-z0-9_]*)[ ]*$")
METRIC_LINE = re.compile(r"^([a-z][a-z0-9_]*):[ ]*(-?[0-9]+(?:\.[0-9]+)?)$")


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""
    return sha256(path.read_bytes()).hexdigest()


def cell_text(cell: Any) -> str:
    """Join stream and text/plain output from one notebook cell."""
    chunks: list[str] = []
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            value = output.get("text", "")
        else:
            value = output.get("data", {}).get("text/plain", "")
        chunks.append("".join(value) if isinstance(value, list) else str(value))
    return ANSI_ESCAPE.sub("", "\n".join(chunks))


def parse_checkpoint_metrics(text: str) -> dict[str, dict[str, float]]:
    """Parse named Keras checkpoint-evaluation blocks from stream text."""
    records: dict[str, dict[str, float]] = {}
    current: str | None = None
    for raw_line in text.splitlines():
        line = raw_line.strip("\r")
        name_match = CHECKPOINT_NAME.fullmatch(line)
        metric_match = METRIC_LINE.fullmatch(line.strip())
        if metric_match and current is not None:
            records[current][metric_match.group(1)] = float(metric_match.group(2))
        elif name_match and "_" in name_match.group(1):
            current = name_match.group(1)
            records.setdefault(current, {})
        elif not line.strip():
            current = None
    return {name: values for name, values in records.items() if values}


def extract_png(cell: Any, target: Path) -> None:
    """Write the first image/png display output from a notebook cell."""
    for output in cell.get("outputs", []):
        encoded = output.get("data", {}).get("image/png")
        if encoded:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(
                b64decode("".join(encoded) if isinstance(encoded, list) else encoded)
            )
            return
    raise ValueError(f"Cell has no image/png output for {target.name}")


METRIC_CELLS = {"model_a": 12, "model_b": 25}
CURVE_CELLS = {
    "model_a_gender_loss": 15,
    "model_a_gender_accuracy": 16,
    "model_a_age_loss": 17,
    "model_a_age_mae": 18,
    "model_b_gender_loss": 29,
    "model_b_gender_accuracy": 30,
    "model_b_age_loss": 31,
    "model_b_age_mae": 32,
}
SELECTED = {"model_a": "best_val_loss", "model_b": "tuned_best_val_loss"}


def extract_reference_results(source_notebook: Path, output_dir: Path) -> dict[str, Any]:
    """Extract selected metrics and rendered curves with source provenance."""
    notebook = nbformat.read(source_notebook, as_version=4)
    output_dir.mkdir(parents=True, exist_ok=True)
    all_metrics: dict[str, Any] = {}
    mime_types: dict[str, list[str]] = {}

    for model_name, cell_index in METRIC_CELLS.items():
        cell = notebook.cells[cell_index]
        candidates = parse_checkpoint_metrics(cell_text(cell))
        selected_name = SELECTED[model_name]
        if selected_name not in candidates:
            raise ValueError(f"Missing {selected_name} in source cell {cell_index}")
        all_metrics[model_name] = {
            "selected_checkpoint": selected_name,
            "selected": candidates[selected_name],
            "candidates": candidates,
        }
        mime_types[str(cell_index)] = [
            output.get("output_type", "") for output in cell.outputs
        ]

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(all_metrics, indent=2) + "\n")

    curve_paths: list[Path] = []
    curve_dir = output_dir / "training_curves"
    for name, cell_index in CURVE_CELLS.items():
        target = curve_dir / f"{name}.png"
        extract_png(notebook.cells[cell_index], target)
        curve_paths.append(target)
        mime_types[str(cell_index)] = ["image/png"]

    manifest = {
        "source_filename": source_notebook.name,
        "source_sha256": sha256_file(source_notebook),
        "extraction_date": date.today().isoformat(),
        "metric_cells": METRIC_CELLS,
        "curve_cells": CURVE_CELLS,
        "output_mime_types": mime_types,
        "known_report_discrepancy": {
            "model_a_report_age_mae": 6.8488,
            "model_a_report_gender_accuracy": 0.9060,
            "policy": "Display the final notebook's structured checkpoint evaluation.",
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "metrics": metrics_path,
        "manifest": manifest_path,
        "curve_paths": curve_paths,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_notebook", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    extract_reference_results(args.source_notebook, args.output_dir)


if __name__ == "__main__":
    main()
