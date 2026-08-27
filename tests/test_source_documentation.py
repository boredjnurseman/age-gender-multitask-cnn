import ast
from pathlib import Path


ROOT = Path(__file__).parents[1]


EXPECTED_DOCUMENTED_SYMBOLS = {
    "data.py": (
        "FaceLabel",
        "DatasetSplit",
        "DatasetArrays",
        "parse_utkface_filename",
        "list_image_paths",
        "split_paths",
        "load_rgb_image",
        "load_dataset",
        "build_augmentation_layers",
    ),
    "models.py": (
        "_residual_block",
        "build_model_a",
        "build_model_b",
        "set_model_b_fine_tuning",
    ),
    "training.py": (
        "CompileConfig",
        "combine_histories",
        "select_checkpoint",
        "compile_multitask",
        "checkpoint_callbacks",
        "evaluate_checkpoints",
        "train_model_a",
        "train_model_b",
    ),
    "inference.py": (
        "ModelSpec",
        "FaceCrop",
        "ModelPrediction",
        "InferenceResult",
        "detect_and_crop_face",
        "prepare_model_batch",
        "download_models",
        "register_upload_bytes",
        "map_named_outputs",
        "predict_photo",
    ),
    "plots.py": (
        "plot_augmentation_grid",
        "plot_reference_curves",
        "plot_selected_metrics",
        "plot_inference_result",
    ),
}


def _top_level_definitions(path: Path) -> dict[str, ast.AST]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    }


def test_public_modules_and_symbols_explain_their_contracts() -> None:
    for filename, symbols in EXPECTED_DOCUMENTED_SYMBOLS.items():
        path = ROOT / "src" / "age_gender_cnn" / filename
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert ast.get_docstring(tree), f"Missing module docstring in {filename}"
        definitions = _top_level_definitions(path)
        for symbol in symbols:
            node = definitions.get(symbol)
            assert node is not None, f"Missing {symbol} in {filename}"
            docstring = ast.get_docstring(node)
            assert docstring and len(docstring.split()) >= 12, (
                f"Docstring for {filename}:{symbol} is too sparse"
            )
