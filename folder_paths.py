"""
Model discovery helpers.
"""
import os
from pathlib import Path

import folder_paths as comfy_paths


ARDY_FOLDER = "ardy"


def get_ardy_directory():

    path = Path(
        comfy_paths.models_dir
    ) / ARDY_FOLDER

    path.mkdir(
        exist_ok=True,
        parents=True
    )

    return path



def get_model_list():

    root = get_ardy_directory()

    models = []

    for item in root.iterdir():

        if item.is_dir() and (item / "config.yaml").is_file():

            models.append(
                item.name
            )

    return sorted(models)



def get_model_path(name):
    root = get_ardy_directory().resolve()
    path = (root / name).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("ARDY model must be inside the models/ardy directory.")
    if not (path / "config.yaml").is_file():
        raise FileNotFoundError(f"ARDY config.yaml not found in {path}")
    return path
