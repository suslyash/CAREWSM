"""Lightweight checks for the inactive CAREWSM package scaffold."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2] / "src"


def test_carewsm_namespace_imports_without_activation() -> None:
    sys.path.insert(0, str(SRC_ROOT))
    try:
        package = importlib.import_module("carewsm")
        assert package.__name__ == "carewsm"
        for component in ("data", "features", "models", "evaluation"):
            module = importlib.import_module(f"carewsm.{component}")
            assert module.__name__ == f"carewsm.{component}"
    finally:
        sys.path.remove(str(SRC_ROOT))


def test_scaffold_contains_no_active_components() -> None:
    package_root = SRC_ROOT / "carewsm"
    assert (package_root / "__init__.py").is_file()
    for component in ("data", "features", "models", "evaluation"):
        assert (package_root / component / "__init__.py").is_file()
