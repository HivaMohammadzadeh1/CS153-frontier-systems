"""Smoke test: the streamlit app module imports cleanly without runtime errors."""

import importlib


def test_app_module_imports():
    """Ensure all imports and top-level statements in scripts/app.py work."""
    mod = importlib.import_module("scripts.app")
    assert hasattr(mod, "main")
