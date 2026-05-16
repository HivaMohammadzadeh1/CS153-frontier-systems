"""Smoke tests for the diagnostic chat flow added to scripts/app.py.

These tests verify that module-level constants and key symbols are present
and importable. They do NOT run the interactive diagnostic loop (which requires
a live LLM + DB) or start the Streamlit server.
"""

import importlib


def test_app_module_imports():
    """Ensure all imports and top-level statements in scripts/app.py work."""
    mod = importlib.import_module("scripts.app")
    assert hasattr(mod, "main")


def test_diagnostic_schema_defined():
    """DIAGNOSTIC_SCHEMA must be defined and have the required keys."""
    mod = importlib.import_module("scripts.app")
    schema = mod.DIAGNOSTIC_SCHEMA
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    props = schema.get("properties", {})
    assert "diagnosis" in props, "DIAGNOSTIC_SCHEMA must include 'diagnosis'"
    assert "follow_up_question" in props, "DIAGNOSTIC_SCHEMA must include 'follow_up_question'"
    assert set(schema.get("required", [])) >= {"diagnosis", "follow_up_question"}


def test_explain_schema_defined():
    """EXPLAIN_SCHEMA must be defined and have the required keys."""
    mod = importlib.import_module("scripts.app")
    schema = mod.EXPLAIN_SCHEMA
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
    props = schema.get("properties", {})
    for key in ("confirmed_misconception", "explanation", "next_action", "next_message"):
        assert key in props, f"EXPLAIN_SCHEMA missing '{key}'"
    assert set(schema.get("required", [])) >= {
        "confirmed_misconception", "explanation", "next_action", "next_message"
    }
    # next_action must be an enum with the three valid values
    next_action_enum = props["next_action"].get("enum", [])
    assert set(next_action_enum) == {"explain", "re_test", "wrap_up"}


def test_diagnostic_threshold_constant():
    """DIAGNOSTIC_THRESHOLD must exist and be a float between 0 and 1."""
    mod = importlib.import_module("scripts.app")
    threshold = mod.DIAGNOSTIC_THRESHOLD
    assert isinstance(threshold, float), "DIAGNOSTIC_THRESHOLD must be a float"
    assert 0.0 < threshold < 1.0, "DIAGNOSTIC_THRESHOLD must be between 0 and 1"


def test_diagnostic_max_turns_constant():
    """DIAGNOSTIC_MAX_TURNS must exist and be a positive integer."""
    mod = importlib.import_module("scripts.app")
    max_turns = mod.DIAGNOSTIC_MAX_TURNS
    assert isinstance(max_turns, int), "DIAGNOSTIC_MAX_TURNS must be an int"
    assert max_turns > 0, "DIAGNOSTIC_MAX_TURNS must be positive"


def test_diagnostic_helper_functions_exist():
    """Key diagnostic helper functions must be importable from scripts.app."""
    mod = importlib.import_module("scripts.app")
    for fn_name in (
        "_generate_diagnostic_question",
        "_generate_explanation",
        "_generate_retest_question",
        "_record_misconception_to_db",
        "_render_diagnostic_flow",
    ):
        assert hasattr(mod, fn_name), f"Missing function: {fn_name}"
