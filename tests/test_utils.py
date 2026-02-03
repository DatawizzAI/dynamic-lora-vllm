"""Tests for src/utils.py (no vLLM/GPU)."""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import (
    ServerState,
    create_health_app,
    get_env_var,
    infer_tool_call_parser,
)


# --- get_env_var ---
def test_get_env_var_str_default():
    os.environ.pop("MISSING_VAR", None)
    assert get_env_var("MISSING_VAR", "default") == "default"
    assert get_env_var("MISSING_VAR", None) is None


def test_get_env_var_int():
    os.environ["TEST_INT"] = "42"
    try:
        assert get_env_var("TEST_INT", "0", int) == 42
    finally:
        os.environ.pop("TEST_INT", None)


def test_get_env_var_bool_true():
    os.environ["TEST_BOOL"] = "true"
    try:
        assert get_env_var("TEST_BOOL", "false", bool) is True
    finally:
        os.environ.pop("TEST_BOOL", None)


def test_get_env_var_bool_false():
    os.environ["TEST_BOOL"] = "false"
    try:
        assert get_env_var("TEST_BOOL", "true", bool) is False
    finally:
        os.environ.pop("TEST_BOOL", None)


# --- health check ---
def test_ping_initializing_returns_204():
    app = create_health_app(get_state=lambda: ServerState.INITIALIZING)
    from fastapi.testclient import TestClient
    r = TestClient(app).get("/ping")
    assert r.status_code == 204


def test_ping_ready_returns_200():
    app = create_health_app(get_state=lambda: ServerState.READY)
    from fastapi.testclient import TestClient
    r = TestClient(app).get("/ping")
    assert r.status_code == 200


def test_ping_error_returns_500():
    app = create_health_app(get_state=lambda: ServerState.ERROR)
    from fastapi.testclient import TestClient
    r = TestClient(app).get("/ping")
    assert r.status_code == 500


# --- infer_tool_call_parser ---
def test_tool_call_hermes():
    assert infer_tool_call_parser("nousresearch/Hermes-2-Pro-Mistral-7B") == "hermes"


def test_tool_call_mistral():
    assert infer_tool_call_parser("mistralai/Mistral-7B-Instruct-v0.2") == "mistral"


def test_tool_call_llama():
    assert infer_tool_call_parser("meta-llama/Llama-3.2-1B-Instruct") == "llama3_json"
    assert infer_tool_call_parser("meta-llama/Llama-4-8B") == "llama4_pythonic"


def test_tool_call_unknown_none():
    assert infer_tool_call_parser("Qwen/Qwen3-4B") is None
    assert infer_tool_call_parser("random/unknown-model") is None


def test_tool_call_case_insensitive():
    assert infer_tool_call_parser("MistralAI/Mistral-7B") == "mistral"
