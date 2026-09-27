from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from llm_lab.backends.base import GenerationRequest
from llm_lab.backends.ollama_backend import (
    OllamaBackend,
    OllamaConnectionError,
    OllamaModelNotFoundError,
    OllamaTimeoutError,
)


def test_ollama_is_available_offline():
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError("Offline")):
        backend = OllamaBackend("http://invalid.host:9999")
        assert backend.is_available() is False


def test_ollama_is_available_online():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    with patch("requests.get", return_value=mock_resp):
        backend = OllamaBackend("http://127.0.0.1:11434")
        assert backend.is_available() is True


def test_ollama_list_models_mocked():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "models": [
            {
                "name": "qwen2.5:0.5b",
                "size": 397000000,
                "details": {"parameter_size": "0.5B", "quantization_level": "Q4_K_M"},
            },
            {
                "name": "qwen3:4b",
                "size": 2500000000,
                "details": {"parameter_size": "4.0B", "quantization_level": "Q4_K_M"},
            },
        ]
    }

    with patch("requests.get", return_value=mock_resp):
        backend = OllamaBackend()
        models = backend.list_models()
        assert len(models) == 2
        assert models[0]["name"] == "qwen2.5:0.5b"
        assert models[1]["details"]["parameter_size"] == "4.0B"


def test_ollama_generate_metrics_mocked():
    """Verify that streaming line responses properly parse TTFT, total latency,

    and server reported metrics.
    """
    stream_lines = [
        json.dumps({"response": "Hello"}).encode("utf-8"),
        json.dumps({"response": " world"}).encode("utf-8"),
        json.dumps({
            "response": "!",
            "done": True,
            "load_duration": 50_000_000,        # 50 ms
            "prompt_eval_count": 10,
            "prompt_eval_duration": 20_000_000, # 20 ms
            "eval_count": 3,
            "eval_duration": 30_000_000,        # 30 ms (100 tok/s)
        }).encode("utf-8"),
    ]

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.iter_lines.return_value = stream_lines
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("requests.post", return_value=mock_resp):
        backend = OllamaBackend()
        req = GenerationRequest(prompt="hi", max_tokens=10)
        res = backend.generate(req, model_name="qwen2.5:0.5b")

        assert res.backend == "ollama"
        assert res.text == "Hello world!"
        assert res.output_tokens == 3
        assert res.prompt_tokens == 10
        assert res.client_ttft_ms >= 0.0
        assert res.client_total_latency_ms >= 0.0
        assert res.server_load_duration_ms == 50.0
        assert res.server_prompt_eval_duration_ms == 20.0
        assert res.server_eval_duration_ms == 30.0
        assert res.server_eval_tokens_per_sec == pytest.approx(100.0, rel=1e-2)


def test_ollama_model_not_found_error():
    mock_resp = MagicMock()
    mock_resp.status_code = 404
    mock_resp.__enter__.return_value = mock_resp
    mock_resp.__exit__.return_value = None

    with patch("requests.post", return_value=mock_resp):
        backend = OllamaBackend()
        req = GenerationRequest(prompt="test")
        with pytest.raises(OllamaModelNotFoundError, match="not found"):
            backend.generate(req, model_name="nonexistent:model")


def test_ollama_timeout_error():
    with patch("requests.post", side_effect=requests.exceptions.Timeout("Timed out")):
        backend = OllamaBackend()
        req = GenerationRequest(prompt="test")
        with pytest.raises(OllamaTimeoutError, match="timed out"):
            backend.generate(req, model_name="qwen2.5:0.5b")


def test_ollama_connection_error():
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Refused")):
        backend = OllamaBackend()
        req = GenerationRequest(prompt="test")
        with pytest.raises(OllamaConnectionError, match="Cannot connect"):
            backend.generate(req, model_name="qwen2.5:0.5b")


# Real live integration test (runs only if local Ollama daemon is active)
@pytest.mark.skipif(not OllamaBackend().is_available(), reason="Local Ollama service offline")
def test_real_ollama_smoke():
    backend = OllamaBackend()
    models = backend.list_models()
    assert len(models) > 0, "At least one model should be installed in Ollama"

    model_name = models[0]["name"]
    req = GenerationRequest(prompt="Hello", max_tokens=5, temperature=0.0)
    res = backend.generate(req, model_name=model_name)

    assert len(res.text) > 0
    assert res.output_tokens > 0
    assert res.client_ttft_ms > 0
    assert res.client_tokens_per_sec > 0
