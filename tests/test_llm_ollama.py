from __future__ import annotations

import json

import pytest

from isynkgr.llm.ollama import OllamaClient
from isynkgr.utils.caching import JsonCache


class _FakeResponse:
    def __init__(self, payload: dict | str) -> None:
        self._payload = payload

    def read(self) -> bytes:
        if isinstance(self._payload, str):
            return self._payload.encode()
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_complete_json_ignores_cached_llm_errors_and_retries(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cache = JsonCache(root=str(tmp_path / "cache"))
    log_path = tmp_path / "ollama_io.jsonl"
    monkeypatch.setenv("OLLAMA_IO_LOG", str(log_path))
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    key = "test-key"
    monkeypatch.setattr("isynkgr.llm.ollama.stable_hash", lambda _: key)
    cache.set(key, {"mappings": [], "_llm_error": {"type": "llm_request_failed"}})

    calls: list[str] = []

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        calls.append(req.full_url)
        return _FakeResponse({"response": json.dumps({"mappings": [{"source_path": "opcua://x", "target_path": "aas://y", "mapping_type": "equivalent"}]})})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = client.complete_json("prompt", "MappingList", 42)

    assert calls == ["http://example/api/generate"]
    assert len(result["mappings"]) == 1
    assert cache.get(key) == result

    events = [json.loads(line)["event"] for line in log_path.read_text().splitlines()]
    assert "request" in events and "response" in events


def test_complete_json_does_not_cache_failed_calls(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cache = JsonCache(root=str(tmp_path / "cache"))
    log_path = tmp_path / "ollama_io.jsonl"
    monkeypatch.setenv("OLLAMA_IO_LOG", str(log_path))
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    key = "test-key-2"
    monkeypatch.setattr("isynkgr.llm.ollama.stable_hash", lambda _: key)

    def _raise(req, timeout=0):  # noqa: ANN001
        raise OSError("network unavailable")

    monkeypatch.setattr("urllib.request.urlopen", _raise)

    result = client.complete_json("prompt", "MappingList", 42)

    assert result.get("_llm_error", {}).get("type") == "llm_request_failed"
    assert cache.get(key) is None
    assert any(json.loads(line)["event"] == "error" for line in log_path.read_text().splitlines())


def test_complete_json_extracts_json_from_wrapped_model_response(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cache = JsonCache(root=str(tmp_path / "cache"))
    monkeypatch.setenv("OLLAMA_IO_LOG", str(tmp_path / "ollama_io.jsonl"))
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        payload = {
            "response": "<think>internal</think>\n{\"mappings\":[{\"source_path\":\"opcua://x\",\"target_path\":\"aas://y/submodel/default/element/v\",\"mapping_type\":\"equivalent\"}]}",
        }
        return _FakeResponse(payload)

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = client.complete_json("prompt", "MappingList", 42)
    assert result["mappings"][0]["mapping_type"] == "equivalent"


def test_complete_json_handles_empty_model_response(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cache = JsonCache(root=str(tmp_path / "cache"))
    monkeypatch.setenv("OLLAMA_IO_LOG", str(tmp_path / "ollama_io.jsonl"))
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        return _FakeResponse({"response": ""})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)
    result = client.complete_json("prompt", "MappingList", 42)
    assert result.get("_llm_error", {}).get("type") == "llm_request_failed"


def test_complete_json_uses_generate_timeout_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    cache = JsonCache(root=str(tmp_path / "cache"))
    monkeypatch.setenv("OLLAMA_IO_LOG", str(tmp_path / "ollama_io.jsonl"))
    monkeypatch.setenv("OLLAMA_GENERATE_TIMEOUT_S", "390")
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    observed_timeouts: list[float] = []

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        observed_timeouts.append(timeout)
        return _FakeResponse({"response": json.dumps({"mappings": []})})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = client.complete_json("prompt", "MappingList", 42)

    assert result["mappings"] == []
    assert observed_timeouts == [390.0]


def test_complete_json_retries_retryable_http_error_when_configured(monkeypatch: pytest.MonkeyPatch, tmp_path):
    import io
    import urllib.error

    cache = JsonCache(root=str(tmp_path / "cache"))
    monkeypatch.setenv("OLLAMA_IO_LOG", str(tmp_path / "ollama_io.jsonl"))
    monkeypatch.setenv("OLLAMA_GENERATE_MAX_RETRIES", "1")
    monkeypatch.setenv("OLLAMA_GENERATE_RETRY_BACKOFF_S", "0")
    client = OllamaClient(model="demo", base_url="http://example", cache=cache)

    calls: list[int] = []

    def _fake_urlopen(req, timeout=0):  # noqa: ANN001
        calls.append(len(calls) + 1)
        if len(calls) == 1:
            raise urllib.error.HTTPError(req.full_url, 500, "server busy", {}, io.BytesIO(b"temporary failure"))
        return _FakeResponse({"response": json.dumps({"mappings": [{"source_path": "opcua://x", "target_path": "aas://y", "mapping_type": "equivalent"}]})})

    monkeypatch.setattr("urllib.request.urlopen", _fake_urlopen)

    result = client.complete_json("prompt", "MappingList", 42)

    assert calls == [1, 2]
    assert len(result["mappings"]) == 1


def test_complete_json_default_generate_timeout_is_over_five_minutes(monkeypatch: pytest.MonkeyPatch, tmp_path):
    monkeypatch.delenv("OLLAMA_GENERATE_TIMEOUT_S", raising=False)
    monkeypatch.delenv("OLLAMA_TIMEOUT_S", raising=False)
    monkeypatch.setenv("OLLAMA_IO_LOG", str(tmp_path / "ollama_io.jsonl"))
    client = OllamaClient(model="demo", base_url="http://example", cache=JsonCache(root=str(tmp_path / "cache")))
    assert client.generate_timeout_s >= 300
