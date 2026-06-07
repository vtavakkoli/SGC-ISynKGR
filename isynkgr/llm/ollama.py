from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from isynkgr.utils.caching import JsonCache
from isynkgr.utils.hashing import stable_hash


_RETRYABLE_HTTP_STATUS = {408, 429, 500, 502, 503, 504}
_DEFAULT_GENERATE_TIMEOUT_S = 360.0  # long local models can exceed the previous 60 s socket timeout


def _env_float(name: str, default: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


class OllamaClient:
    def __init__(self, model: str = "gemma4:e2b", base_url: str | None = None, cache: JsonCache | None = None) -> None:
        self.model = model
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://host.docker.internal:11434")).rstrip("/")
        self.cache = cache or JsonCache()
        self.last_error: dict | None = None
        self.io_log_path = Path(os.getenv("OLLAMA_IO_LOG", "output/ollama_io.jsonl"))
        self.io_log_path.parent.mkdir(parents=True, exist_ok=True)

        # The previous hard-coded 60 s timeout caused local Ollama calls to be cut
        # while the model was still generating. The uploaded logs show /api/generate
        # failing almost exactly at 1m0s, so keep the connection open for more than
        # five minutes by default and allow easy tuning without code changes.
        timeout_default = _env_float("OLLAMA_TIMEOUT_S", _DEFAULT_GENERATE_TIMEOUT_S)
        self.generate_timeout_s = max(1.0, _env_float("OLLAMA_GENERATE_TIMEOUT_S", timeout_default))
        self.max_retries = max(0, _env_int("OLLAMA_GENERATE_MAX_RETRIES", 0))
        self.retry_backoff_s = max(0.0, _env_float("OLLAMA_GENERATE_RETRY_BACKOFF_S", 2.0))

    def _log_io(self, event: str, payload: dict) -> None:
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "model": self.model,
            **payload,
        }
        with self.io_log_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")

    @staticmethod
    def _parse_model_json(response_text: str) -> dict:
        text = (response_text or "").strip()
        if not text:
            raise ValueError("Ollama returned empty model response")
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start < 0 or end <= start:
                raise ValueError(f"Model response is not JSON: {text[:200]}") from None
            parsed = json.loads(text[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("Model response JSON must be an object")
        return parsed

    def _call_generate(self, endpoint: str, prompt: str, seed: int) -> dict:
        body = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"seed": seed, "temperature": 0},
        }
        self._log_io("request", {"endpoint": endpoint, "timeout_s": self.generate_timeout_s, "request_body": body})
        req = urllib.request.Request(endpoint, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.generate_timeout_s) as resp:
            raw_text = resp.read().decode(errors="replace").strip()
        if not raw_text:
            raise ValueError("Empty response body from Ollama")
        raw = json.loads(raw_text)
        parsed = self._parse_model_json(str(raw.get("response", "")))
        self._log_io("response", {"endpoint": endpoint, "response_body": raw, "parsed": parsed})
        return parsed

    @staticmethod
    def _http_error_payload(endpoint: str, attempt: int, exc: urllib.error.HTTPError) -> dict:
        body = ""
        try:
            body = exc.read().decode(errors="replace")
        except Exception:
            body = ""
        return {
            "endpoint": endpoint,
            "attempt": attempt,
            "status": exc.code,
            "reason": str(exc.reason),
            "body": body[:2000],
            "retryable": exc.code in _RETRYABLE_HTTP_STATUS,
        }

    def complete_json(self, prompt: str, schema_name: str, seed: int) -> dict:
        key = stable_hash({"m": self.model, "p": prompt, "s": schema_name, "seed": seed})
        cached = self.cache.get(key)
        if cached and not cached.get("_llm_error"):
            self.last_error = cached.get("_llm_error")
            self._log_io("cache_hit", {"cache_key": key, "schema_name": schema_name, "cached": cached})
            return cached

        endpoint = f"{self.base_url}/api/generate"
        errors: list[dict] = []
        total_attempts = self.max_retries + 1
        for attempt in range(1, total_attempts + 1):
            try:
                parsed = self._call_generate(endpoint, prompt, seed)
                self.last_error = None
                self.cache.set(key, parsed)
                return parsed
            except urllib.error.HTTPError as exc:
                err = self._http_error_payload(endpoint, attempt, exc)
                errors.append(err)
                should_retry = err["retryable"] and attempt < total_attempts
            except Exception as exc:
                errors.append({"endpoint": endpoint, "attempt": attempt, "error": str(exc)})
                should_retry = attempt < total_attempts

            if should_retry:
                self._log_io("retry", {"endpoint": endpoint, "attempt": attempt, "sleep_s": self.retry_backoff_s, "last_error": errors[-1]})
                if self.retry_backoff_s:
                    time.sleep(self.retry_backoff_s)

        llm_error = {
            "type": "llm_request_failed",
            "message": "Ollama request failed",
            "attempts": errors,
            "timeout_s": self.generate_timeout_s,
            "hint": "Local generation can be slow. Increase OLLAMA_GENERATE_TIMEOUT_S, check OLLAMA_BASE_URL, and ensure /api/generate is healthy.",
        }
        parsed = {"mappings": [], "_llm_error": llm_error}
        self.last_error = llm_error
        self._log_io("error", {"endpoint": endpoint, "error": llm_error, "prompt": prompt})
        return parsed
