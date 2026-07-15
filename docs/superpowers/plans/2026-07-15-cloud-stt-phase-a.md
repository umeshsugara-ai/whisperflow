# Phase A: Multi-Provider Cloud STT Engine — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a provider registry + a generic OpenAI-compatible STT engine (covers Groq and OpenAI) plus a small Deepgram engine, wire them into `create_engine`/`recommend`, so any user can transcribe via a free or paid cloud provider — starting with Groq, which runs the same `whisper-large-v3-turbo` model as the local engine but in the cloud, free up to 2,000 requests/day.

**Architecture:** One new data module (`whisperflow/stt/providers.py`) is a plain registry of `Provider` dataclasses — no framework, no LangChain. `create_engine()` dispatches on each provider's `kind` field to one of: the existing `FasterWhisperEngine`/`GeminiEngine`, or two new engines (`OpenAICompatibleEngine`, `DeepgramEngine`) that both follow the exact "plain urllib REST, in-memory WAV" pattern already proven in `gemini_engine.py`.

**Tech Stack:** Python stdlib only (`urllib.request`, `json`, `io`, `struct`) — no new pip dependencies. pytest + `monkeypatch` for HTTP mocking, matching `tests/test_gemini_engine.py`.

## Global Constraints

- No LangChain, no provider SDKs — plain REST via `urllib`, per the spec's explicit non-goal (keeps the frozen build lean).
- Every engine implements `whisperflow.stt.base.SttEngine` (`load()`, `transcribe(audio, language, initial_prompt) -> RawResult`) — see `whisperflow/stt/base.py:24-35`.
- Audio is always float32 mono @ 16kHz (`whisperflow/stt/base.py:32`); WAV encoding reuses the exact byte-for-byte pattern in `whisperflow/stt/gemini_engine.py:49-60` (`_float32_to_wav_bytes`).
- API keys are never logged. Every engine's error messages may reference the provider name but never the key value.
- `[model].engine` values must exactly match a `Provider.id` in the registry — `ConfigError` on mismatch, matching the existing style in `whisperflow/config.py:153-178` (`_validate`).
- Dev-mode paths and existing local/gemini behavior are unchanged — this plan only adds new provider ids, it does not modify `FasterWhisperEngine` or `GeminiEngine` internals except where explicitly stated (Task 6, default model only).
- Gemini's cloud STT default model changes from `gemini-2.5-flash` to `gemini-2.5-flash-lite` (verified 2026-07-15: same audio-input capability, ~3x cheaper — $0.30 vs $1.00 per million audio tokens, current/stable, not deprecated).

---

### Task 1: Provider registry

**Files:**
- Create: `whisperflow/stt/providers.py`
- Test: `tests/test_providers.py`

**Interfaces:**
- Consumes: nothing (pure data module).
- Produces: `Provider` dataclass (fields: `id`, `display_name`, `kind`, `base_url`, `default_model`, `api_key_env`, `signup_url`, `cost_tier`, `cost_note`, `quality_tier`, `speed_note`, `setup_steps`), `PROVIDERS: dict[str, Provider]`, `get(provider_id: str) -> Provider` (raises `KeyError` with a clear message if unknown), `all_providers() -> list[Provider]` (registry order), `cloud_providers() -> list[Provider]` (excludes `kind == "local"`), `is_cloud(provider_id: str) -> bool`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_providers.py
# -*- coding: utf-8 -*-
"""Cloud STT provider registry — data only, no network."""

import pytest

from whisperflow.stt import providers


def test_groq_is_registered_and_openai_compatible():
    p = providers.get("groq")
    assert p.kind == "openai_compatible"
    assert p.base_url == "https://api.groq.com/openai/v1"
    assert p.default_model == "whisper-large-v3-turbo"
    assert p.api_key_env == "GROQ_API_KEY"
    assert p.cost_tier == "free"
    assert p.setup_steps  # non-empty guide


def test_gemini_is_registered_with_cheap_default():
    p = providers.get("gemini")
    assert p.kind == "gemini"
    assert p.default_model == "gemini-2.5-flash-lite"
    assert p.api_key_env == "GEMINI_API_KEY"


def test_openai_is_registered_and_paid():
    p = providers.get("openai")
    assert p.kind == "openai_compatible"
    assert p.base_url == "https://api.openai.com/v1"
    assert p.cost_tier == "paid"


def test_deepgram_is_registered():
    p = providers.get("deepgram")
    assert p.kind == "deepgram"
    assert p.api_key_env == "DEEPGRAM_API_KEY"


def test_local_is_registered_and_excluded_from_cloud_list():
    p = providers.get("local")
    assert p.kind == "local"
    ids = [x.id for x in providers.cloud_providers()]
    assert "local" not in ids
    assert "groq" in ids


def test_get_unknown_id_raises_clear_error():
    with pytest.raises(KeyError, match="unknown speech engine 'nonsense'"):
        providers.get("nonsense")


def test_is_cloud():
    assert providers.is_cloud("groq") is True
    assert providers.is_cloud("local") is False


def test_all_providers_includes_local_and_every_cloud_id():
    ids = {p.id for p in providers.all_providers()}
    assert ids == {"local", "groq", "gemini", "openai", "deepgram"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_providers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'whisperflow.stt.providers'`

- [ ] **Step 3: Write the registry**

```python
# whisperflow/stt/providers.py
"""Cloud + local speech-to-text provider registry.

A plain data table, not a framework — each provider is a `Provider` row
describing how to reach it (kind + base_url + key env var) and how to
explain it to a non-technical user (cost/quality/speed notes, signup link,
step-by-step key guide). `create_engine()` (registry.py) dispatches on
`kind`; the Settings UI (Phase B) renders badges straight from these
fields — no per-provider UI code needed to add a new one.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Provider:
    id: str
    display_name: str
    kind: str  # openai_compatible | gemini | deepgram | local
    base_url: str  # "" for gemini/local (their engines hardcode the endpoint)
    default_model: str
    api_key_env: str  # "" for local
    signup_url: str  # "" for local
    cost_tier: str  # free | freemium | paid
    cost_note: str
    quality_tier: str  # good | better | best
    speed_note: str
    setup_steps: tuple[str, ...] = field(default_factory=tuple)


PROVIDERS: dict[str, Provider] = {
    "local": Provider(
        id="local",
        display_name="Local (on-device)",
        kind="local",
        base_url="",
        default_model="large-v3-turbo",
        api_key_env="",
        signup_url="",
        cost_tier="free",
        cost_note="Free — runs on your machine",
        quality_tier="best",
        speed_note="Depends on your GPU",
        setup_steps=(),
    ),
    "groq": Provider(
        id="groq",
        display_name="Groq (free, fast cloud)",
        kind="openai_compatible",
        base_url="https://api.groq.com/openai/v1",
        default_model="whisper-large-v3-turbo",
        api_key_env="GROQ_API_KEY",
        signup_url="https://console.groq.com/keys",
        cost_tier="free",
        cost_note="Free — 2,000 requests/day",
        quality_tier="better",
        speed_note="Instant",
        setup_steps=(
            "Open console.groq.com/keys (click 'Get a free key' below).",
            "Sign in with Google or GitHub — no credit card needed.",
            "Click 'Create API Key', give it any name, and copy the key.",
            "Paste it into the field below.",
        ),
    ),
    "gemini": Provider(
        id="gemini",
        display_name="Google Gemini (free)",
        kind="gemini",
        base_url="",
        default_model="gemini-2.5-flash-lite",
        api_key_env="GEMINI_API_KEY",
        signup_url="https://aistudio.google.com/apikey",
        cost_tier="free",
        cost_note="Free tier — generous daily quota",
        quality_tier="better",
        speed_note="Fast",
        setup_steps=(
            "Open aistudio.google.com/apikey (click 'Get a free key' below).",
            "Sign in with your Google account.",
            "Click 'Create API key' and copy it.",
            "Paste it into the field below.",
        ),
    ),
    "openai": Provider(
        id="openai",
        display_name="OpenAI (paid, high accuracy)",
        kind="openai_compatible",
        base_url="https://api.openai.com/v1",
        default_model="gpt-4o-transcribe",
        api_key_env="OPENAI_API_KEY",
        signup_url="https://platform.openai.com/api-keys",
        cost_tier="paid",
        cost_note="~$0.006/minute — add billing to your OpenAI account",
        quality_tier="best",
        speed_note="Fast",
        setup_steps=(
            "Open platform.openai.com/api-keys (click 'Get a key' below).",
            "Sign in and add a payment method (Settings > Billing) — required even for light use.",
            "Click 'Create new secret key' and copy it immediately (shown once).",
            "Paste it into the field below.",
        ),
    ),
    "deepgram": Provider(
        id="deepgram",
        display_name="Deepgram (paid, best accuracy)",
        kind="deepgram",
        base_url="https://api.deepgram.com/v1",
        default_model="nova-3",
        api_key_env="DEEPGRAM_API_KEY",
        signup_url="https://console.deepgram.com",
        cost_tier="paid",
        cost_note="$200 free credit, then pay-as-you-go",
        quality_tier="best",
        speed_note="Fast",
        setup_steps=(
            "Open console.deepgram.com (click 'Get a key' below) and sign up.",
            "Go to API Keys in the left sidebar.",
            "Click 'Create a New API Key', copy it.",
            "Paste it into the field below.",
        ),
    ),
}


def get(provider_id: str) -> Provider:
    try:
        return PROVIDERS[provider_id]
    except KeyError:
        raise KeyError(f"unknown speech engine {provider_id!r}") from None


def all_providers() -> list[Provider]:
    return list(PROVIDERS.values())


def cloud_providers() -> list[Provider]:
    return [p for p in PROVIDERS.values() if p.kind != "local"]


def is_cloud(provider_id: str) -> bool:
    return get(provider_id).kind != "local"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_providers.py -v`
Expected: PASS (9 passed)

- [ ] **Step 5: Commit**

```bash
git add whisperflow/stt/providers.py tests/test_providers.py
git commit -m "Add cloud STT provider registry (Groq, Gemini, OpenAI, Deepgram, local)"
```

---

### Task 2: `OpenAICompatibleEngine` — request building + response parsing

**Files:**
- Create: `whisperflow/stt/openai_compatible_engine.py`
- Test: `tests/test_openai_compatible_engine.py`

**Interfaces:**
- Consumes: `whisperflow.config.ModelConfig` (existing), `whisperflow.stt.base.RawResult`/`SttEngine` (existing), `whisperflow.stt.providers.get` (Task 1).
- Produces: `OpenAICompatibleEngine(cfg: ModelConfig)` class with `.load()` and `.transcribe(audio, language="", initial_prompt="") -> RawResult`; module-level `_multipart_body(fields: dict, file_field: str, filename: str, file_bytes: bytes, content_type: str) -> tuple[bytes, str]` returning `(body, content_type_header)`.

**Design notes:** `ModelConfig` needs to know which provider it's using. Reuse the existing `engine` field directly as the provider id (e.g. `cfg.engine = "groq"`) — this is exactly how `gemini`/`local` already work (`create_engine` in registry.py switches on `cfg.engine`), so no new config field is needed. The engine looks up its own `Provider` row via `providers.get(cfg.engine)` in `__init__`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_openai_compatible_engine.py
# -*- coding: utf-8 -*-
"""Generic OpenAI-compatible STT engine (Groq, OpenAI) — request/response, mocked HTTP."""

import io
import json

import numpy as np
import pytest

from whisperflow.config import ModelConfig
from whisperflow.stt.openai_compatible_engine import OpenAICompatibleEngine, _multipart_body


def cfg(**kw) -> ModelConfig:
    defaults = dict(engine="groq", api_key="test-groq-key")
    defaults.update(kw)
    return ModelConfig(**defaults)


def test_multipart_body_contains_field_and_file():
    body, content_type = _multipart_body(
        fields={"model": "whisper-large-v3-turbo", "language": "en"},
        file_field="file",
        filename="audio.wav",
        file_bytes=b"RIFF....",
        content_type="audio/wav",
    )
    assert content_type.startswith("multipart/form-data; boundary=")
    boundary = content_type.split("boundary=")[1]
    text = body.decode("latin-1")
    assert boundary in text
    assert 'name="model"' in text
    assert "whisper-large-v3-turbo" in text
    assert 'name="file"; filename="audio.wav"' in text
    assert "Content-Type: audio/wav" in text
    assert text.rstrip().endswith(f"--{boundary}--")


def test_load_requires_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    engine = OpenAICompatibleEngine(cfg(api_key=""))
    with pytest.raises(RuntimeError, match="API key"):
        engine.load()


def test_load_resolves_provider_and_model():
    engine = OpenAICompatibleEngine(cfg())
    engine.load()
    assert engine.model_id == "whisper-large-v3-turbo"
    assert engine.provider.id == "groq"


def test_transcribe_sends_bearer_auth_and_parses_text(monkeypatch):
    engine = OpenAICompatibleEngine(cfg())
    engine.load()

    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps({"text": "hello from groq"}).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    audio = np.zeros(32000, dtype=np.float32)  # 2s @ 16kHz
    result = engine.transcribe(audio, language="en", initial_prompt="Vidysea")

    assert result.text == "hello from groq"
    assert result.duration_s == 2.0
    assert captured["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
    assert captured["headers"].get("Authorization") == "Bearer test-groq-key"
    assert b"whisper-large-v3-turbo" in captured["body"]
    assert b"Vidysea" in captured["body"]


def test_transcribe_raises_readable_error_on_http_401(monkeypatch):
    import urllib.error

    engine = OpenAICompatibleEngine(cfg())
    engine.load()

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "Unauthorized", {}, io.BytesIO(b"bad key"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Groq API error 401"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


def test_registry_dispatches_openai_compatible_for_groq_and_openai():
    from whisperflow.stt.registry import create_engine

    assert isinstance(create_engine(cfg(engine="groq")), OpenAICompatibleEngine)
    assert isinstance(create_engine(cfg(engine="openai", api_key="k")), OpenAICompatibleEngine)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_openai_compatible_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'whisperflow.stt.openai_compatible_engine'`

- [ ] **Step 3: Write the engine**

```python
# whisperflow/stt/openai_compatible_engine.py
"""Generic OpenAI-compatible STT engine — covers Groq and OpenAI (and any
other provider that implements the same POST /audio/transcriptions
multipart endpoint). One engine, `base_url`/`model`/`key` differ per
provider (see providers.py).

Uses plain `urllib` multipart encoding (no `requests` dependency, matching
gemini_engine.py's "no SDK" pattern) — audio is wrapped as in-memory WAV,
same helper as the Gemini engine.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid

import numpy as np

from whisperflow.config import ModelConfig

from . import providers
from .base import RawResult, SttEngine
from .gemini_engine import SAMPLE_RATE, _float32_to_wav_bytes

log = logging.getLogger(__name__)

# OpenAI/Groq cap the transcription prompt at 224 tokens; this is a rough
# character budget (not a real tokenizer) just to avoid an obvious 400.
MAX_PROMPT_CHARS = 800


def _multipart_body(
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> tuple[bytes, str]:
    """Build a multipart/form-data body + its Content-Type header value."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in fields.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode(
                "utf-8"
            )
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; '
        f'filename="{filename}"\r\nContent-Type: {content_type}\r\n\r\n'.encode("utf-8")
    )
    parts.append(file_bytes)
    parts.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


class OpenAICompatibleEngine(SttEngine):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.provider = providers.get(cfg.engine)
        self.model_id = cfg.cloud_model or self.provider.default_model
        self._api_key = ""

    def load(self) -> None:
        self._api_key = self.cfg.resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                f"{self.provider.display_name} needs an API key — set [model].api_key "
                f"or ${self.provider.api_key_env}"
            )
        log.warning(
            "CLOUD ENGINE ACTIVE: dictation audio will be sent to %s (%s).",
            self.provider.display_name,
            self.model_id,
        )

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "",
        initial_prompt: str = "",
    ) -> RawResult:
        if not self._api_key:
            raise RuntimeError("engine not loaded — call load() first")

        t0 = time.perf_counter()
        duration_s = len(audio) / SAMPLE_RATE

        fields = {"model": self.model_id, "response_format": "json", "temperature": "0"}
        if language:
            fields["language"] = language
        if initial_prompt:
            fields["prompt"] = initial_prompt[:MAX_PROMPT_CHARS]

        body, content_type = _multipart_body(
            fields=fields,
            file_field="file",
            filename="audio.wav",
            file_bytes=_float32_to_wav_bytes(audio),
            content_type="audio/wav",
        )

        req = urllib.request.Request(
            f"{self.provider.base_url}/audio/transcriptions",
            data=body,
            headers={
                "Content-Type": content_type,
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(
                f"{self.provider.display_name} API error {exc.code}: {detail}"
            ) from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"{self.provider.display_name} API unreachable: {exc}") from exc

        text = result.get("text", "").strip()
        return RawResult(
            text=text,
            language=language or result.get("language", "auto"),
            language_probability=0.0,  # not reported by this API shape
            duration_s=duration_s,
            transcribe_seconds=time.perf_counter() - t0,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_openai_compatible_engine.py -v`
Expected: 5 passed, 1 failed (`test_registry_dispatches_openai_compatible_for_groq_and_openai` fails — `create_engine` doesn't know about `"groq"`/`"openai"` yet; that's Task 4). Confirm the other 5 pass.

- [ ] **Step 5: Commit**

```bash
git add whisperflow/stt/openai_compatible_engine.py tests/test_openai_compatible_engine.py
git commit -m "Add generic OpenAI-compatible STT engine (Groq, OpenAI)"
```

---

### Task 3: `DeepgramEngine`

**Files:**
- Create: `whisperflow/stt/deepgram_engine.py`
- Test: `tests/test_deepgram_engine.py`

**Interfaces:**
- Consumes: same as Task 2 (`ModelConfig`, `RawResult`/`SttEngine`, `providers.get`, `_float32_to_wav_bytes`, `SAMPLE_RATE`).
- Produces: `DeepgramEngine(cfg: ModelConfig)` with `.load()`/`.transcribe(...)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_deepgram_engine.py
# -*- coding: utf-8 -*-
"""Deepgram STT engine — request building + response parsing, mocked HTTP."""

import io
import json

import numpy as np
import pytest

from whisperflow.config import ModelConfig
from whisperflow.stt.deepgram_engine import DeepgramEngine


def cfg(**kw) -> ModelConfig:
    defaults = dict(engine="deepgram", api_key="test-dg-key")
    defaults.update(kw)
    return ModelConfig(**defaults)


def test_load_requires_key(monkeypatch):
    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
    engine = DeepgramEngine(cfg(api_key=""))
    with pytest.raises(RuntimeError, match="API key"):
        engine.load()


def test_transcribe_sends_token_auth_and_parses_response(monkeypatch):
    engine = DeepgramEngine(cfg())
    engine.load()

    captured = {}

    class FakeResponse:
        def read(self):
            return json.dumps(
                {
                    "results": {
                        "channels": [
                            {"alternatives": [{"transcript": "hello from deepgram"}]}
                        ]
                    }
                }
            ).encode("utf-8")

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.headers)
        captured["body"] = req.data
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    audio = np.zeros(16000, dtype=np.float32)  # 1s
    result = engine.transcribe(audio)

    assert result.text == "hello from deepgram"
    assert result.duration_s == 1.0
    assert "nova-3" in captured["url"]
    assert captured["headers"].get("Authorization") == "Token test-dg-key"
    assert captured["body"].startswith(b"RIFF")  # raw WAV body, not multipart


def test_transcribe_raises_readable_error_on_http_error(monkeypatch):
    import urllib.error

    engine = DeepgramEngine(cfg())
    engine.load()

    def fake_urlopen(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b"bad audio"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(RuntimeError, match="Deepgram API error 400"):
        engine.transcribe(np.zeros(16000, dtype=np.float32))


def test_registry_dispatches_deepgram():
    from whisperflow.stt.registry import create_engine

    assert isinstance(create_engine(cfg()), DeepgramEngine)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_deepgram_engine.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'whisperflow.stt.deepgram_engine'`

- [ ] **Step 3: Write the engine**

```python
# whisperflow/stt/deepgram_engine.py
"""Deepgram STT engine — raw-WAV POST to /v1/listen (not multipart; Deepgram
takes the audio bytes directly as the request body with a Content-Type
header), Token auth. Same "plain urllib REST" pattern as the other cloud
engines.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

from whisperflow.config import ModelConfig

from . import providers
from .base import RawResult, SttEngine
from .gemini_engine import SAMPLE_RATE, _float32_to_wav_bytes

log = logging.getLogger(__name__)


class DeepgramEngine(SttEngine):
    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg
        self.provider = providers.get("deepgram")
        self.model_id = cfg.cloud_model or self.provider.default_model
        self._api_key = ""

    def load(self) -> None:
        self._api_key = self.cfg.resolve_api_key()
        if not self._api_key:
            raise RuntimeError(
                f"Deepgram needs an API key — set [model].api_key or ${self.provider.api_key_env}"
            )
        log.warning(
            "CLOUD ENGINE ACTIVE: dictation audio will be sent to Deepgram (%s).", self.model_id
        )

    def transcribe(
        self,
        audio: np.ndarray,
        language: str = "",
        initial_prompt: str = "",
    ) -> RawResult:
        if not self._api_key:
            raise RuntimeError("engine not loaded — call load() first")

        t0 = time.perf_counter()
        duration_s = len(audio) / SAMPLE_RATE

        params = {"model": self.model_id, "punctuate": "true"}
        if language:
            params["language"] = language
        query = urllib.parse.urlencode(params)

        req = urllib.request.Request(
            f"{self.provider.base_url}/listen?{query}",
            data=_float32_to_wav_bytes(audio),
            headers={
                "Content-Type": "audio/wav",
                "Authorization": f"Token {self._api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"Deepgram API error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(f"Deepgram API unreachable: {exc}") from exc

        text = self._extract_text(body)
        return RawResult(
            text=text,
            language=language or "auto",
            language_probability=0.0,
            duration_s=duration_s,
            transcribe_seconds=time.perf_counter() - t0,
        )

    @staticmethod
    def _extract_text(body: dict) -> str:
        try:
            return body["results"]["channels"][0]["alternatives"][0]["transcript"].strip()
        except (KeyError, IndexError, TypeError):
            raise RuntimeError(f"Deepgram returned no transcription: {body}") from None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_deepgram_engine.py -v`
Expected: 3 passed, 1 failed (`test_registry_dispatches_deepgram` — Task 4 wires this). Confirm the other 3 pass.

- [ ] **Step 5: Commit**

```bash
git add whisperflow/stt/deepgram_engine.py tests/test_deepgram_engine.py
git commit -m "Add Deepgram STT engine"
```

---

### Task 4: Wire dispatch in `registry.py` + validate config against the provider registry

**Files:**
- Modify: `whisperflow/stt/registry.py`
- Modify: `whisperflow/config.py:54` (`VALID_ENGINES`), `whisperflow/config.py:186-188` (`_validate`)
- Test: `tests/test_openai_compatible_engine.py::test_registry_dispatches_openai_compatible_for_groq_and_openai` (Task 2, now passes), `tests/test_deepgram_engine.py::test_registry_dispatches_deepgram` (Task 3, now passes)
- Test: `tests/test_config_write.py` (extend — see Step 1)

**Interfaces:**
- Consumes: `providers.get`, `providers.PROVIDERS` (Task 1), `OpenAICompatibleEngine` (Task 2), `DeepgramEngine` (Task 3).
- Produces: `create_engine(cfg: ModelConfig) -> SttEngine` now dispatches on `providers.get(cfg.engine).kind` instead of the old `if cfg.engine == "gemini"` check; `VALID_ENGINES` is derived from the registry so any provider id validates automatically.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_config_write.py` (open the file first to match its existing style — append these two functions):

```python
def test_config_accepts_any_registered_cloud_engine(tmp_path, monkeypatch):
    from whisperflow.config import load_config

    monkeypatch.setenv("GROQ_API_KEY", "k")
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[model]\nengine = "groq"\n', encoding="utf-8")
    cfg = load_config(cfg_path)
    assert cfg.model.engine == "groq"


def test_config_rejects_unregistered_engine(tmp_path):
    from whisperflow.config import ConfigError, load_config

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text('[model]\nengine = "not-a-real-provider"\n', encoding="utf-8")
    try:
        load_config(cfg_path)
        assert False, "expected ConfigError"
    except ConfigError as exc:
        assert "engine" in str(exc)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_config_write.py -k "registered_cloud_engine or rejects_unregistered" -v`
Expected: FAIL — `test_config_accepts_any_registered_cloud_engine` fails because `VALID_ENGINES = {"local", "gemini"}` doesn't include `"groq"`.

- [ ] **Step 3: Update `registry.py` and `config.py`**

Replace `whisperflow/stt/registry.py` entirely:

```python
"""STT engine dispatch: provider id -> engine instance.

`ModelConfig.engine` is a provider id from `whisperflow.stt.providers`
(e.g. "local", "groq", "gemini", "openai", "deepgram"). Which concrete
engine class handles it is decided by the provider's `kind` — adding a new
provider that reuses an existing kind (e.g. another openai_compatible
service) needs zero changes here, just a new providers.py row.
"""

from __future__ import annotations

from whisperflow.config import ModelConfig

from . import providers
from .base import SttEngine

# friendly local-model name -> HF repo id (all CTranslate2/faster-whisper format)
FASTER_WHISPER_MODELS: dict[str, str] = {
    "large-v3-turbo": "mobiuslabsgmbh/faster-whisper-large-v3-turbo",  # default; ~1.5GB
    "large-v3": "Systran/faster-whisper-large-v3",  # best Hindi accuracy; ~2.9GB, ~5x slower
    "medium": "Systran/faster-whisper-medium",
    "small": "Systran/faster-whisper-small",
}


def resolve_model_id(name: str) -> str:
    """Friendly registry name or raw HF repo id passthrough."""
    return FASTER_WHISPER_MODELS.get(name, name)


_ENGINE_BY_KIND = {
    "gemini": "whisperflow.stt.gemini_engine.GeminiEngine",
    "openai_compatible": "whisperflow.stt.openai_compatible_engine.OpenAICompatibleEngine",
    "deepgram": "whisperflow.stt.deepgram_engine.DeepgramEngine",
    "local": "whisperflow.stt.faster_whisper_engine.FasterWhisperEngine",
}


def create_engine(cfg: ModelConfig) -> SttEngine:
    provider = providers.get(cfg.engine)
    module_path, class_name = _ENGINE_BY_KIND[provider.kind].rsplit(".", 1)
    import importlib

    module = importlib.import_module(module_path)
    engine_cls = getattr(module, class_name)
    return engine_cls(cfg)
```

Edit `whisperflow/config.py` — replace the `VALID_ENGINES` line (`config.py:54`):

```python
# whisperflow/config.py:54 (old)
VALID_ENGINES = {"local", "gemini"}
```

with a lazy import (avoid a circular import — `providers.py` doesn't import `config.py`, so this is safe):

```python
# whisperflow/config.py:54 (new)
def _valid_engines() -> set[str]:
    from whisperflow.stt import providers

    return set(providers.PROVIDERS.keys())
```

Then in `_validate` (`whisperflow/config.py:186-188`, currently):

```python
    if m.engine not in VALID_ENGINES:
        raise ConfigError(f"[model].engine must be one of {sorted(VALID_ENGINES)}, got {m.engine!r}")
    if m.engine != "local" and not m.resolve_api_key():
```

change to:

```python
    from whisperflow.stt import providers as _providers

    valid_engines = _valid_engines()
    if m.engine not in valid_engines:
        raise ConfigError(f"[model].engine must be one of {sorted(valid_engines)}, got {m.engine!r}")
    if _providers.is_cloud(m.engine) and not m.resolve_api_key():
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_config_write.py tests/test_openai_compatible_engine.py tests/test_deepgram_engine.py tests/test_gemini_engine.py tests/test_providers.py -v`
Expected: ALL PASS — this also re-confirms `test_registry_dispatches_gemini` (existing test) still passes, since `gemini` kind still maps to `GeminiEngine`.

- [ ] **Step 5: Run the full suite to catch regressions**

Run: `python -m pytest -q`
Expected: all tests pass (115 existing + new ones from Tasks 1-4).

- [ ] **Step 6: Commit**

```bash
git add whisperflow/stt/registry.py whisperflow/config.py tests/test_config_write.py
git commit -m "Dispatch STT engines through the provider registry; validate [model].engine against it"
```

---

### Task 5: `recommend()` prefers Groq for weak/GPU-less machines

**Files:**
- Modify: `whisperflow/sysinfo.py:221-300` (`recommend`)
- Test: `tests/test_sysinfo.py` (extend)

**Interfaces:**
- Consumes: `providers.get("groq")` for the display name in the reason string (optional — plain string is fine too, keep it simple).
- Produces: `recommend(specs, has_api_key: bool = False) -> Recommendation` — signature unchanged, but the weak-machine branch now returns `engine="groq"` instead of `engine="gemini"`, and `has_api_key` semantically becomes "any cloud key is present" (the caller in `app.py`/`main.py` decides what counts — this task only changes the returned engine id).

**Design note:** the existing tests `test_weak_machine_with_key_gets_cloud` and `test_gpu_owner_with_key_gets_cloud_as_alternative_not_default` (in `tests/test_sysinfo.py:35-51`) currently assert `rec.engine == "gemini"` — this task changes that expectation to `"groq"`. This is a deliberate spec-driven behavior change, not a regression: update those two tests in place.

- [ ] **Step 1: Update the existing tests to expect Groq**

In `tests/test_sysinfo.py`, replace:

```python
def test_weak_machine_with_key_gets_cloud():
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=True)
    assert rec.engine == "gemini"
    assert "audio leaves the machine" in rec.reason
```

with:

```python
def test_weak_machine_with_key_gets_cloud():
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=True)
    assert rec.engine == "groq"
    assert "audio leaves the machine" in rec.reason
```

And replace:

```python
def test_gpu_owner_with_key_gets_cloud_as_alternative_not_default():
    rec = recommend(specs(vram_mb=8192, gpu="RTX 4060"), has_api_key=True)
    assert rec.engine == "local"  # local stays default when hardware allows
    assert any("gemini" in a for a in rec.alternatives)
```

with:

```python
def test_gpu_owner_with_key_gets_cloud_as_alternative_not_default():
    rec = recommend(specs(vram_mb=8192, gpu="RTX 4060"), has_api_key=True)
    assert rec.engine == "local"  # local stays default when hardware allows
    assert any("groq" in a for a in rec.alternatives)
```

Also update `test_weak_machine_without_key_gets_small_with_cloud_alternative` (`tests/test_sysinfo.py:41-45`):

```python
def test_weak_machine_without_key_gets_small_with_cloud_alternative():
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=False)
    assert rec.engine == "local"
    assert rec.name == "small"
    assert any("gemini" in a for a in rec.alternatives)
```

to:

```python
def test_weak_machine_without_key_gets_small_with_cloud_alternative():
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=False)
    assert rec.engine == "local"
    assert rec.name == "small"
    assert any("groq" in a for a in rec.alternatives)
```

Add one new test:

```python
def test_no_gpu_weak_cpu_without_key_still_mentions_groq_as_free_option():
    # even with NO key, the free-tier cloud option should be surfaced —
    # unlike the old gemini-only behavior, groq needs no pre-existing key
    # to be worth recommending (it's free to sign up for).
    rec = recommend(specs(vram_mb=0, ram_gb=4, cores=2), has_api_key=False)
    assert any("groq" in a.lower() for a in rec.alternatives)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_sysinfo.py -v`
Expected: FAIL — `rec.engine == "gemini"` assertions fail because production code still returns `"gemini"`.

- [ ] **Step 3: Update `recommend()`**

In `whisperflow/sysinfo.py`, in the `>= 5000` VRAM branch (`:233-245`), change:

```python
    if specs.vram_mb >= 5000:
        alts.append("large-v3 (best Hindi accuracy, ~5x slower) if you can wait")
        if has_api_key:
            alts.append("engine='gemini' to save all VRAM for other GPU work")
```

to:

```python
    if specs.vram_mb >= 5000:
        alts.append("large-v3 (best Hindi accuracy, ~5x slower) if you can wait")
        if has_api_key:
            alts.append("engine='groq' to save all VRAM for other GPU work")
```

In the `>= 3000` VRAM branch (`:259-267`), change:

```python
    if specs.vram_mb > 0:
        return Recommendation(
            engine="local",
            name="small",
            device="cuda",
            compute_type="int8_float16",
            reason=f"{specs.gpu_name} has only {specs.vram_mb / 1024:.1f}GB VRAM — small is the safe fit",
            alternatives=["engine='gemini' (BYOK) for better accuracy than small"] if has_api_key else [],
        )
```

to:

```python
    if specs.vram_mb > 0:
        return Recommendation(
            engine="local",
            name="small",
            device="cuda",
            compute_type="int8_float16",
            reason=f"{specs.gpu_name} has only {specs.vram_mb / 1024:.1f}GB VRAM — small is the safe fit",
            alternatives=["engine='groq' (free) for better accuracy than small"],
        )
```

In the CPU-with-RAM branch (`:269-281`), change:

```python
    if specs.ram_gb >= 8 and specs.cpu_cores >= 4:
        alts = ["medium on cpu if accuracy matters more than speed"]
        if has_api_key:
            alts.insert(0, "engine='gemini' (BYOK) — much better accuracy than small, no download")
        return Recommendation(
            engine="local",
            name="small",
            device="cpu",
            compute_type="int8",
            reason=f"no NVIDIA GPU detected; {specs.cpu_cores} cores / {specs.ram_gb:.0f}GB RAM "
            "can run small on CPU (expect a few seconds per dictation)",
            alternatives=alts,
        )
```

to:

```python
    if specs.ram_gb >= 8 and specs.cpu_cores >= 4:
        alts = [
            "engine='groq' (free, 2000/day) — instant cloud transcription, no download",
            "medium on cpu if accuracy matters more than speed and you want to stay offline",
        ]
        return Recommendation(
            engine="local",
            name="small",
            device="cpu",
            compute_type="int8",
            reason=f"no NVIDIA GPU detected; {specs.cpu_cores} cores / {specs.ram_gb:.0f}GB RAM "
            "can run small on CPU (expect a few seconds per dictation)",
            alternatives=alts,
        )
```

In the weak-machine cloud branch (`:283-293`), change:

```python
    if has_api_key:
        return Recommendation(
            engine="gemini",
            name="gemini-2.5-flash",
            device="cpu",
            compute_type="int8",
            reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
            "is too weak for a good local model — BYOK cloud is the honest recommendation "
            "(note: audio leaves the machine)",
            alternatives=["small on cpu (fully private but slow and less accurate)"],
        )
```

to:

```python
    if has_api_key:
        return Recommendation(
            engine="groq",
            name="whisper-large-v3-turbo",
            device="cpu",
            compute_type="int8",
            reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
            "is too weak for a good local model — free cloud (Groq) is the honest recommendation "
            "(note: audio leaves the machine)",
            alternatives=["small on cpu (fully private but slow and less accurate)"],
        )
```

The final fallback (`whisperflow/sysinfo.py:295-303`, reached when `has_api_key` is False) currently reads:

```python
    return Recommendation(
        engine="local",
        name="small",
        device="cpu",
        compute_type="int8",
        reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
        "will be slow — consider [model].engine='gemini' with your own API key for better quality",
        alternatives=["engine='gemini' (set GEMINI_API_KEY) — better accuracy, but audio goes to Google"],
    )
```

Change it to (stays `engine="local"` — but now surfaces Groq as a free-to-sign-up alternative even without a pre-existing key, per the new test):

```python
    return Recommendation(
        engine="local",
        name="small",
        device="cpu",
        compute_type="int8",
        reason=f"this machine ({specs.cpu_cores} cores, {specs.ram_gb:.0f}GB RAM, no NVIDIA GPU) "
        "will run small slowly — consider a free Groq key for instant cloud transcription instead",
        alternatives=["engine='groq' — free, 2000 requests/day, no local download needed"],
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_sysinfo.py -v`
Expected: PASS (all recommend-ladder tests, including the 4 updated/added ones)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add whisperflow/sysinfo.py tests/test_sysinfo.py
git commit -m "recommend(): weak/GPU-less machines suggest free Groq cloud instead of Gemini"
```

---

### Task 6: Gemini default model → `gemini-2.5-flash-lite`

**Files:**
- Modify: `whisperflow/config.py:67` (`ModelConfig.cloud_model` default)
- Test: `tests/test_gemini_engine.py` (update one assertion)

**Interfaces:** none new — this is a one-field default change plus a matching test update.

- [ ] **Step 1: Update the test**

In `tests/test_gemini_engine.py::test_transcribe_parses_response` (`tests/test_gemini_engine.py:81`), change:

```python
    assert "gemini-2.5-flash" in captured["url"]
```

to:

```python
    assert "gemini-2.5-flash-lite" in captured["url"]
```

(This works because `cfg()` in that test file builds a plain `ModelConfig()` with no `cloud_model` override, so it picks up the dataclass default.)

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_gemini_engine.py::test_transcribe_parses_response -v`
Expected: FAIL — `captured["url"]` still contains `gemini-2.5-flash` (not `-lite`) because production default hasn't changed yet.

- [ ] **Step 3: Update the default**

In `whisperflow/config.py`, in `ModelConfig` (line 67):

```python
    cloud_model: str = "gemini-2.5-flash"
```

to:

```python
    cloud_model: str = "gemini-2.5-flash-lite"  # ~3x cheaper than -flash, same audio-input support
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_gemini_engine.py -v`
Expected: PASS (all Gemini tests)

- [ ] **Step 5: Run the full suite**

Run: `python -m pytest -q`
Expected: all pass — this also updates the auto-generated `config.toml` comment text (`serialize_config` interpolates `m.cloud_model`, no template change needed) and `bootstrap_config()`'s output for any future `Config()` default construction.

- [ ] **Step 6: Commit**

```bash
git add whisperflow/config.py tests/test_gemini_engine.py
git commit -m "Default Gemini STT model to gemini-2.5-flash-lite (3x cheaper, same capability)"
```

---

### Task 7: Manual smoke-test script (real API calls, not part of the unit suite)

**Files:**
- Create: `scripts/test_cloud_stt.py`

**Interfaces:**
- Consumes: `whisperflow.stt.registry.create_engine`, `whisperflow.stt.providers.cloud_providers`, `whisperflow.config.ModelConfig`.
- Produces: a standalone script (not pytest-collected — no `test_` prefix conflicts since it lives in `scripts/`, matching the existing `scripts/test_*.py` manual-smoke-test convention noted in the Explore-agent's earlier recon).

- [ ] **Step 1: Write the script**

```python
# scripts/test_cloud_stt.py
"""Manual smoke test: transcribe 1s of silence through every configured
cloud provider with a real API key. NOT part of the pytest suite (hits
real network APIs, costs real quota/money for paid providers).

Usage:
    set GROQ_API_KEY=...      (or GEMINI_API_KEY / OPENAI_API_KEY / DEEPGRAM_API_KEY)
    python scripts/test_cloud_stt.py groq
    python scripts/test_cloud_stt.py          # tries every provider with a key present
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from whisperflow.config import ModelConfig
from whisperflow.stt.providers import cloud_providers
from whisperflow.stt.registry import create_engine


def try_provider(provider_id: str) -> None:
    from whisperflow.stt import providers

    provider = providers.get(provider_id)
    key = os.environ.get(provider.api_key_env, "")
    if not key:
        print(f"SKIP {provider_id}: ${provider.api_key_env} not set")
        return
    cfg = ModelConfig(engine=provider_id)
    engine = create_engine(cfg)
    try:
        engine.load()
        # 1s of near-silence (small noise so it's not pure zeros, some APIs
        # reject dead-silent audio as invalid)
        audio = (np.random.randn(16000) * 0.001).astype(np.float32)
        result = engine.transcribe(audio, language="en")
        print(f"OK   {provider_id}: transcribed {result.duration_s:.1f}s in "
              f"{result.transcribe_seconds:.2f}s -> {result.text!r}")
    except Exception as exc:  # noqa: BLE001 — smoke test, report and continue
        print(f"FAIL {provider_id}: {exc}")


def main() -> None:
    targets = sys.argv[1:] or [p.id for p in cloud_providers()]
    for provider_id in targets:
        try_provider(provider_id)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it manually with at least one real key** (not gated by CI/pytest)

Run (with a real Groq key from console.groq.com/keys):
```
set GROQ_API_KEY=gsk_your_real_key_here
python scripts/test_cloud_stt.py groq
```
Expected output: `OK   groq: transcribed 1.0s in 0.XXs -> '...'` (some short/empty transcript of near-silent audio is fine — the point is no exception and a real HTTP round trip).

- [ ] **Step 3: Commit**

```bash
git add scripts/test_cloud_stt.py
git commit -m "Add manual smoke-test script for cloud STT providers"
```

---

## Self-Review Notes (completed during plan authoring)

- **Spec coverage:** Provider registry (Task 1) ✓, generic OpenAI-compatible engine covering Groq+OpenAI (Task 2) ✓, Deepgram engine (Task 3) ✓, `create_engine` dispatch on `kind` (Task 4) ✓, config validation against registry (Task 4) ✓, `recommend()` suggesting Groq (Task 5) ✓, Gemini default model fix (Task 6) ✓, manual smoke script (Task 7) ✓. NVIDIA is intentionally registry-absent in this phase (spec: "documented-only (gRPC; deferred)") — no task needed yet.
- **Placeholder scan:** none found — every step has literal code, exact file paths, exact commands.
- **Type consistency:** `RawResult`/`SttEngine` used identically across Tasks 2-3 (matches `base.py`). `Provider` fields referenced consistently across Tasks 1-4 (`kind`, `base_url`, `default_model`, `api_key_env`, `display_name`). `create_engine(cfg: ModelConfig) -> SttEngine` signature unchanged from the original, preserving every existing call site (`app.py:build_controller`, `app.py:print_recommendation`, all engine tests).
