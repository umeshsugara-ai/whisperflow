"""STT engine interface — the seam for pluggable models — plus the shared
HTTP plumbing every cloud engine routes through (`request_json`,
`check_upload_size`): one place for retry-on-network-blip, upload-size
guards, and friendly 401/429 error mapping instead of five copies.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RawResult:
    text: str
    language: str
    language_probability: float
    duration_s: float
    transcribe_seconds: float  # wall-clock inference time


def check_upload_size(nbytes: int, provider) -> None:
    """Fail BEFORE uploading when the audio exceeds the provider's request
    limit — a clear "too long" message beats the provider's raw HTTP 413.
    `provider.max_upload_bytes == 0` means no known limit (skip the check)."""
    limit = getattr(provider, "max_upload_bytes", 0)
    if limit and nbytes > limit:
        raise RuntimeError(
            f"This recording ({nbytes / 1e6:.0f}MB) is too long for "
            f"{provider.display_name} (max {limit / 1e6:.0f}MB) — try a shorter "
            "dictation, or switch engines in Settings → Speech engine."
        )


def request_json(
    req: urllib.request.Request,
    *,
    provider_name: str,
    signup_url: str = "",
    timeout: float = 30.0,
) -> dict:
    """Send `req`, decode the JSON response.

    - One retry (1s backoff) on transient network failure — a single flaky
      packet must not throw away the user's dictation.
    - 401 maps to a friendly "key invalid — regenerate at <signup_url>"
      message; 429 to a "rate limit" one. Other HTTP errors keep the raw
      status + body excerpt (403 stays raw on purpose: on Cloudflare-fronted
      APIs it can mean bot-blocking, not a bad key).
    """
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            if exc.code == 401:
                hint = f" Get a new key at {signup_url} and" if signup_url else " Please"
                raise RuntimeError(
                    f"{provider_name} rejected your API key (it may be invalid, "
                    f"expired, or revoked).{hint} update it in Settings → Speech engine."
                ) from exc
            if exc.code == 429:
                raise RuntimeError(
                    f"{provider_name} rate limit reached — wait a minute and try "
                    "again, or switch engines in Settings → Speech engine."
                ) from exc
            raise RuntimeError(f"{provider_name} API error {exc.code}: {detail}") from exc
        except (urllib.error.URLError, OSError) as exc:
            last_exc = exc
            if attempt == 1:
                log.warning("%s request failed (%s) — retrying once", provider_name, exc)
                time.sleep(1.0)
    raise RuntimeError(f"{provider_name} API unreachable: {last_exc}") from last_exc


class SttEngine(ABC):
    @abstractmethod
    def load(self) -> None:
        """Load model weights (called once at daemon startup)."""

    @abstractmethod
    def transcribe(
        self,
        audio: np.ndarray,  # float32 mono @16k
        language: str = "",  # "" = auto-detect
        initial_prompt: str = "",  # vocabulary bias
    ) -> RawResult: ...
