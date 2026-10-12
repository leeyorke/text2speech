"""Base types for TTS providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from ..config import ProviderConfig


class ProviderError(RuntimeError):
    """Raised when a provider call fails (network, auth, bad request...)."""


@dataclass(slots=True)
class SynthesisResult:
    """Successful synthesis output plus useful metadata."""

    audio: bytes
    content_type: str
    generation_id: str | None = None
    raw_headers: dict[str, str] = field(default_factory=dict)


class TTSProvider(Protocol):
    """Interface every provider backend implements."""

    config: ProviderConfig

    def synthesize(
        self,
        text: str,
        *,
        model: str,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float | None = None,
        timeout: float | None = None,
    ) -> SynthesisResult: ...


def error_message_from_response(response: Any) -> str:
    """Best-effort extraction of an error message from an httpx response."""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                return str(error.get("message") or error)
            if error:
                return str(error)
            if payload.get("message"):
                return str(payload["message"])
    except Exception:
        pass
    body = (response.text or "").strip()
    return body[:500] if body else f"HTTP {response.status_code}"
