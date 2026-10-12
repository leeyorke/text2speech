"""OpenAI-compatible ``POST /audio/speech`` provider.

This is the backend used by OpenRouter (the reference implementation from the
project requirements) and any other service exposing the same API shape::

    POST {base_url}/audio/speech
    Authorization: Bearer <API_KEY>
    {"model": "...", "input": "...", "voice": "...", "response_format": "mp3"}
"""

from __future__ import annotations

from typing import Any

import httpx

from ..config import ProviderConfig, get_settings
from .base import ProviderError, SynthesisResult, error_message_from_response

_FORMAT_CONTENT_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}


class OpenAICompatibleProvider:
    """Calls an OpenAI-compatible text-to-speech endpoint."""

    type = "openai-compatible"
    # None = whatever response_format values the endpoint accepts.
    supported_formats = None

    def __init__(
        self,
        config: ProviderConfig,
        *,
        api_key: str | None = None,
        client: httpx.Client | None = None,
    ) -> None:
        self.config = config
        self.api_key = api_key if api_key is not None else config.api_key
        self._client = client

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            try:
                # Respect HTTP(S)_PROXY / NO_PROXY from the environment...
                self._client = httpx.Client()
            except Exception:
                # ...but fall back when the environment itself is unparseable
                # (e.g. httpx 0.28.x chokes on IPv6 entries like "[::1]" in NO_PROXY).
                self._client = httpx.Client(trust_env=False)
        return self._client

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> "OpenAICompatibleProvider":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    def synthesize(
        self,
        text: str,
        *,
        model: str,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float | None = None,
        timeout: float | None = None,
    ) -> SynthesisResult:
        if not self.api_key:
            raise ProviderError(
                f"Provider '{self.config.id}' has no API key. "
                f"Set the {self.config.api_key_env or '<api key>'} environment variable."
            )
        if not text.strip():
            raise ProviderError("Cannot synthesize empty text.")

        payload: dict[str, Any] = {
            "model": model,
            "input": text,
            "response_format": response_format,
        }
        if voice:
            payload["voice"] = voice
        if speed is not None:
            payload["speed"] = speed
        payload.update(self.config.extra_body)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": _FORMAT_CONTENT_TYPES.get(response_format, "audio/*"),
        }
        headers.update(self.config.resolved_extra_headers())

        url = f"{self.config.base_url.rstrip('/')}/audio/speech"
        settings = get_settings()
        try:
            response = self.client.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout or settings.request_timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError(f"Provider '{self.config.id}' timed out: {exc}") from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Provider '{self.config.id}' request failed: {exc}") from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"Provider '{self.config.id}' returned HTTP {response.status_code}: "
                f"{error_message_from_response(response)}"
            )
        if not response.content:
            raise ProviderError(f"Provider '{self.config.id}' returned an empty audio body.")

        generation_id = response.headers.get("x-generation-id") or response.headers.get(
            "x-request-id"
        )
        return SynthesisResult(
            audio=response.content,
            content_type=response.headers.get("content-type")
            or _FORMAT_CONTENT_TYPES.get(response_format, "application/octet-stream"),
            generation_id=generation_id,
            raw_headers=dict(response.headers),
        )
