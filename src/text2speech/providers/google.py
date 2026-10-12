"""Google Gemini Interactions API TTS provider.

Uses the Interactions API (``POST {base_url}/v1beta/interactions``) with
``x-goog-api-key`` authentication — the API shape documented at
https://aistudio.google.com/docs/speech-generation ::

    {
      "model": "gemini-3.8-flash-tts",
      "input": [{"type": "user_input", "content": [{
          "type": "text", "text": "...",
          "annotations": [{"type": "speech_metadata", "style": "..."}]}]}],
      "response_format": {"type": "audio"},
      "generation_config": {"speech_config": [{"voice": "Kore"}]}
    }

The reply is an Interaction resource whose ``steps`` contain the audio as a
base64-encoded WAV (unary requests) inside ``model_output`` steps.
"""

from __future__ import annotations

import base64
from typing import Any

import httpx

from ..config import ProviderConfig, get_settings
from .base import ProviderError, SynthesisResult, error_message_from_response

# Google mime types for the formats this provider can produce.
_FORMAT_MIME_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/l16",
}
_FILE_EXTENSIONS = {"wav": "wav", "pcm": "pcm"}


class GoogleInteractionsProvider:
    """Calls the Gemini Interactions API for text-to-speech."""

    type = "google"
    supported_formats = frozenset(_FORMAT_MIME_TYPES)

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

    def __enter__(self) -> "GoogleInteractionsProvider":
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()

    # -- request building ---------------------------------------------------

    def _build_payload(
        self, text: str, *, model: str, voice: str | None, response_format: str
    ) -> dict[str, Any]:
        fmt = response_format.lower().lstrip(".")
        if fmt not in _FORMAT_MIME_TYPES:
            raise ProviderError(
                f"Provider '{self.config.id}' (Google) cannot produce '{response_format}'. "
                f"Supported formats: {', '.join(sorted(_FORMAT_MIME_TYPES))}."
            )

        content: dict[str, Any] = {"type": "text", "text": text}
        style = (self.config.style or "").strip()
        if style:
            # Turn-level delivery instructions live in speech_metadata, never in
            # the transcript itself (the text field is read verbatim).
            content["annotations"] = [{"type": "speech_metadata", "style": style}]

        response_format_body: dict[str, Any] = {"type": "audio"}
        if fmt != "wav":
            response_format_body["mime_type"] = _FORMAT_MIME_TYPES[fmt]

        speech_voice = (voice or self.config.default_voice or "").strip()
        speech_config: list[dict[str, Any]] = (
            [{"voice": speech_voice}] if speech_voice else []
        )

        payload: dict[str, Any] = {
            "model": model,
            "input": [{"type": "user_input", "content": [content]}],
            "response_format": response_format_body,
            "generation_config": {"speech_config": speech_config},
        }
        # extra_body lets users add fields like {"store": false} without code changes.
        payload.update(self.config.extra_body)
        return payload

    # -- response parsing ----------------------------------------------------

    @staticmethod
    def _extract_audio(payload: dict[str, Any]) -> tuple[bytes, str | None]:
        """Pull the last audio block out of an Interaction response."""
        audio_b64: str | None = None
        for step in payload.get("steps") or []:
            if not isinstance(step, dict) or step.get("type") != "model_output":
                continue
            for part in step.get("content") or []:
                if isinstance(part, dict) and part.get("type") == "audio" and part.get("data"):
                    audio_b64 = part["data"]
        if not audio_b64:
            raise ProviderError(
                "Google response contained no audio content "
                f"(steps: {[s.get('type') for s in payload.get('steps') or []]})."
            )
        try:
            audio = base64.b64decode(audio_b64)
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(f"Google returned undecodable base64 audio: {exc}") from exc
        generation_id = payload.get("id") or payload.get("interactionId")
        return audio, generation_id

    # -- main entry point -----------------------------------------------------

    def synthesize(
        self,
        text: str,
        *,
        model: str,
        voice: str | None = None,
        response_format: str = "wav",
        speed: float | None = None,
        timeout: float | None = None,
    ) -> SynthesisResult:
        # `speed` has no equivalent in the Interactions TTS API and is ignored.
        if not self.api_key:
            raise ProviderError(
                f"Provider '{self.config.id}' has no API key. "
                f"Set the {self.config.api_key_env or '<api key>'} environment variable."
            )
        if not text.strip():
            raise ProviderError("Cannot synthesize empty text.")

        payload = self._build_payload(
            text, model=model, voice=voice, response_format=response_format
        )
        headers = {
            "x-goog-api-key": self.api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        headers.update(self.config.resolved_extra_headers())

        url = f"{self.config.base_url.rstrip('/')}/v1beta/interactions"
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
        try:
            body = response.json()
        except Exception as exc:  # noqa: BLE001
            raise ProviderError(
                f"Provider '{self.config.id}' returned a non-JSON response."
            ) from exc

        audio, generation_id = self._extract_audio(body)
        if not audio:
            raise ProviderError(f"Provider '{self.config.id}' returned an empty audio body.")

        fmt = response_format.lower().lstrip(".")
        content_type = response.headers.get("content-type") or _FORMAT_MIME_TYPES.get(
            fmt, "audio/wav"
        )
        return SynthesisResult(
            audio=audio,
            content_type=content_type,
            generation_id=generation_id,
            raw_headers=dict(response.headers),
        )
