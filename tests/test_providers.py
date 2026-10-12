"""Tests for the OpenAI-compatible provider client (mocked transport)."""

from __future__ import annotations

import httpx
import pytest

from text2speech.config import ProviderConfig
from text2speech.providers.base import ProviderError
from text2speech.providers.openai_compatible import OpenAICompatibleProvider


def make_config(**overrides) -> ProviderConfig:
    defaults = dict(
        id="test",
        name="Test",
        type="openai-compatible",
        base_url="https://tts.example.com/api/v1",
        api_key_env="TEST_TTS_API_KEY",
        models=["model-a"],
        default_model="model-a",
        extra_headers={"X-Custom": "abc", "X-Interp": "{env:TEST_TTS_HEADER}"},
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def make_provider(handler, config=None, api_key="secret-key", monkeypatch=None):
    if monkeypatch is not None:
        monkeypatch.setenv("TEST_TTS_HEADER", "interpolated")
    transport = httpx.MockTransport(handler)
    client = httpx.Client(transport=transport)
    provider = OpenAICompatibleProvider(config or make_config(), api_key=api_key, client=client)
    return provider


class TestSynthesize:
    def test_success_payload_and_headers(self, monkeypatch):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            import json

            captured["json"] = json.loads(request.content)
            return httpx.Response(
                200,
                content=b"ID3AUDIOBYTES",
                headers={
                    "content-type": "audio/mpeg",
                    "x-generation-id": "gen-123",
                },
            )

        provider = make_provider(handler, monkeypatch=monkeypatch)
        result = provider.synthesize("hello world", model="model-a", response_format="mp3")

        assert captured["url"] == "https://tts.example.com/api/v1/audio/speech"
        assert captured["headers"]["authorization"] == "Bearer secret-key"
        assert captured["headers"]["x-custom"] == "abc"
        assert captured["headers"]["x-interp"] == "interpolated"
        assert captured["json"] == {
            "model": "model-a",
            "input": "hello world",
            "response_format": "mp3",
        }
        assert result.audio == b"ID3AUDIOBYTES"
        assert result.generation_id == "gen-123"
        assert result.content_type == "audio/mpeg"

    def test_optional_voice_and_speed(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["json"] = json.loads(request.content)
            return httpx.Response(200, content=b"audio")

        provider = make_provider(handler)
        provider.synthesize("hi", model="m", voice="nova", speed=1.5, response_format="wav")
        assert captured["json"]["voice"] == "nova"
        assert captured["json"]["speed"] == 1.5
        assert captured["json"]["response_format"] == "wav"

    def test_extra_body_merged(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            import json

            captured["json"] = json.loads(request.content)
            return httpx.Response(200, content=b"audio")

        config = make_config(extra_body={"language": "zh"})
        provider = make_provider(handler, config=config)
        provider.synthesize("hi", model="m")
        assert captured["json"]["language"] == "zh"

    def test_http_error_raises_provider_error_with_message(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": {"message": "Invalid API key"}},
            )

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="HTTP 401.*Invalid API key"):
            provider.synthesize("hi", model="m")

    def test_empty_body_raises(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, content=b"")

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="empty audio body"):
            provider.synthesize("hi", model="m")

    def test_missing_api_key_raises(self):
        provider = OpenAICompatibleProvider(make_config(), api_key=None)
        with pytest.raises(ProviderError, match="no API key"):
            provider.synthesize("hi", model="m")

    def test_empty_text_raises(self):
        provider = make_provider(lambda r: httpx.Response(200, content=b"a"))
        with pytest.raises(ProviderError, match="empty text"):
            provider.synthesize("   ", model="m")

    def test_network_error_raises_provider_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused")

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="request failed"):
            provider.synthesize("hi", model="m")
