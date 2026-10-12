"""Tests for the Google (Gemini Interactions API) provider — mocked transport."""

from __future__ import annotations

import base64
import json

import httpx
import pytest

from text2speech.config import ProviderConfig
from text2speech.providers.base import ProviderError
from text2speech.providers.google import GoogleInteractionsProvider


def make_config(**overrides) -> ProviderConfig:
    defaults = dict(
        id="google",
        name="Google Gemini",
        type="google",
        base_url="https://generativelanguage.googleapis.com",
        api_key_env="TEST_GEMINI_API_KEY",
        models=["gemini-3.8-flash-tts"],
        default_model="gemini-3.8-flash-tts",
        default_voice="Kore",
    )
    defaults.update(overrides)
    return ProviderConfig(**defaults)


def make_provider(handler, config=None, api_key="goog-key"):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    return GoogleInteractionsProvider(config or make_config(), api_key=api_key, client=client)


def interactions_response(
    audio: bytes = b"RIFFxxxxWAVEDATA",
    *,
    interaction_id: str = "interaction-123",
    extra_steps: list | None = None,
    extra_audio: bytes | None = None,
) -> httpx.Response:
    content = [{"type": "text", "text": "echo"}, {"type": "audio", "data": base64.b64encode(audio).decode()}]
    if extra_audio is not None:
        content.append({"type": "audio", "data": base64.b64encode(extra_audio).decode()})
    steps = [{"type": "thought", "content": [{"type": "text", "text": "hmm"}]}]
    steps.extend(extra_steps or [])
    steps.append({"type": "model_output", "content": content})
    return httpx.Response(
        200, json={"id": interaction_id, "steps": steps}, headers={"content-type": "application/json"}
    )


class TestPayloadShape:
    def test_request_url_headers_and_body(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["url"] = str(request.url)
            captured["headers"] = dict(request.headers)
            captured["json"] = json.loads(request.content)
            return interactions_response()

        provider = make_provider(handler)
        result = provider.synthesize("Have a wonderful day!", model="gemini-3.8-flash-tts")

        assert captured["url"] == "https://generativelanguage.googleapis.com/v1beta/interactions"
        assert captured["headers"]["x-goog-api-key"] == "goog-key"
        assert captured["json"] == {
            "model": "gemini-3.8-flash-tts",
            "input": [
                {
                    "type": "user_input",
                    "content": [{"type": "text", "text": "Have a wonderful day!"}],
                }
            ],
            "response_format": {"type": "audio"},
            "generation_config": {"speech_config": [{"voice": "Kore"}]},
        }
        assert result.audio == b"RIFFxxxxWAVEDATA"
        assert result.generation_id == "interaction-123"
        assert result.content_type == "application/json"

    def test_voice_override(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return interactions_response()

        provider = make_provider(handler)
        provider.synthesize("hi", model="m", voice="Puck")
        assert captured["json"]["generation_config"]["speech_config"] == [{"voice": "Puck"}]

    def test_style_annotation_from_config(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return interactions_response()

        config = make_config(style="whispered urgently")
        provider = make_provider(handler, config=config)
        provider.synthesize("hi", model="m")
        content = captured["json"]["input"][0]["content"][0]
        assert content["annotations"] == [{"type": "speech_metadata", "style": "whispered urgently"}]

    def test_no_style_annotation_when_unset(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return interactions_response()

        provider = make_provider(handler)  # style defaults to None
        provider.synthesize("hi", model="m")
        assert "annotations" not in captured["json"]["input"][0]["content"][0]

    def test_no_voice_leaves_empty_speech_config(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return interactions_response()

        provider = make_provider(handler, config=make_config(default_voice=None))
        provider.synthesize("hi", model="m")
        assert captured["json"]["generation_config"] == {"speech_config": []}

    def test_pcm_format_uses_l16_mime(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return interactions_response()

        provider = make_provider(handler)
        provider.synthesize("hi", model="m", response_format="pcm")
        assert captured["json"]["response_format"] == {"type": "audio", "mime_type": "audio/l16"}

    def test_wav_format_is_default_mime(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return interactions_response()

        provider = make_provider(handler)
        provider.synthesize("hi", model="m", response_format="wav")
        assert captured["json"]["response_format"] == {"type": "audio"}

    def test_extra_body_merged(self):
        captured = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured["json"] = json.loads(request.content)
            return interactions_response()

        provider = make_provider(handler, config=make_config(extra_body={"store": False}))
        provider.synthesize("hi", model="m")
        assert captured["json"]["store"] is False


class TestResponseParsing:
    def test_last_audio_block_wins(self):
        provider = make_provider(lambda r: interactions_response(extra_audio=b"SECOND-AUDIO"))
        result = provider.synthesize("hi", model="m")
        assert result.audio == b"SECOND-AUDIO"

    def test_non_model_output_steps_ignored(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return interactions_response(
                extra_steps=[{"type": "tool_call", "content": [{"type": "audio", "data": "AAAA"}]}]
            )

        provider = make_provider(handler)
        result = provider.synthesize("hi", model="m")
        assert result.audio == b"RIFFxxxxWAVEDATA"

    def test_interaction_id_fallback(self):
        def handler(request: httpx.Request) -> httpx.Response:
            response = interactions_response()
            body = json.loads(response.content)
            body["interactionId"] = "fallback-id"
            del body["id"]
            return httpx.Response(200, json=body)

        provider = make_provider(handler)
        assert provider.synthesize("hi", model="m").generation_id == "fallback-id"


class TestErrors:
    def test_unsupported_format_raises_before_request(self):
        def handler(request: httpx.Request) -> httpx.Response:
            raise AssertionError("should not be called")

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="cannot produce 'mp3'"):
            provider.synthesize("hi", model="m", response_format="mp3")

    def test_http_error_message(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(401, json={"error": {"message": "API key not valid"}})

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="HTTP 401.*API key not valid"):
            provider.synthesize("hi", model="m")

    def test_no_audio_in_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200, json={"id": "x", "steps": [{"type": "model_output", "content": [{"type": "text", "text": "no audio"}]}]}
            )

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="no audio content"):
            provider.synthesize("hi", model="m")

    def test_bad_base64(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "id": "x",
                    "steps": [
                        {"type": "model_output", "content": [{"type": "audio", "data": "!!!not-base64!!!"}]}
                    ],
                },
            )

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="undecodable base64"):
            provider.synthesize("hi", model="m")

    def test_non_json_response(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<html>gateway error</html>")

        provider = make_provider(handler)
        with pytest.raises(ProviderError, match="non-JSON"):
            provider.synthesize("hi", model="m")

    def test_missing_api_key(self):
        provider = GoogleInteractionsProvider(make_config(), api_key=None)
        with pytest.raises(ProviderError, match="no API key"):
            provider.synthesize("hi", model="m")

    def test_empty_text(self):
        provider = make_provider(lambda r: interactions_response())
        with pytest.raises(ProviderError, match="empty text"):
            provider.synthesize("   ", model="m")
