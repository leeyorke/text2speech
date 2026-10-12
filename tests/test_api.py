"""End-to-end API tests using a mock provider backend."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from text2speech import main
from text2speech.config import get_settings
from text2speech.main import app
from text2speech.providers import register_provider_type, reset_cache
from text2speech.providers.base import ProviderError


class FakeProvider:
    """In-memory provider that records every synthesis request."""

    type = "mock"
    # When set, the Nth synthesize() call raises instead of returning audio.
    fail_on_call: int | None = None

    def __init__(self, config, api_key: str | None = None, client=None) -> None:
        self.config = config
        self.calls: list[dict] = []

    def synthesize(
        self,
        text: str,
        *,
        model: str,
        voice: str | None = None,
        response_format: str = "mp3",
        speed: float | None = None,
        timeout: float | None = None,
    ):
        from text2speech.providers.base import SynthesisResult

        if text.startswith("BOOM"):
            raise ProviderError("mock provider exploded")
        call_number = len(self.calls) + 1
        if self.fail_on_call is not None and call_number == self.fail_on_call:
            raise ProviderError(f"mock provider exploded on call {call_number}")
        self.calls.append(
            {
                "text": text,
                "model": model,
                "voice": voice,
                "response_format": response_format,
                "speed": speed,
            }
        )
        return SynthesisResult(
            audio=b"FAKE-AUDIO-DATA",
            content_type="audio/mpeg",
            generation_id=f"gen-fake-{call_number}",
        )


def _providers_json() -> dict:
    return {
        "providers": [
            {
                "id": "mock",
                "name": "Mock TTS",
                "type": "mock",
                "base_url": "https://mock.invalid",
                "api_key_env": "MOCK_TTS_API_KEY",
                "models": ["mock-model-1", "mock-model-2"],
                "default_model": "mock-model-1",
                "voices": ["voice-a"],
            },
            {
                "id": "keyless",
                "name": "Keyless Mock",
                "type": "mock",
                "base_url": "https://mock.invalid",
                "enabled": True,
            },
        ]
    }


def _make_client(tmp_path, monkeypatch, chunk_chars: int | None = None):
    providers_file = tmp_path / "providers.json"
    providers_file.write_text(json.dumps(_providers_json()), encoding="utf-8")
    outputs_dir = tmp_path / "outputs"

    monkeypatch.setenv("T2S_PROVIDERS_FILE", str(providers_file))
    monkeypatch.setenv("T2S_OUTPUTS_DIR", str(outputs_dir))
    monkeypatch.setenv("MOCK_TTS_API_KEY", "mock-key")
    if chunk_chars is not None:
        monkeypatch.setenv("T2S_CHUNK_CHARS", str(chunk_chars))
    register_provider_type("mock", FakeProvider)
    FakeProvider.fail_on_call = None
    get_settings.cache_clear()
    reset_cache()

    with TestClient(app) as test_client:
        yield test_client

    FakeProvider.fail_on_call = None
    get_settings.cache_clear()
    reset_cache()


@pytest.fixture()
def client(tmp_path, monkeypatch):
    yield from _make_client(tmp_path, monkeypatch)


@pytest.fixture()
def chunk_client(tmp_path, monkeypatch):
    """Client with a tiny chunk threshold so short texts get split."""
    yield from _make_client(tmp_path, monkeypatch, chunk_chars=60)


def last_provider() -> FakeProvider:
    from text2speech.providers import get_provider

    provider = get_provider("mock")
    assert isinstance(provider, FakeProvider)
    return provider


class TestMetaEndpoints:
    def test_health(self, client):
        res = client.get("/api/health")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "ok"
        assert body["providers"] == 2

    def test_index_html(self, client):
        res = client.get("/")
        assert res.status_code == 200
        assert "text2speech" in res.text

    def test_providers_listing(self, client):
        res = client.get("/api/providers")
        assert res.status_code == 200
        providers = {p["id"]: p for p in res.json()}
        assert providers["mock"]["has_api_key"] is True
        assert providers["mock"]["models"] == ["mock-model-1", "mock-model-2"]
        assert providers["mock"]["default_model"] == "mock-model-1"
        assert providers["keyless"]["has_api_key"] is False

    def test_providers_reload(self, client):
        res = client.post("/api/providers/reload")
        assert res.status_code == 200
        assert res.json() == {"providers": 2}


class TestTtsEndpoint:
    def test_pasted_text(self, client):
        res = client.post(
            "/api/tts",
            data={"provider_id": "mock", "text": "Hello from the API"},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["format"] == "mp3"
        assert body["model"] == "mock-model-1"
        assert body["generation_id"] == "gen-fake-1"
        assert body["characters"] == len("Hello from the API")
        assert body["kind"] == "txt"
        assert last_provider().calls[-1]["text"] == "Hello from the API"

        audio = client.get(body["url"])
        assert audio.status_code == 200
        assert audio.content == b"FAKE-AUDIO-DATA"

    def test_markdown_upload_is_converted(self, client):
        md = b"# Title\n\nSome **bold** text.\n\n```python\nprint('nope')\n```"
        res = client.post(
            "/api/tts",
            data={"provider_id": "mock", "include_code_blocks": "false"},
            files={"file": ("notes.md", md, "text/markdown")},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["kind"] == "markdown"
        assert body["source_name"] == "notes.md"
        sent = last_provider().calls[-1]["text"]
        assert "Title" in sent and "bold text" in sent
        assert "nope" not in sent
        assert "```" not in sent

    def test_markdown_upload_can_keep_code(self, client):
        md = b"```\nkeep this code\n```"
        res = client.post(
            "/api/tts",
            data={"provider_id": "mock", "include_code_blocks": "true"},
            files={"file": ("a.md", md, "text/markdown")},
        )
        assert res.status_code == 200
        assert "keep this code" in last_provider().calls[-1]["text"]

    def test_txt_upload(self, client):
        res = client.post(
            "/api/tts",
            data={"provider_id": "mock"},
            files={"file": ("a.txt", b"plain file text", "text/plain")},
        )
        assert res.status_code == 200
        assert res.json()["kind"] == "txt"
        assert last_provider().calls[-1]["text"] == "plain file text"

    def test_options_forwarded(self, client):
        res = client.post(
            "/api/tts",
            data={
                "provider_id": "mock",
                "text": "hi",
                "model": "mock-model-2",
                "voice": "voice-a",
                "response_format": "wav",
                "speed": "1.5",
            },
        )
        assert res.status_code == 200, res.text
        call = last_provider().calls[-1]
        assert call["model"] == "mock-model-2"
        assert call["voice"] == "voice-a"
        assert call["response_format"] == "wav"
        assert call["speed"] == 1.5
        assert res.json()["url"].endswith(".wav")

    def test_no_input_returns_400(self, client):
        res = client.post("/api/tts", data={"provider_id": "mock", "text": "  "})
        assert res.status_code == 400
        assert "No input" in res.json()["detail"]

    def test_unsupported_file_returns_400(self, client):
        res = client.post(
            "/api/tts",
            data={"provider_id": "mock"},
            files={"file": ("a.pdf", b"%PDF", "application/pdf")},
        )
        assert res.status_code == 400
        assert "Unsupported file type" in res.json()["detail"]

    def test_unknown_provider_returns_400(self, client):
        res = client.post("/api/tts", data={"provider_id": "nope", "text": "hi"})
        assert res.status_code == 400
        assert "Unknown provider" in res.json()["detail"]

    def test_provider_error_returns_502(self, client):
        res = client.post("/api/tts", data={"provider_id": "mock", "text": "BOOM now"})
        assert res.status_code == 502
        assert "mock provider exploded" in res.json()["detail"]

    def test_bad_format_returns_400(self, client):
        res = client.post(
            "/api/tts", data={"provider_id": "mock", "text": "hi", "response_format": "ogg"}
        )
        assert res.status_code == 400

    def test_outputs_listing_and_traversal_guard(self, client):
        res = client.post("/api/tts", data={"provider_id": "mock", "text": "list me"})
        filename = res.json()["filename"]

        listing = client.get("/api/outputs")
        assert listing.status_code == 200
        assert any(item["filename"] == filename for item in listing.json())

        assert client.get("/api/audio/..%5C..%5Cpyproject.toml").status_code == 400
        assert client.get("/api/audio/missing.mp3").status_code == 404


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        for line in block.split("\n"):
            if line.startswith("data:"):
                events.append(json.loads(line[len("data:"):].strip()))
    return events


class TestDeleteOutput:
    def test_delete_removes_file(self, client):
        res = client.post("/api/tts", data={"provider_id": "mock", "text": "delete me"})
        filename = res.json()["filename"]

        deleted = client.delete(f"/api/outputs/{filename}")
        assert deleted.status_code == 200
        assert deleted.json() == {"deleted": filename}
        assert client.get(f"/api/audio/{filename}").status_code == 404
        assert all(item["filename"] != filename for item in client.get("/api/outputs").json())

    def test_delete_missing_file_404(self, client):
        assert client.delete("/api/outputs/nope.mp3").status_code == 404

    def test_delete_traversal_rejected(self, client):
        assert client.delete("/api/outputs/..%5Cpyproject.toml").status_code == 400
        assert client.delete("/api/outputs/.env").status_code == 400


class TestStreamEndpoint:
    def test_stream_emits_progress_and_job(self, client):
        res = client.post(
            "/api/tts/stream",
            data={"provider_id": "mock", "text": "stream this"},
        )
        assert res.status_code == 200
        assert res.headers["content-type"].startswith("text/event-stream")

        events = parse_sse(res.text)
        stages = [e["stage"] for e in events]
        assert stages[0] == "extract"
        assert "submit" in stages
        assert stages[-1] == "done"
        assert all(0 <= e.get("progress", 0) <= 100 for e in events)
        assert any("characters" in e.get("message", "") for e in events)

        job = events[-1]["job"]
        assert job["characters"] == len("stream this")
        assert job["generation_id"] == "gen-fake-1"
        assert client.get(job["url"]).content == b"FAKE-AUDIO-DATA"
        assert last_provider().calls[-1]["text"] == "stream this"

    def test_stream_markdown_upload(self, client):
        res = client.post(
            "/api/tts/stream",
            data={"provider_id": "mock"},
            files={"file": ("notes.md", b"# Title\n\nbody", "text/markdown")},
        )
        assert res.status_code == 200
        events = parse_sse(res.text)
        assert events[-1]["stage"] == "done"
        assert events[-1]["job"]["kind"] == "markdown"
        sent = last_provider().calls[-1]["text"]
        assert "Title" in sent and "body" in sent and "#" not in sent

    def test_stream_error_event_on_bad_input(self, client):
        res = client.post("/api/tts/stream", data={"provider_id": "mock", "text": "  "})
        assert res.status_code == 200  # errors travel as SSE events, not HTTP status
        events = parse_sse(res.text)
        assert events[-1]["stage"] == "error"
        assert "No input" in events[-1]["message"]

    def test_stream_error_event_on_unknown_provider(self, client):
        res = client.post("/api/tts/stream", data={"provider_id": "ghost", "text": "hi"})
        events = parse_sse(res.text)
        assert events[-1]["stage"] == "error"
        assert "Unknown provider" in events[-1]["message"]

    def test_stream_error_event_on_provider_failure(self, client):
        res = client.post("/api/tts/stream", data={"provider_id": "mock", "text": "BOOM"})
        events = parse_sse(res.text)
        assert events[-1]["stage"] == "error"
        assert "mock provider exploded" in events[-1]["message"]

    def test_stream_options_forwarded(self, client):
        res = client.post(
            "/api/tts/stream",
            data={
                "provider_id": "mock",
                "text": "hi",
                "model": "mock-model-2",
                "response_format": "wav",
            },
        )
        events = parse_sse(res.text)
        assert events[-1]["stage"] == "done"
        assert events[-1]["job"]["url"].endswith(".wav")
        call = last_provider().calls[-1]
        assert call["model"] == "mock-model-2"
        assert call["response_format"] == "wav"


# ~700 characters: several chunks at the chunk_client threshold of 60.
CHUNKED_TEXT = " ".join(f"This is sentence number {i} of the test." for i in range(20))


class TestChunkedSynthesis:
    def test_long_text_is_chunked_and_merged(self, chunk_client):
        res = chunk_client.post(
            "/api/tts", data={"provider_id": "mock", "text": CHUNKED_TEXT}
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["chunked"] is True
        assert body["chunks"] > 1
        assert body["characters"] == len(CHUNKED_TEXT)

        calls = last_provider().calls
        assert len(calls) == body["chunks"]
        assert all(len(c["text"]) <= 60 for c in calls)
        assert " ".join(c["text"] for c in calls).split() == CHUNKED_TEXT.split()
        assert body["generation_ids"] == [f"gen-fake-{i}" for i in range(1, body["chunks"] + 1)]

        audio = chunk_client.get(body["url"])
        assert audio.status_code == 200
        assert audio.content == b"FAKE-AUDIO-DATA" * body["chunks"]

    def test_short_text_stays_single_chunk(self, chunk_client):
        res = chunk_client.post("/api/tts", data={"provider_id": "mock", "text": "tiny"})
        body = res.json()
        assert body["chunked"] is False
        assert body["chunks"] == 1
        assert len(last_provider().calls) == 1

    def test_markdown_upload_is_chunked(self, chunk_client):
        md = ("# Title\n\n" + "A markdown paragraph with several sentences to speak. " * 8).encode()
        res = chunk_client.post(
            "/api/tts",
            data={"provider_id": "mock"},
            files={"file": ("a.md", md, "text/markdown")},
        )
        assert res.status_code == 200, res.text
        body = res.json()
        assert body["kind"] == "markdown"
        assert body["chunked"] is True
        calls = last_provider().calls
        assert len(calls) == body["chunks"] > 1
        assert all("#" not in c["text"] for c in calls)

    def test_chunk_failure_returns_502_with_chunk_number(self, chunk_client):
        FakeProvider.fail_on_call = 2
        res = chunk_client.post(
            "/api/tts", data={"provider_id": "mock", "text": CHUNKED_TEXT}
        )
        assert res.status_code == 502
        assert "Chunk 2/" in res.json()["detail"]
        assert len(last_provider().calls) == 1  # the failing call is not recorded

    def test_stream_emits_chunk_events(self, chunk_client):
        res = chunk_client.post(
            "/api/tts/stream", data={"provider_id": "mock", "text": CHUNKED_TEXT}
        )
        assert res.status_code == 200
        events = parse_sse(res.text)
        stages = [e["stage"] for e in events]
        assert stages[0] == "extract"
        assert "chunk_plan" in stages
        assert stages.count("chunk_start") == stages.count("chunk_done")
        assert "merge" in stages
        assert stages[-1] == "done"

        job = events[-1]["job"]
        assert job["chunked"] is True
        assert job["chunks"] > 1
        assert len(job["generation_ids"]) == job["chunks"]
        progresses = [e["progress"] for e in events]
        assert progresses[0] == 15 and progresses[-1] == 100
        assert all(b >= a for a, b in zip(progresses, progresses[1:]))

    def test_stream_chunk_failure_error_event(self, chunk_client):
        FakeProvider.fail_on_call = 1
        res = chunk_client.post(
            "/api/tts/stream", data={"provider_id": "mock", "text": CHUNKED_TEXT}
        )
        events = parse_sse(res.text)
        assert events[-1]["stage"] == "error"
        assert "Chunk 1/" in events[-1]["message"]

    def test_health_exposes_thresholds(self, chunk_client):
        body = chunk_client.get("/api/health").json()
        assert body["chunk_chars"] == 60
        assert body["max_text_chars"] >= 60

    def test_over_hard_cap_still_rejected(self, chunk_client):
        res = chunk_client.post(
            "/api/tts",
            data={"provider_id": "mock", "text": "x" * 61},
        )
        # 61 chars > chunk threshold of 60 but far below the 200k hard cap:
        # it must be chunked, not rejected.
        assert res.status_code == 200
        assert res.json()["chunks"] >= 2


TEARDOWN_MESSAGE = (
    "Exception in callback _ProactorBasePipeTransport._call_connection_lost(None)"
)


class TestDeadConnectionNoiseFilter:
    """Windows prints a scary traceback when the browser drops a connection."""

    @staticmethod
    def _teardown_context(exc: BaseException) -> dict:
        return {"message": TEARDOWN_MESSAGE, "exception": exc, "handle": object()}

    @pytest.mark.parametrize(
        "exc",
        [
            ConnectionResetError(10054, "远程主机强迫关闭了一个现有的连接"),
            ConnectionAbortedError(10053, "软件导致连接中止"),
            OSError(10057, "由于套接字没有连接"),
            BrokenPipeError(),
        ],
    )
    def test_socket_teardown_errors_are_filtered(self, exc):
        assert main.is_dead_connection_noise(self._teardown_context(exc)) is True

    def test_errors_from_other_callbacks_are_not_filtered(self):
        context = {
            "message": "Exception in callback something_else()",
            "exception": ConnectionResetError(10054, "reset"),
        }
        assert main.is_dead_connection_noise(context) is False

    def test_non_oserror_from_teardown_is_not_filtered(self):
        assert main.is_dead_connection_noise(self._teardown_context(ValueError("boom"))) is False

    def test_context_without_message_is_not_filtered(self):
        assert main.is_dead_connection_noise({"exception": ConnectionResetError()}) is False

    def test_handler_swallows_noise_and_forwards_the_rest(self):
        loop = asyncio.new_event_loop()
        try:
            forwarded: list[dict] = []
            loop.set_exception_handler(lambda _loop, context: forwarded.append(context))
            main.install_dead_connection_filter(loop)
            handler = loop.get_exception_handler()

            handler(loop, self._teardown_context(ConnectionResetError(10054, "reset")))
            assert forwarded == []

            real = {
                "message": "Exception in callback something_else()",
                "exception": ValueError("boom"),
            }
            handler(loop, real)
            assert forwarded == [real]
        finally:
            loop.close()

    def test_default_handler_is_used_when_no_previous_handler(self):
        loop = asyncio.new_event_loop()
        try:
            main.install_dead_connection_filter(loop)
            handler = loop.get_exception_handler()
            seen: list[dict] = []
            loop.default_exception_handler = lambda context: seen.append(context)  # type: ignore[method-assign]

            handler(loop, self._teardown_context(ConnectionResetError(10054, "reset")))
            assert seen == []

            handler(loop, {"message": "Exception in callback something_else()", "exception": ValueError("boom")})
            assert len(seen) == 1
        finally:
            loop.close()

    def test_lifespan_installs_the_filter_and_restores_it_afterwards(self, caplog, monkeypatch, tmp_path):
        monkeypatch.setenv("T2S_OUTPUTS_DIR", str(tmp_path / "out"))
        get_settings.cache_clear()

        async def scenario() -> object:
            async with main.lifespan(app):
                loop = asyncio.get_running_loop()
                handler = loop.get_exception_handler()
                assert handler is not None
                with caplog.at_level(logging.ERROR, logger="asyncio"):
                    handler(loop, self._teardown_context(ConnectionResetError(10054, "reset")))
                assert TEARDOWN_MESSAGE not in caplog.text
            return loop.get_exception_handler()

        try:
            assert asyncio.run(scenario()) is None
        finally:
            get_settings.cache_clear()
