"""Tests for the chunked synthesis orchestration."""

from __future__ import annotations

import pytest

from text2speech.config import ProviderConfig
from text2speech.providers.base import ProviderError, SynthesisResult
from text2speech.synthesis import synthesize_chunked


class FakeProvider:
    """Records every call; optionally fails on the Nth call."""

    def __init__(self, fail_on_call: int | None = None) -> None:
        self.config = ProviderConfig(
            id="fake", name="Fake", base_url="https://fake.invalid"
        )
        self.calls: list[dict] = []
        self.fail_on_call = fail_on_call

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
        self.calls.append({"text": text, "model": model, "voice": voice})
        if self.fail_on_call is not None and len(self.calls) == self.fail_on_call:
            raise ProviderError("boom on this chunk")
        return SynthesisResult(
            audio=f"AUDIO-{len(self.calls)}".encode(),
            content_type="audio/mpeg",
            generation_id=f"gen-{len(self.calls)}",
        )


def run(text: str, *, chunk_chars: int, fail_on_call: int | None = None):
    provider = FakeProvider(fail_on_call=fail_on_call)
    events: list[tuple[str, str, int, dict]] = []
    result = synthesize_chunked(
        provider,
        text,
        model="m",
        response_format="mp3",
        chunk_chars=chunk_chars,
        on_event=lambda stage, message, progress, extra: events.append(
            (stage, message, progress, extra)
        ),
    )
    return provider, result, events


TEXT = " ".join(f"Sentence number {i} is here." for i in range(30))


class TestSingleChunkPath:
    def test_short_text_makes_one_call_and_no_events(self):
        provider, result, events = run("A short sentence.", chunk_chars=1000)
        assert len(provider.calls) == 1
        assert events == []
        assert result.chunked is False
        assert result.chunks and len(result.chunks) == 1
        assert result.generation_id == "gen-1"
        assert result.audio == b"AUDIO-1"

    def test_text_exactly_at_threshold_is_single(self):
        provider, result, _ = run("x" * 50, chunk_chars=50)
        assert len(provider.calls) == 1
        assert result.chunked is False


class TestChunkedPath:
    def test_long_text_is_split_and_queued(self):
        provider, result, events = run(TEXT, chunk_chars=100)
        assert len(provider.calls) > 1
        assert all(len(c["text"]) <= 100 for c in provider.calls)
        assert result.chunked is True
        assert len(result.chunks) == len(provider.calls)
        assert result.characters == len(TEXT)

    def test_text_is_conserved_across_chunks(self):
        provider, _, _ = run(TEXT, chunk_chars=100)
        joined = " ".join(c["text"] for c in provider.calls)
        assert joined.split() == TEXT.split()

    def test_audio_parts_are_merged_in_order(self):
        _, result, _ = run(TEXT, chunk_chars=100)
        expected = b"".join(f"AUDIO-{i}".encode() for i in range(1, len(result.chunks) + 1))
        assert result.audio == expected

    def test_generation_ids_collected(self):
        _, result, _ = run(TEXT, chunk_chars=100)
        assert result.generation_ids == [f"gen-{i}" for i in range(1, len(result.chunks) + 1)]
        assert result.generation_id == "gen-1"

    def test_event_sequence(self):
        _, _, events = run(TEXT, chunk_chars=100)
        stages = [e[0] for e in events]
        assert stages[0] == "chunk_plan"
        assert stages[-1] == "merge"
        assert stages.count("chunk_start") == stages.count("chunk_done")
        assert "waiting" not in stages  # heartbeats are the transport's job
        progresses = [e[2] for e in events]
        assert progresses[0] == 25
        assert progresses[-1] == 90
        assert all(b >= a for a, b in zip(progresses, progresses[1:]))

    def test_chunk_plan_reports_counts(self):
        _, _, events = run(TEXT, chunk_chars=100)
        stage, message, _, extra = events[0]
        assert stage == "chunk_plan"
        assert "split into" in message
        assert extra["chunks"] > 1

    def test_options_forwarded_to_every_chunk(self):
        provider, _, _ = run(TEXT, chunk_chars=100)
        assert all(c["model"] == "m" for c in provider.calls)

    def test_failure_aborts_with_chunk_number(self):
        provider = FakeProvider(fail_on_call=2)
        with pytest.raises(ProviderError, match=r"Chunk 2/\d+ failed: boom on this chunk"):
            synthesize_chunked(
                provider, TEXT, model="m", response_format="mp3", chunk_chars=100
            )
        assert len(provider.calls) == 2  # stopped at the failing chunk

    def test_failure_emits_chunk_failed_event(self):
        events: list[tuple[str, str, int, dict]] = []
        provider = FakeProvider(fail_on_call=1)
        with pytest.raises(ProviderError):
            synthesize_chunked(
                provider,
                TEXT,
                model="m",
                response_format="mp3",
                chunk_chars=100,
                on_event=lambda s, m, p, e: events.append((s, m, p, e)),
            )
        assert events[-1][0] == "chunk_failed"
