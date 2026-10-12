"""Chunked synthesis: split long text, synthesize each part in order, merge the audio.

Texts up to ``chunk_chars`` take the exact same path as before (a single provider
call). Longer texts are split on sentence boundaries, queued, converted one chunk at
a time and merged back into a single audio file.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .audio import AudioMergeError, merge_audio
from .providers.base import ProviderError
from .textio import split_text

# Progress callback: (stage, message, progress_percent, extra_dict)
EventCallback = Callable[[str, str, int, dict], None]

# Progress layout for chunked runs (single-chunk runs keep the original numbers).
_PLAN_PROGRESS = 25
_CHUNK_START_PROGRESS = 30
_CHUNK_END_PROGRESS = 85
_MERGE_PROGRESS = 90


@dataclass(slots=True)
class ChunkInfo:
    index: int
    total: int
    characters: int
    generation_id: str | None = None
    seconds: float = 0.0


@dataclass(slots=True)
class ChunkedResult:
    audio: bytes
    chunks: list[ChunkInfo] = field(default_factory=list)
    characters: int = 0

    @property
    def chunked(self) -> bool:
        return len(self.chunks) > 1

    @property
    def generation_ids(self) -> list[str]:
        return [c.generation_id for c in self.chunks if c.generation_id]

    @property
    def generation_id(self) -> str | None:
        ids = self.generation_ids
        return ids[0] if ids else None


def synthesize_chunked(
    provider,
    text: str,
    *,
    model: str,
    voice: str | None = None,
    response_format: str = "mp3",
    speed: float | None = None,
    chunk_chars: int = 20_000,
    timeout: float | None = None,
    on_event: EventCallback | None = None,
) -> ChunkedResult:
    """Synthesize ``text``, splitting it into queued chunks when it is too long.

    Raises ``ProviderError`` (with the failing chunk number) if any chunk fails —
    partial audio is never returned. Raises ``AudioMergeError`` if the parts cannot
    be merged.
    """

    def emit(stage: str, message: str, progress: int, **extra) -> None:
        if on_event is not None:
            on_event(stage, message, progress, extra)

    pieces = split_text(text, chunk_chars) if len(text) > chunk_chars else [text]
    total = len(pieces)

    if total == 1:
        result = provider.synthesize(
            pieces[0],
            model=model,
            voice=voice,
            response_format=response_format,
            speed=speed,
            timeout=timeout,
        )
        return ChunkedResult(
            audio=result.audio,
            chunks=[
                ChunkInfo(
                    index=1,
                    total=1,
                    characters=len(pieces[0]),
                    generation_id=result.generation_id,
                )
            ],
            characters=len(text),
        )

    emit(
        "chunk_plan",
        f"Text is {len(text):,} characters — split into {total} chunks "
        f"(~{max(len(p) for p in pieces):,} chars each)",
        _PLAN_PROGRESS,
        chunks=total,
    )

    infos: list[ChunkInfo] = []
    parts: list[bytes] = []
    for index, piece in enumerate(pieces, start=1):
        emit(
            "chunk_start",
            f"Chunk {index}/{total} — sending {len(piece):,} characters…",
            _chunk_progress(index, total, done=False),
            chunk=index,
            chunks=total,
        )
        started = time.monotonic()
        try:
            result = provider.synthesize(
                piece,
                model=model,
                voice=voice,
                response_format=response_format,
                speed=speed,
                timeout=timeout,
            )
        except ProviderError as exc:
            emit(
                "chunk_failed",
                f"Chunk {index}/{total} failed: {exc}",
                _chunk_progress(index, total, done=True),
                chunk=index,
                chunks=total,
            )
            raise ProviderError(f"Chunk {index}/{total} failed: {exc}") from exc

        elapsed = time.monotonic() - started
        infos.append(
            ChunkInfo(
                index=index,
                total=total,
                characters=len(piece),
                generation_id=result.generation_id,
                seconds=elapsed,
            )
        )
        parts.append(result.audio)
        emit(
            "chunk_done",
            f"Chunk {index}/{total} done — {len(result.audio):,} bytes in {elapsed:.1f}s",
            _chunk_progress(index, total, done=True),
            chunk=index,
            chunks=total,
        )

    emit("merge", f"Merging {total} audio parts…", _MERGE_PROGRESS, chunks=total)
    try:
        audio = merge_audio(parts, response_format)
    except AudioMergeError as exc:
        raise ProviderError(f"Could not merge audio chunks: {exc}") from exc

    return ChunkedResult(audio=audio, chunks=infos, characters=len(text))


def _chunk_progress(index: int, total: int, *, done: bool) -> int:
    """Map chunk progress into the chunk band: 25% before chunk 1 … 85% after chunk n."""
    span = _CHUNK_END_PROGRESS - _CHUNK_START_PROGRESS
    fraction = index / total if done else (index - 1) / total
    return int(_CHUNK_START_PROGRESS + span * fraction)
