"""FastAPI application: web UI + JSON API for text-to-speech conversion."""

from __future__ import annotations

import asyncio
import json
import re
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, AsyncIterator, Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .audio import AudioMergeError
from .config import get_settings
from .providers import get_provider, get_providers, reset_cache
from .providers.base import ProviderError
from .schemas import HealthResponse, OutputFile, ProviderInfo, TTSJob
from .synthesis import synthesize_chunked
from .textio import ExtractedText, TextInputError, extract_text

STATIC_DIR = Path(__file__).resolve().parent / "static"
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
AUDIO_MEDIA_TYPES = {
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "opus": "audio/opus",
    "aac": "audio/aac",
    "flac": "audio/flac",
    "pcm": "audio/pcm",
}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

# ---------------------------------------------------------------------------
# Windows asyncio log-noise filter
# ---------------------------------------------------------------------------

# asyncio's proactor transport tears a socket down in a deferred callback. When the
# browser has already dropped the connection (it aborts an <audio> range request,
# closes the tab, or recycles a keep-alive socket) the socket arrives reset, so
# ``socket.shutdown()`` inside that callback raises ConnectionResetError
# [WinError 10054] *after* the response was fully served. CPython does not guard
# that call, so asyncio's default handler prints a full traceback for a connection
# that was closed on purpose by the client (see CPython issue #149388).
_TEARDOWN_CALLBACK_NAMES = ("_ProactorBasePipeTransport._call_connection_lost",)


def is_dead_connection_noise(context: dict) -> bool:
    """True for asyncio's harmless "connection forcibly closed" teardown errors.

    Only errors raised *by that specific teardown callback* are filtered: the
    protocol has already been told the connection is lost by then, so the socket
    state carries no actionable information. Every other callback error still
    reaches the default handler.
    """
    message = context.get("message") or ""
    if not any(
        f"Exception in callback {name}" in message for name in _TEARDOWN_CALLBACK_NAMES
    ):
        return False
    return isinstance(context.get("exception"), OSError)


def install_dead_connection_filter(
    loop: asyncio.AbstractEventLoop,
) -> Callable | None:
    """Silence teardown noise on ``loop``; returns the previous handler."""
    previous = loop.get_exception_handler()

    def handler(active_loop: asyncio.AbstractEventLoop, context: dict) -> None:
        if is_dead_connection_noise(context):
            return
        if previous is not None:
            previous(active_loop, context)
        else:
            active_loop.default_exception_handler(context)

    loop.set_exception_handler(handler)
    return previous


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings = get_settings()
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    loop = asyncio.get_running_loop()
    previous_handler = install_dead_connection_filter(loop)
    try:
        yield
    finally:
        loop.set_exception_handler(previous_handler)


app = FastAPI(
    title="text2speech",
    description="Convert txt/markdown text into speech using configurable TTS providers.",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    settings = get_settings()
    return HealthResponse(
        status="ok",
        providers=len(get_providers()),
        version=__version__,
        chunk_chars=settings.chunk_chars,
        max_text_chars=settings.max_text_chars,
    )


@app.get("/api/providers", response_model=list[ProviderInfo])
def list_providers() -> list[ProviderInfo]:
    infos = []
    for provider in get_providers():
        config = provider.config
        supported = getattr(provider, "supported_formats", None)
        infos.append(
            ProviderInfo(
                id=config.id,
                name=config.name,
                type=config.type,
                base_url=config.base_url,
                models=list(config.models),
                default_model=config.default_model or (config.models[0] if config.models else None),
                voices=list(config.voices),
                default_voice=config.default_voice,
                has_api_key=bool(config.api_key),
                api_key_env=config.api_key_env,
                supported_formats=sorted(supported) if supported else None,
            )
        )
    return infos


@app.post("/api/providers/reload")
def reload_providers() -> dict[str, int]:
    """Re-read the providers config (e.g. after editing it)."""
    reset_cache()
    return {"providers": len(get_providers())}


# ---------------------------------------------------------------------------
# Shared request handling
# ---------------------------------------------------------------------------


def _pick_model(provider, requested: str | None) -> str:
    config = provider.config
    if requested and requested.strip():
        return requested.strip()
    if config.default_model:
        return config.default_model
    if config.models:
        return config.models[0]
    raise HTTPException(
        status_code=400,
        detail=f"Provider '{config.id}' has no model configured. Set 'model' in the request "
        f"or 'default_model' in the providers config.",
    )


def _pick_format(requested: str | None, provider=None) -> str:
    settings = get_settings()
    fmt = (requested or settings.default_response_format).lower().lstrip(".")
    if fmt not in AUDIO_MEDIA_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported response_format '{requested}'. "
            f"Use one of: {', '.join(sorted(AUDIO_MEDIA_TYPES))}.",
        )
    supported = getattr(provider, "supported_formats", None)
    if supported and fmt not in supported:
        name = getattr(getattr(provider, "config", None), "id", "this provider")
        raise HTTPException(
            status_code=400,
            detail=f"Provider '{name}' does not support '{fmt}' audio. "
            f"Supported formats: {', '.join(sorted(supported))}.",
        )
    return fmt


async def prepare_request(
    *,
    provider_id: str,
    text: str | None,
    file: UploadFile | None,
    model: str | None,
    voice: str | None,
    response_format: str | None,
    speed: float | None,
    include_code_blocks: bool,
    kind: str | None,
) -> tuple[object, ExtractedText, str, str, str | None, float | None]:
    """Validate the request and extract text. Raises HTTPException on bad input."""
    settings = get_settings()

    try:
        provider = get_provider(provider_id)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    file_bytes = await file.read() if file is not None else None
    if file_bytes is not None and len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Uploaded file is larger than 10 MB.")

    try:
        extracted = extract_text(
            raw_text=text,
            file_name=file.filename if file is not None else None,
            file_bytes=file_bytes,
            declared_kind=kind,
            include_code_blocks=include_code_blocks,
            max_chars=settings.max_text_chars,
        )
    except TextInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chosen_model = _pick_model(provider, model)
    fmt = _pick_format(response_format, provider)
    return provider, extracted, chosen_model, fmt, (voice or None), speed


def save_audio(audio: bytes, fmt: str) -> str:
    """Write audio bytes to the outputs directory and return the filename."""
    settings = get_settings()
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:8]}.{fmt}"
    (settings.outputs_dir / filename).write_bytes(audio)
    return filename


def build_job(
    *,
    provider,
    extracted: ExtractedText,
    result,
    filename: str,
    chosen_model: str,
    fmt: str,
) -> TTSJob:
    return TTSJob(
        filename=filename,
        url=f"/api/audio/{filename}",
        provider_id=provider.config.id,
        provider_name=provider.config.name,
        model=chosen_model,
        format=fmt,
        characters=extracted.characters,
        kind=extracted.kind,
        source_name=extracted.source_name,
        generation_id=result.generation_id,
        text_preview=extracted.text[:200],
        chunked=result.chunked,
        chunks=len(result.chunks),
        generation_ids=result.generation_ids,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/tts", response_model=TTSJob)
async def synthesize(
    provider_id: Annotated[str, Form()],
    text: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    model: Annotated[str | None, Form()] = None,
    voice: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "mp3",
    speed: Annotated[float | None, Form()] = None,
    include_code_blocks: Annotated[bool, Form()] = False,
    kind: Annotated[str | None, Form()] = None,
) -> TTSJob:
    """Convert text (pasted or uploaded .txt/.md) into an audio file (single JSON reply)."""
    provider, extracted, chosen_model, fmt, voice, speed = await prepare_request(
        provider_id=provider_id,
        text=text,
        file=file,
        model=model,
        voice=voice,
        response_format=response_format,
        speed=speed,
        include_code_blocks=include_code_blocks,
        kind=kind,
    )
    settings = get_settings()

    try:
        result = await run_in_threadpool(
            synthesize_chunked,
            provider,
            extracted.text,
            model=chosen_model,
            voice=voice,
            response_format=fmt,
            speed=speed,
            chunk_chars=settings.chunk_chars,
            timeout=settings.request_timeout_seconds,
        )
    except ProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc

    filename = save_audio(result.audio, fmt)
    return build_job(
        provider=provider,
        extracted=extracted,
        result=result,
        filename=filename,
        chosen_model=chosen_model,
        fmt=fmt,
    )


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/tts/stream")
async def synthesize_stream(
    provider_id: Annotated[str, Form()],
    text: Annotated[str | None, Form()] = None,
    file: Annotated[UploadFile | None, File()] = None,
    model: Annotated[str | None, Form()] = None,
    voice: Annotated[str | None, Form()] = None,
    response_format: Annotated[str, Form()] = "mp3",
    speed: Annotated[float | None, Form()] = None,
    include_code_blocks: Annotated[bool, Form()] = False,
    kind: Annotated[str | None, Form()] = None,
) -> StreamingResponse:
    """Same as ``/api/tts`` but streams Server-Sent Events with live progress.

    Events are JSON objects with ``stage`` (extract/submit/chunk_plan/chunk_start/
    waiting/chunk_done/merge/save/done/error), ``message``, ``progress`` (0-100) and,
    on success, ``job``.
    """
    settings = get_settings()

    async def events() -> AsyncIterator[str]:
        try:
            provider, extracted, chosen_model, fmt, chosen_voice, chosen_speed = (
                await prepare_request(
                    provider_id=provider_id,
                    text=text,
                    file=file,
                    model=model,
                    voice=voice,
                    response_format=response_format,
                    speed=speed,
                    include_code_blocks=include_code_blocks,
                    kind=kind,
                )
            )
        except HTTPException as exc:
            yield _sse({"stage": "error", "message": str(exc.detail), "progress": 100})
            return

        yield _sse(
            {
                "stage": "extract",
                "message": f"Read {extracted.source_name} — "
                f"{extracted.characters:,} characters to speak",
                "progress": 15,
            }
        )
        yield _sse(
            {
                "stage": "submit",
                "message": f"Sending to {provider.config.name} ({chosen_model})…",
                "progress": 20,
            }
        )

        # Bridge progress events emitted by the worker thread into this generator.
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[tuple[str, str, int, dict]] = asyncio.Queue()

        def on_event(stage: str, message: str, progress: int, extra: dict) -> None:
            try:
                loop.call_soon_threadsafe(queue.put_nowait, (stage, message, progress, extra))
            except RuntimeError:  # pragma: no cover - client disconnected
                pass

        task = asyncio.ensure_future(
            run_in_threadpool(
                synthesize_chunked,
                provider,
                extracted.text,
                model=chosen_model,
                voice=chosen_voice,
                response_format=fmt,
                speed=chosen_speed,
                chunk_chars=settings.chunk_chars,
                timeout=settings.request_timeout_seconds,
                on_event=on_event,
            )
        )

        last_progress = 20
        current_chunk: tuple[int, int] | None = None
        waited = 0.0
        while True:
            get_task = asyncio.ensure_future(queue.get())
            done, _ = await asyncio.wait(
                {task, get_task}, timeout=1.0, return_when=asyncio.FIRST_COMPLETED
            )
            if get_task in done:
                stage, message, progress, extra = get_task.result()
                last_progress = progress
                waited = 0.0
                if stage == "chunk_start":
                    current_chunk = (extra.get("chunk", 1), extra.get("chunks", 1))
                yield _sse({"stage": stage, "message": message, "progress": progress, **extra})
                continue
            get_task.cancel()
            if task in done:
                break
            waited += 1.0
            if current_chunk:
                message = (
                    f"Chunk {current_chunk[0]}/{current_chunk[1]} — "
                    f"{provider.config.name} is generating audio… ({int(waited)}s)"
                )
            else:
                message = f"{provider.config.name} is generating audio… ({int(waited)}s)"
            yield _sse({"stage": "waiting", "message": message, "progress": last_progress})

        # Drain any events queued just before the worker finished (e.g. "merge").
        while not queue.empty():
            stage, message, progress, extra = queue.get_nowait()
            yield _sse({"stage": stage, "message": message, "progress": progress, **extra})

        try:
            result = task.result()
        except ProviderError as exc:
            yield _sse({"stage": "error", "message": str(exc), "progress": 100})
            return
        except Exception as exc:  # noqa: BLE001 - surface anything to the client
            yield _sse({"stage": "error", "message": f"Internal error: {exc}", "progress": 100})
            return

        yield _sse(
            {
                "stage": "save",
                "message": f"Received {len(result.audio):,} bytes — saving…",
                "progress": 90,
            }
        )
        filename = save_audio(result.audio, fmt)
        job = build_job(
            provider=provider,
            extracted=extracted,
            result=result,
            filename=filename,
            chosen_model=chosen_model,
            fmt=fmt,
        )
        yield _sse(
            {
                "stage": "done",
                "message": "Done",
                "progress": 100,
                "job": job.model_dump(),
            }
        )

    return StreamingResponse(events(), media_type="text/event-stream")


@app.get("/api/audio/{filename}")
def get_audio(filename: str) -> FileResponse:
    if not SAFE_NAME_RE.match(filename) or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    settings = get_settings()
    path = (settings.outputs_dir / filename).resolve()
    if not path.is_file() or path.parent != settings.outputs_dir.resolve():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    media_type = AUDIO_MEDIA_TYPES.get(
        path.suffix.lstrip(".").lower(), "application/octet-stream"
    )
    return FileResponse(path, media_type=media_type, filename=filename)


@app.delete("/api/outputs/{filename}")
def delete_output(filename: str) -> dict[str, str]:
    """Delete a generated audio file."""
    if not SAFE_NAME_RE.match(filename) or filename.startswith("."):
        raise HTTPException(status_code=400, detail="Invalid filename.")
    settings = get_settings()
    path = (settings.outputs_dir / filename).resolve()
    if not path.is_file() or path.parent != settings.outputs_dir.resolve():
        raise HTTPException(status_code=404, detail="Audio file not found.")
    path.unlink()
    return {"deleted": filename}


@app.get("/api/outputs", response_model=list[OutputFile])
def list_outputs() -> list[OutputFile]:
    settings = get_settings()
    files = []
    if settings.outputs_dir.exists():
        for path in sorted(
            settings.outputs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            if path.is_file():
                stat = path.stat()
                files.append(
                    OutputFile(
                        filename=path.name,
                        url=f"/api/audio/{path.name}",
                        size_bytes=stat.st_size,
                        modified_at=stat.st_mtime,
                    )
                )
    return files[:100]


@app.exception_handler(Exception)
async def unhandled_error(request: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(status_code=500, content={"detail": f"Internal error: {exc}"})


def main() -> None:
    """Console-script entry point: ``text2speech``."""
    import uvicorn

    settings = get_settings()
    uvicorn.run("text2speech.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()
