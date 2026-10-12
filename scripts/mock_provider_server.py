"""A fake OpenAI-compatible TTS provider for local testing / demos.

Run it next to text2speech to exercise the full pipeline (UI -> API -> provider)
without an API key or network access::

    uv run python -m scripts.mock_provider_server

Then point a provider in config/providers.json at http://127.0.0.1:9111/api/v1
with ``api_key_env`` set to any non-empty value.

It responds to POST /api/v1/audio/speech with a short generated WAV beep and
reports the received payload at GET /api/v1/_last_request.
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import struct
import uuid
import wave

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, Response

app = FastAPI(title="mock TTS provider")

LAST_REQUEST: dict = {}
LAST_INTERACTION: dict = {}

SAMPLE_RATE = 8000
DURATION_SECONDS = 1.0
FREQUENCY_HZ = 440.0


def _beep_wav(seed_text: str) -> bytes:
    """A short sine beep whose pitch varies with the input, as a real WAV file."""
    freq = FREQUENCY_HZ + (sum(ord(c) for c in seed_text) % 220)
    frames = int(SAMPLE_RATE * DURATION_SECONDS)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        for i in range(frames):
            value = int(12000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE))
            wav.writeframes(struct.pack("<h", value))
    return buffer.getvalue()


@app.post("/api/v1/audio/speech")
async def speech(
    request: Request,
    authorization: str | None = Header(default=None),
) -> Response:
    if not authorization or not authorization.startswith("Bearer "):
        return JSONResponse(
            status_code=401, content={"error": {"message": "Missing bearer token"}}
        )
    payload = await request.json()
    LAST_REQUEST.clear()
    LAST_REQUEST.update(payload)

    fmt = str(payload.get("response_format", "wav")).lower()
    if fmt != "wav":
        return JSONResponse(
            status_code=400,
            content={"error": {"message": f"mock provider only supports 'wav', got '{fmt}'"}},
        )
    # Optional artificial delay ("delay": seconds in the payload, e.g. via the
    # provider's extra_body) to exercise progress reporting.
    delay = payload.get("delay")
    if delay:
        await asyncio.sleep(min(float(delay), 30.0))
    audio = _beep_wav(str(payload.get("input", "")))
    return Response(
        content=audio,
        media_type="audio/wav",
        headers={"x-generation-id": f"mock-{uuid.uuid4().hex[:12]}"},
    )


@app.get("/api/v1/_last_request")
def last_request() -> dict:
    return LAST_REQUEST


@app.post("/v1beta/interactions")
async def interactions(
    request: Request,
    x_goog_api_key: str | None = Header(default=None, alias="x-goog-api-key"),
) -> Response:
    """Google Interactions API shape: x-goog-api-key auth, steps[] response."""
    if not x_goog_api_key:
        return JSONResponse(
            status_code=401,
            content={"error": {"code": 401, "message": "API key not valid", "status": "UNAUTHENTICATED"}},
        )
    payload = await request.json()
    LAST_INTERACTION.clear()
    LAST_INTERACTION.update(payload)

    fmt = (payload.get("response_format") or {}).get("mime_type", "audio/wav")
    if fmt not in ("audio/wav", "audio/l16"):
        return JSONResponse(
            status_code=400,
            content={"error": {"code": 400, "message": f"unsupported mime_type {fmt}"}},
        )

    delay = payload.get("delay")
    if delay:
        await asyncio.sleep(min(float(delay), 30.0))

    text = ""
    for turn in payload.get("input") or []:
        for part in turn.get("content") or []:
            if part.get("type") == "text":
                text += part.get("text", "")
    audio = _beep_wav(text)
    body = {
        "id": f"mock-interaction-{uuid.uuid4().hex[:12]}",
        "steps": [
            # a non-model_output step that must be ignored by the client
            {"type": "thought", "content": [{"type": "text", "text": "planning..."}]},
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "transcript echo"},
                    {"type": "audio", "data": base64.b64encode(audio).decode("ascii")},
                ],
            },
        ],
    }
    return JSONResponse(content=body)


@app.get("/api/v1/_last_interaction")
def last_interaction() -> dict:
    return LAST_INTERACTION


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9111, log_level="info")


if __name__ == "__main__":
    main()
