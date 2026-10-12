"""End-to-end smoke test against a *running* text2speech instance.

Usage:
    1. uv run python -m scripts.mock_provider_server        # fake TTS upstream
    2. $env:T2S_PROVIDERS_FILE = "config\\providers.smoke.json"
       $env:MOCK_TTS_API_KEY = "smoke-key"
       $env:T2S_CHUNK_CHARS = "120"                          # exercise chunking
       uv run python -m uvicorn text2speech.main:app --port 8123
    3. uv run python -m scripts.smoke_test

Exits non-zero if any check fails.
"""

from __future__ import annotations

import io
import json
import sys
import uuid
import wave

import httpx

BASE_URL = "http://127.0.0.1:8123"
MOCK_URL = "http://127.0.0.1:9111"

FAILURES: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(name)


def main() -> int:
    # trust_env=False: the smoke targets are localhost, and some environments
    # carry IPv6 entries in NO_PROXY that break httpx's env-proxy parsing.
    with httpx.Client(base_url=BASE_URL, timeout=60.0, trust_env=False) as client:
        health = client.get("/api/health")
        check("GET /api/health", health.status_code == 200, health.text)
        check("health reports providers", health.json().get("providers", 0) >= 1)

        index = client.get("/")
        check("GET / serves the web UI", index.status_code == 200 and "provider" in index.text)

        providers = client.get("/api/providers").json()
        provider_id = providers[0]["id"] if providers else None
        check("GET /api/providers returns a provider", bool(provider_id), str(providers))

        # --- markdown upload: markdown syntax must be stripped, code skipped ---
        markdown = (
            "# Release Notes\n\n"
            "Version **1.0** ships with *speed* and [docs](https://example.com).\n\n"
            "```python\nprint('this code must not be spoken')\n```\n"
        )
        res = client.post(
            "/api/tts",
            data={"provider_id": provider_id, "response_format": "wav"},
            files={"file": ("notes.md", markdown.encode(), "text/markdown")},
        )
        check("POST /api/tts (markdown upload)", res.status_code == 200, res.text)
        job = res.json()
        check("job reports markdown kind", job.get("kind") == "markdown", str(job))
        check("generation id captured", bool(job.get("generation_id")), str(job))

        upstream = httpx.get(
            f"{MOCK_URL}/api/v1/_last_request", timeout=10, trust_env=False
        ).json()
        sent_text = upstream.get("input", "")
        check("markdown syntax stripped", "**" not in sent_text and "#" not in sent_text, sent_text)
        check("code block excluded", "must not be spoken" not in sent_text, sent_text)
        check("prose preserved", "Version 1.0 ships with speed and docs." in sent_text, sent_text)
        check("model forwarded", upstream.get("model") == job.get("model"), str(upstream))

        audio = client.get(job["url"])
        check("GET audio file", audio.status_code == 200 and len(audio.content) > 100)
        check("audio content-type", audio.headers.get("content-type") == "audio/wav")

        # --- markdown with code blocks included ---
        res = client.post(
            "/api/tts",
            data={
                "provider_id": provider_id,
                "response_format": "wav",
                "include_code_blocks": "true",
            },
            files={"file": ("a.md", b"```\nspoken code\n```", "text/markdown")},
        )
        check("include_code_blocks=true", res.status_code == 200, res.text)
        upstream = httpx.get(
            f"{MOCK_URL}/api/v1/_last_request", timeout=10, trust_env=False
        ).json()
        check("code block included", "spoken code" in upstream.get("input", ""), str(upstream))

        # --- plain pasted text ---
        res = client.post(
            "/api/tts",
            data={"provider_id": provider_id, "text": "Just a quick pasted sentence.", "response_format": "wav"},
        )
        check("POST /api/tts (pasted text)", res.status_code == 200, res.text)

        # --- txt upload ---
        res = client.post(
            "/api/tts",
            data={"provider_id": provider_id, "response_format": "wav"},
            files={"file": ("notes.txt", b"plain text file contents", "text/plain")},
        )
        check("POST /api/tts (txt upload)", res.status_code == 200, res.text)

        # --- chunked synthesis: long text is split, queued and merged ---
        long_text = " ".join(f"This is smoke test sentence number {i}." for i in range(30))
        res = client.post(
            "/api/tts",
            data={"provider_id": provider_id, "text": long_text, "response_format": "wav"},
        )
        check("POST /api/tts (chunked)", res.status_code == 200, res.text)
        if res.status_code == 200:
            job = res.json()
            check(
                "long text was chunked",
                job.get("chunked") is True and job["chunks"] > 3,
                f"chunks={job.get('chunks')}",
            )
            check(
                "generation ids per chunk",
                len(job.get("generation_ids") or []) == job["chunks"],
                str(job.get("generation_ids")),
            )
            audio = client.get(job["url"])
            check(
                "merged audio is a valid WAV",
                audio.status_code == 200 and audio.content[:4] == b"RIFF",
                f"{audio.status_code} {audio.content[:8]!r}",
            )
            # Every mock part is the same 1-second beep: merged frames == parts × 1s.
            with wave.open(io.BytesIO(audio.content)) as merged:
                check(
                    "merged duration equals sum of parts",
                    merged.getnframes() == 8000 * job["chunks"],
                    f"{merged.getnframes()} frames vs {8000 * job['chunks']}",
                )

        with client.stream(
            "POST",
            "/api/tts/stream",
            data={"provider_id": provider_id, "text": long_text, "response_format": "wav"},
        ) as chunk_stream:
            chunk_sse = "".join(chunk for chunk in chunk_stream.iter_text())
        chunk_events = [
            json.loads(line[len("data:"):].strip())
            for block in chunk_sse.strip().split("\n\n")
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        stages = [e["stage"] for e in chunk_events]
        check("stream reports a chunk plan", "chunk_plan" in stages, str(stages))
        check(
            "stream reports every chunk start/done",
            stages.count("chunk_start") == stages.count("chunk_done") > 1,
            str(stages),
        )
        check("stream merges at the end", "merge" in stages, str(stages))
        check(
            "stream done job is chunked",
            chunk_events[-1]["stage"] == "done"
            and chunk_events[-1]["job"]["chunked"] is True,
            str(chunk_events[-1].get("job", {}).get("chunks")),
        )
        progresses = [e.get("progress", 0) for e in chunk_events]
        check(
            "chunk progress is monotonic",
            all(b >= a for a, b in zip(progresses, progresses[1:])),
            str(progresses),
        )

        # --- Google (Interactions API) provider ---
        google_id = next((p["id"] for p in providers if p["id"] == "google-mock"), None)
        if google_id:
            res = client.post(
                "/api/tts",
                data={
                    "provider_id": google_id,
                    "text": "Google interactions API check.",
                    "response_format": "wav",
                    "voice": "Puck",
                },
            )
            check("POST /api/tts (google provider)", res.status_code == 200, res.text)
            if res.status_code == 200:
                job = res.json()
                check("google job has generation id", bool(job.get("generation_id")), str(job))
                audio = client.get(job["url"])
                check(
                    "google audio is a valid WAV",
                    audio.status_code == 200 and audio.content[:4] == b"RIFF",
                    f"{audio.status_code} {audio.content[:8]!r}",
                )
            interaction = httpx.get(
                f"{MOCK_URL}/api/v1/_last_interaction", timeout=10, trust_env=False
            ).json()
            check(
                "google payload has interactions shape",
                interaction.get("model") == "gemini-3.8-flash-tts"
                and (interaction.get("input") or [{}])[0].get("type") == "user_input"
                and ((interaction.get("input") or [{}])[0].get("content") or [{}])[0].get("type")
                == "text",
                str(interaction),
            )
            check(
                "google voice is forwarded",
                ((interaction.get("generation_config") or {}).get("speech_config") or [{}])[0].get(
                    "voice"
                )
                == "Puck",
                str(interaction.get("generation_config")),
            )
            check(
                "google style annotation is attached",
                ((interaction.get("input") or [{}])[0].get("content") or [{}])[0]
                .get("annotations", [{}])[0]
                .get("style")
                == "cheerful and friendly",
                str(interaction),
            )
            res = client.post(
                "/api/tts",
                data={
                    "provider_id": google_id,
                    "text": "mp3 is not supported by google",
                    "response_format": "mp3",
                },
            )
            check("google rejects mp3 -> 400", res.status_code == 400, res.text)
            check(
                "google format error is explanatory",
                "does not support" in res.text,
                res.text,
            )

        # --- error paths ---
        res = client.post("/api/tts", data={"provider_id": provider_id})
        check("no input -> 400", res.status_code == 400, res.text)
        res = client.post("/api/tts", data={"provider_id": "does-not-exist", "text": "hi"})
        check("unknown provider -> 400", res.status_code == 400, res.text)
        res = client.post(
            "/api/tts",
            data={"provider_id": provider_id},
            files={"file": ("bad.pdf", b"%PDF", "application/pdf")},
        )
        check("bad extension -> 400", res.status_code == 400, res.text)

        # --- streaming progress (SSE) ---
        with client.stream(
            "POST",
            "/api/tts/stream",
            data={
                "provider_id": provider_id,
                "text": "streaming progress check",
                "response_format": "wav",
            },
        ) as stream:
            check("stream endpoint is SSE", stream.headers["content-type"].startswith("text/event-stream"))
            sse_text = "".join(chunk for chunk in stream.iter_text())
        events = [
            json.loads(line[len("data:"):].strip())
            for block in sse_text.strip().split("\n\n")
            for line in block.split("\n")
            if line.startswith("data:")
        ]
        stages = [e["stage"] for e in events]
        check("stream starts with extract", stages[0] == "extract", str(stages))
        check("stream submits to provider", "submit" in stages, str(stages))
        check("stream finishes with done", stages[-1] == "done", str(stages))
        check(
            "stream progress is monotonic-ish",
            all(0 <= e.get("progress", 0) <= 100 for e in events),
            str(stages),
        )
        stream_job = events[-1].get("job", {})
        check("stream job has audio url", bool(stream_job.get("url")), str(stream_job))

        # --- slow provider must emit "waiting" heartbeat events ---
        slow_id = next((p["id"] for p in providers if p["id"] == "mock-slow"), None)
        if slow_id:
            with client.stream(
                "POST",
                "/api/tts/stream",
                data={
                    "provider_id": slow_id,
                    "text": "slow generation with heartbeats",
                    "response_format": "wav",
                },
            ) as slow_stream:
                slow_sse = "".join(chunk for chunk in slow_stream.iter_text())
            slow_events = [
                json.loads(line[len("data:"):].strip())
                for block in slow_sse.strip().split("\n\n")
                for line in block.split("\n")
                if line.startswith("data:")
            ]
            waiting = [e for e in slow_events if e["stage"] == "waiting"]
            check("slow provider emits waiting events", len(waiting) >= 2, str(len(waiting)))
            check(
                "waiting events count elapsed seconds",
                any("s)" in e.get("message", "") for e in waiting),
                str(waiting[:1]),
            )

        outputs = client.get("/api/outputs").json()
        check("GET /api/outputs lists files", len(outputs) >= 3, str(len(outputs)))

        # --- delete a recent output ---
        victim = stream_job.get("filename") or outputs[0]["filename"]
        deleted = client.delete(f"/api/outputs/{victim}")
        check("DELETE /api/outputs/{file}", deleted.status_code == 200, deleted.text)
        check(
            "deleted file is gone",
            client.get(f"/api/audio/{victim}").status_code == 404,
        )
        check(
            "deleting again -> 404",
            client.delete(f"/api/outputs/{victim}").status_code == 404,
        )
        check(
            "path traversal rejected",
            client.delete("/api/outputs/..%5Cpyproject.toml").status_code == 400,
        )

    print()
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("All smoke checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
