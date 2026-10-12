# text2speech

<p align="center">
  <strong>English</strong>
  &nbsp;·&nbsp;
  <a href="./README_CN.md">简体中文</a>
</p>

A web service that converts **text (pasted, `.txt`, or `.md`)** into speech by calling
configurable text-to-speech model APIs. Python + [uv](https://docs.astral.sh/uv/),
FastAPI backend, single-page web UI — no build step.
中文文档：[README_CN.md](README_CN.md)。

```
browser UI ──► FastAPI (/api/tts) ──► provider (OpenAI-compatible /audio/speech) ──► audio file ──► player + download
                    │
                    └─ txt/markdown extraction & normalization (code blocks skipped by default)
```

## Quick start

```bash
# 1. Install dependencies into a local virtual environment (uv manages it)
uv sync

# 2. Configure your API key(s)
cp .env.example ~/.text2speech/.env
#    edit ~/.text2speech/.env and set OPENROUTER_API_KEY (https://openrouter.ai/keys)
#    and/or GEMINI_API_KEY (https://aistudio.google.com/apikey)

# 3. Run the service
uv run text2speech          # or: uv run python -m uvicorn text2speech.main:app
```

All user-level configuration lives in **`~/.text2speech/`**: `config.json` (provider
definitions) and `.env` (API keys and `T2S_*` settings). On first run the bundled
template `config/providers.json` is copied to `~/.text2speech/config.json`.

Open <http://127.0.0.1:8000>, paste text or drop a `.txt`/`.md` file, pick a provider,
and press **Generate speech**. The audio is saved under `outputs/` and playable /
downloadable in the browser.

### Try it without an API key

A mock TTS provider is included so you can exercise the whole pipeline offline:

```bash
# terminal 1 — fake upstream that returns a generated WAV beep
uv run python -m scripts.mock_provider_server

# terminal 2 — the app, pointed at the mock
$env:T2S_PROVIDERS_FILE = "config/providers.smoke.json"   # PowerShell
$env:MOCK_TTS_API_KEY = "smoke-key"
uv run python -m uvicorn text2speech.main:app --port 8123

# terminal 3 — end-to-end checks (50 assertions)
uv run python -m scripts.smoke_test
```

## How it works

### Input handling

| Input | Behavior |
| --- | --- |
| Pasted text | Used as-is (only whitespace normalized) |
| `.txt` upload | Read as text (UTF-8/BOM/GBK tolerant decoding) |
| `.md` / `.markdown` upload | Converted to speakable plain text: headings, emphasis, links, images, lists, tables, blockquotes and HTML comments are flattened |

Markdown **fenced code blocks are dropped by default** — code read aloud is noise.
Tick *"Read code blocks aloud"* (or send `include_code_blocks=true`) to keep them.
Files larger than 10 MB, or longer than `T2S_MAX_TEXT_CHARS` (default 200 000 chars),
are rejected with a clear error; texts above `T2S_CHUNK_CHARS` are chunked instead of
rejected (see below).

### Live progress

The UI streams generation progress over Server-Sent Events (`POST /api/tts/stream`),
so you see each stage instead of a spinner: text extraction → request submitted →
per-second *waiting* heartbeats while the provider generates → audio received →
saved. A progress bar in the UI reflects the same events. `POST /api/tts` remains
available for plain JSON clients (curl, scripts).

### Chunked synthesis (long texts)

When the text to speak exceeds **`T2S_CHUNK_CHARS`** (default 20 000), it is
automatically:

1. **split** on paragraph → sentence boundaries (Chinese and English punctuation;
   a single oversized sentence is hard-cut), so chunks never cut mid-sentence;
2. **queued and synthesized one chunk at a time** (sequential, rate-limit friendly),
   each with its own progress events and per-chunk timeout;
3. **merged** back into a single audio file that is saved, played and downloaded
   as one piece.

The UI shows the estimated chunk count next to the character counter, a per-chunk
status list (⏳ / ✓ / ✗) while running, and an “N chunks merged” badge on the result.
`job.chunks` / `job.chunked` / `job.generation_ids` expose the same information via
the API. If any chunk fails the whole job stops with `Chunk k/n failed: …` — partial
audio is never returned.

Merging is **sample-accurate for WAV and raw PCM**; container formats (MP3, Opus,
AAC, FLAC) are byte-concatenated, which players render as one continuous stream.
`T2S_MAX_TEXT_CHARS` (default 200 000) remains the hard cap on the *total* input.

### Providers

Providers are declared in **`~/.text2speech/config.json`** (user-level, shared by all
checkouts; override with `T2S_PROVIDERS_FILE`). On first run the bundled template
[`config/providers.json`](config/providers.json) is copied there automatically, so
there is always something to edit. Two provider types ship built in:

| `type` | API shape | Notes |
| --- | --- | --- |
| `openai-compatible` | `POST {base_url}/audio/speech`, `Authorization: Bearer` | OpenRouter (default) and most TTS gateways |
| `google` | Gemini **Interactions API** `POST {base_url}/v1beta/interactions`, `x-goog-api-key` header | `gemini-3.8-flash-tts` & friends; alias: `gemini` |

Add as many entries as you like; they appear in the UI dropdown.

```json
{
  "providers": [
    {
      "id": "openrouter",
      "name": "OpenRouter",
      "type": "openai-compatible",
      "base_url": "https://openrouter.ai/api/v1",
      "api_key_env": "OPENROUTER_API_KEY",
      "models": ["fish-audio/s2.1-pro-free:free"],
      "default_model": "fish-audio/s2.1-pro-free:free",
      "extra_headers": {
        "HTTP-Referer": "{env:OPENROUTER_SITE_URL}",
        "X-OpenRouter-Title": "{env:OPENROUTER_SITE_NAME}"
      }
    },
    {
      "id": "google",
      "name": "Google Gemini",
      "type": "google",
      "base_url": "https://generativelanguage.googleapis.com",
      "api_key_env": "GEMINI_API_KEY",
      "models": ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts"],
      "default_model": "gemini-3.8-flash-tts",
      "default_voice": "Kore",
      "voices": ["Kore", "Puck", "Charon", "Zephyr", "..."],
      "style": ""
    }
  ]
}
```

Common fields:

* `api_key_env` — name of the environment variable holding the key (keys live in
  `.env`/environment only, never in the config file).
* `models` / `default_model` — model IDs offered in the UI.
* `voices` / `default_voice` — voices offered in the UI; `default_voice` is used when a
  request does not pick one.
* `style` — optional speech-style hint. For the Google provider it is sent as a
  `speech_metadata.style` annotation (turn-level delivery, *not* read aloud).
* `extra_headers` / `extra_body` — merged into every request; `{env:VAR}` placeholders
  are expanded from the environment.
* `enabled: false` — keep an entry around without showing it.
* After editing the file, click **reload** via `POST /api/providers/reload` or restart.

The reference calls the service makes:

```python
# openai-compatible (e.g. OpenRouter)
requests.post(
    url="https://openrouter.ai/api/v1/audio/speech",
    headers={"Authorization": "Bearer <OPENROUTER_API_KEY>", "Content-Type": "application/json"},
    json={"model": "fish-audio/s2.1-pro-free:free", "input": "Hello! This is a text-to-speech test."},
)
```

```sh
# google (Gemini Interactions API)
curl -X POST "https://generativelanguage.googleapis.com/v1beta/interactions" \
  -H "x-goog-api-key: $GEMINI_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemini-3.8-flash-tts",
    "input": [{"type": "user_input", "content": [{
      "type": "text", "text": "Have a wonderful day!",
      "annotations": [{"type": "speech_metadata", "style": "cheerful and friendly"}]}]}],
    "response_format": {"type": "audio"},
    "generation_config": {"speech_config": [{"voice": "Kore"}]}
  }'
# -> steps[].content[] where type=="audio", base64 .data (WAV on unary requests)
```

**Format support differs by provider**: OpenAI-compatible endpoints accept the usual
`mp3`/`wav`/`opus`/`aac`/`flac`/`pcm`, while Google TTS produces `wav` (default) and
`pcm` (raw `audio/l16`). The UI disables unsupported formats and the API returns a
clear 400 listing what the selected provider supports.

Custom backends can be plugged in by subclassing the provider protocol and calling
`register_provider_type("my-type", MyProvider)` — `type` in the JSON then selects it.

### HTTP API

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | Liveness + provider count |
| `GET` | `/api/providers` | Configured providers (models, voices, key status) |
| `POST` | `/api/providers/reload` | Re-read the providers config |
| `POST` | `/api/tts` | Synthesize (multipart form) → JSON job |
| `POST` | `/api/tts/stream` | Same, but streams SSE progress events |
| `GET` | `/api/audio/{filename}` | Download a generated file |
| `GET` | `/api/outputs` | Recent outputs |
| `DELETE` | `/api/outputs/{filename}` | Delete a generated file |

`POST /api/tts` form fields: `provider_id` (required), `text` **or** `file`
(`.txt`/`.md`/`.markdown`), `model`, `voice`, `response_format`
(`mp3`/`wav`/`opus`/`aac`/`flac`/`pcm`), `speed`, `include_code_blocks`.
Returns JSON with `url`, `generation_id`, `characters`, `kind`, ….

```bash
curl -F "provider_id=openrouter" -F "text=Hello world" \
     -F "model=fish-audio/s2.1-pro-free:free" \
     http://127.0.0.1:8000/api/tts
```

SSE events look like:

```
data: {"stage": "extract",     "message": "Read notes.md — 1,234 characters to speak", "progress": 15}
data: {"stage": "submit",      "message": "Sending to OpenRouter (fish-audio/s2.1-pro-free:free)…", "progress": 20}
data: {"stage": "chunk_plan",  "message": "Text is 52,340 characters — split into 3 chunks (~20,000 chars each)", "progress": 25}
data: {"stage": "chunk_start", "message": "Chunk 1/3 — sending 20,112 characters…", "progress": 30, "chunk": 1, "chunks": 3}
data: {"stage": "waiting",     "message": "Chunk 1/3 — OpenRouter is generating audio… (4s)", "progress": 30}
data: {"stage": "chunk_done",  "message": "Chunk 1/3 done — 312,004 bytes in 8.2s", "progress": 48, "chunk": 1, "chunks": 3}
data: {"stage": "merge",       "message": "Merging 3 audio parts…", "progress": 90}
data: {"stage": "save",        "message": "Received 936,012 bytes — saving…", "progress": 90}
data: {"stage": "done",        "message": "Done", "progress": 100, "job": { … }}
```

Errors arrive as `{"stage": "error", "message": "…"}` on the same stream (a failed
chunk reports `Chunk k/n failed: …`).

## Configuration

Settings come from environment variables (prefix `T2S_`, optionally via a `.env` file):

| Variable | Default | Meaning |
| --- | --- | --- |
| `T2S_HOST` / `T2S_PORT` | `127.0.0.1` / `8000` | Bind address |
| `T2S_OUTPUTS_DIR` | `./outputs` | Where audio files are written |
| `T2S_PROVIDERS_FILE` | `~/.text2speech/config.json` | Provider definitions |
| `T2S_ENV_FILE` | `~/.text2speech/.env` | API keys / app settings file |
| `T2S_CHUNK_CHARS` | `20000` | Split threshold: longer texts are chunked, queued and merged |
| `T2S_MAX_TEXT_CHARS` | `200000` | Hard cap on the total input per request |
| `T2S_REQUEST_TIMEOUT_SECONDS` | `180` | Upstream HTTP timeout (applies per chunk) |

`.env` lives at **`~/.text2speech/.env`**, next to `config.json`, and is loaded into the
process environment on startup — so provider keys (`OPENROUTER_API_KEY`,
`GEMINI_API_KEY`, …) and `{env:VAR}` header references resolve from it. Set
`T2S_ENV_FILE` to use a different location; a legacy `./​.env` in the project root is
still honored as a fallback when `~/.text2speech/.env` does not exist. **Restart the
server after editing it.** Real environment variables always take precedence over
`.env` values.

## Project layout

```
src/text2speech/
├── main.py                     # FastAPI app, endpoints, CLI entry point
├── config.py                   # settings + provider config loading (seeds ~/.text2speech)
├── schemas.py                  # API models
├── textio.py                   # txt/markdown → speakable text + sentence-aware splitting
├── audio.py                    # merging per-chunk audio parts
├── synthesis.py                # chunked orchestration (split → queue → merge)
├── providers/
│   ├── base.py                 # provider protocol, SynthesisResult, errors
│   ├── openai_compatible.py    # OpenAI-compatible /audio/speech client
│   ├── google.py               # Gemini Interactions API client
│   └── registry.py             # builds providers from config
└── static/index.html           # web UI (no build step)
config/providers.json           # template copied to ~/.text2speech/config.json on first run
scripts/mock_provider_server.py # fake upstream for offline testing (OpenAI + Google shapes)
scripts/smoke_test.py           # end-to-end checks against a running instance
tests/                          # pytest suite
```

## Development

```bash
uv sync --dev        # or: uv sync (dev group is included)
uv run pytest        # 144 unit/integration tests
uv run python -m scripts.smoke_test   # against a running instance (see above)
```

Tests cover markdown→text conversion (headings, links, tables, code blocks, encodings),
sentence-aware splitting, audio merging (WAV frame-accurate, byte concat), the chunked
orchestration (ordering, conservation, failure abort, event sequence), both provider
clients (payloads, auth headers, error paths, mocked transports), the config
location/seeding logic and the full API stack (uploads, validation errors, SSE progress
stream, chunked jobs, audio serving, delete, path-traversal guard).

### Note on proxies

The upstream client honors `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`. Some environments
list IPv6 addresses in `NO_PROXY` (e.g. `[::1]`), which makes `httpx.Client()` raise
while parsing them; in that case the client automatically falls back to
`trust_env=False` so synthesis keeps working.
