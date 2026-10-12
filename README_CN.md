# text2speech（中文文档）

<p align="center">
  <strong>简体中文</strong>
  &nbsp;·&nbsp;
  <a href="./README.md">English</a>
</p>

一个把**文本（粘贴、`.txt`、`.md`）转换为语音**的网页服务，通过可配置的文本转语音模型 API 实现。
技术栈：Python + [uv](https://docs.astral.sh/uv/) + FastAPI，单页 Web UI，无构建步骤。
English version: [README.md](README.md)。

```
浏览器 UI ──► FastAPI (/api/tts) ──► 提供商（OpenAI 兼容 /audio/speech 等） ──► 音频文件 ──► 播放 + 下载
                    │
                    └─ txt/markdown 提取与规范化（代码块默认剔除）
```

## 快速开始

```bash
# 1. 安装依赖到本地虚拟环境（uv 全权管理）
uv sync

# 2. 配置 API 密钥
cp .env.example ~/.text2speech/.env
#    编辑 ~/.text2speech/.env，设置 OPENROUTER_API_KEY（https://openrouter.ai/keys）
#    和/或 GEMINI_API_KEY（https://aistudio.google.com/apikey）

# 3. 启动服务
uv run text2speech          # 或：uv run python -m uvicorn text2speech.main:app
```

所有用户级配置都集中在 **`~/.text2speech/`**：`config.json`（提供商定义）和 `.env`
（API 密钥与 `T2S_*` 设置）。首次运行时，项目自带的模板 `config/providers.json` 会被
自动复制为 `~/.text2speech/config.json`。

打开 <http://127.0.0.1:8000>，粘贴文本或拖入 `.txt`/`.md` 文件，选择提供商，点击
**Generate speech**。音频保存在 `outputs/`，可在浏览器中直接播放和下载。

### 没有 API key 也能试用

内置了一个模拟 TTS 提供商，可离线跑通整条链路：

```bash
# 终端 1 —— 假上游，返回一段生成的 WAV 提示音
uv run python -m scripts.mock_provider_server

# 终端 2 —— 应用指向 mock
$env:T2S_PROVIDERS_FILE = "config/providers.smoke.json"   # PowerShell
$env:MOCK_TTS_API_KEY = "smoke-key"
uv run python -m uvicorn text2speech.main:app --port 8123

# 终端 3 —— 端到端检查（50 项断言）
uv run python -m scripts.smoke_test
```

## 工作原理

### 输入处理

| 输入 | 行为 |
| --- | --- |
| 粘贴的文本 | 原样使用（仅规范化空白字符） |
| `.txt` 上传 | 按文本读取（容忍 UTF-8/BOM/GBK 编码） |
| `.md` / `.markdown` 上传 | 转换为可朗读的纯文本：标题、加粗、链接、图片、列表、表格、引用和 HTML 注释全部展平 |

Markdown 的**围栏代码块默认剔除**——代码被朗读出来只是噪音。勾选
*"Read code blocks aloud"*（或传 `include_code_blocks=true`）可保留。
超过 10 MB 的文件或超过 `T2S_MAX_TEXT_CHARS`（默认 200000 字符）的文本会被明确拒绝。

### 实时进度

UI 通过 Server-Sent Events（`POST /api/tts/stream`）流式推送生成进度，每个阶段都
可见，而不是干等一个转圈：文本提取 → 请求已提交 → 等待期间的每秒心跳 → 收到音频 →
保存。UI 进度条同步反映这些事件。`POST /api/tts` 保留给纯 JSON 客户端（curl、脚本）。

### 分段合成（长文本）

当待朗读文本超过 **`T2S_CHUNK_CHARS`**（默认 20000）时，自动执行：

1. **切分** —— 按“段落 → 句子（中英文标点）→ 硬切”的优先级切分，不会切断句子
   （单个句子超长才硬切）；
2. **排队合成** —— 逐块顺序调用提供商（顺序执行，对速率限制友好），每块有独立的
   进度事件和独立超时；
3. **合并** —— 合并回单一音频文件，作为一个整体保存、播放、下载。

UI 在字符计数旁显示预估块数，生成期间显示逐块状态（⏳ / ✓ / ✗），结果上标注
“N chunks merged”。API 侧通过 `job.chunks` / `job.chunked` / `job.generation_ids`
暴露相同信息。任一块失败则整个任务中止并报 `Chunk k/n failed: …`——不会返回部分音频。

合并对 **WAV 和原始 PCM 是采样级精确的**；容器格式（MP3、Opus、AAC、FLAC）按字节
拼接，播放器会渲染为一段连续音频。`T2S_MAX_TEXT_CHARS`（默认 200000）仍是*总输入*
的硬上限。

### 提供商

提供商在 **`~/.text2speech/config.json`** 中声明（用户级，所有 checkout 共享；
可用 `T2S_PROVIDERS_FILE` 覆盖）。首次运行会自动把项目模板
[`config/providers.json`](config/providers.json) 复制过去，开箱即可编辑。内置两种
提供商类型：

| `type` | API 形态 | 说明 |
| --- | --- | --- |
| `openai-compatible` | `POST {base_url}/audio/speech`，`Authorization: Bearer` | OpenRouter（默认）及大多数 TTS 网关 |
| `google` | Gemini **Interactions API** `POST {base_url}/v1beta/interactions`，`x-goog-api-key` 头 | `gemini-3.8-flash-tts` 等；别名 `gemini` |

想加多少提供商都可以，它们会出现在 UI 下拉框里。

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

通用字段：

* `api_key_env` —— 存放密钥的环境变量名（密钥只存在于 `.env`/环境中，绝不写进配置文件）。
* `models` / `default_model` —— UI 中提供的模型 ID。
* `voices` / `default_voice` —— UI 中提供的语音；请求未指定时使用 `default_voice`。
* `style` —— 可选的语音风格提示。对 Google 提供商会作为 `speech_metadata.style`
  注解发送（控制整段演绎方式，*不会*被朗读出来）。
* `extra_headers` / `extra_body` —— 合并进每个请求；`{env:VAR}` 占位符从环境展开。
* `enabled: false` —— 保留条目但不在 UI 显示。
* 修改文件后，通过 `POST /api/providers/reload` 重载，或重启服务。

服务实际发出的参考请求：

```python
# openai-compatible（例如 OpenRouter）
requests.post(
    url="https://openrouter.ai/api/v1/audio/speech",
    headers={"Authorization": "Bearer <OPENROUTER_API_KEY>", "Content-Type": "application/json"},
    json={"model": "fish-audio/s2.1-pro-free:free", "input": "Hello! This is a text-to-speech test."},
)
```

```sh
# google（Gemini Interactions API）
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
# -> steps[].content[] 中 type=="audio" 的项，.data 为 base64（单轮请求返回 WAV）
```

**各提供商支持的格式不同**：OpenAI 兼容端点接受常见的
`mp3`/`wav`/`opus`/`aac`/`flac`/`pcm`，而 Google TTS 只产出 `wav`（默认）和 `pcm`
（原始 `audio/l16`）。UI 会自动禁用不支持的格式，API 也会返回明确的 400 并列出该
提供商支持的格式。

自定义后端可以通过继承提供商协议并调用
`register_provider_type("my-type", MyProvider)` 接入——JSON 里的 `type` 即选择它。

### HTTP API

| 方法 | 端点 | 用途 |
| --- | --- | --- |
| `GET` | `/` | Web UI |
| `GET` | `/api/health` | 存活检查 + 提供商数量 |
| `GET` | `/api/providers` | 已配置的提供商（模型、语音、密钥状态） |
| `POST` | `/api/providers/reload` | 重新读取提供商配置 |
| `POST` | `/api/tts` | 合成（multipart 表单）→ JSON 任务 |
| `POST` | `/api/tts/stream` | 同上，但流式推送 SSE 进度事件 |
| `GET` | `/api/audio/{filename}` | 下载生成的音频 |
| `GET` | `/api/outputs` | 最近的输出 |
| `DELETE` | `/api/outputs/{filename}` | 删除生成的音频 |

`POST /api/tts` 表单字段：`provider_id`（必填）、`text` **或** `file`
（`.txt`/`.md`/`.markdown`）、`model`、`voice`、`response_format`
（`mp3`/`wav`/`opus`/`aac`/`flac`/`pcm`）、`speed`、`include_code_blocks`。
返回 JSON，含 `url`、`generation_id`、`characters`、`kind` 等字段。

```bash
curl -F "provider_id=openrouter" -F "text=Hello world" \
     -F "model=fish-audio/s2.1-pro-free:free" \
     http://127.0.0.1:8000/api/tts
```

SSE 事件形如：

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

错误以 `{"stage": "error", "message": "…"}` 在同一条流上返回（分块失败时为
`Chunk k/n failed: …`）。

## 配置

设置来自环境变量（前缀 `T2S_`，也可通过 `.env` 文件提供）：

| 变量 | 默认值 | 含义 |
| --- | --- | --- |
| `T2S_HOST` / `T2S_PORT` | `127.0.0.1` / `8000` | 监听地址 |
| `T2S_OUTPUTS_DIR` | `./outputs` | 音频文件输出目录 |
| `T2S_PROVIDERS_FILE` | `~/.text2speech/config.json` | 提供商定义文件 |
| `T2S_ENV_FILE` | `~/.text2speech/.env` | 密钥 / 应用设置文件 |
| `T2S_CHUNK_CHARS` | `20000` | 分块阈值：超过则切分、排队、合并 |
| `T2S_MAX_TEXT_CHARS` | `200000` | 单次请求总输入的硬上限 |
| `T2S_REQUEST_TIMEOUT_SECONDS` | `180` | 上游 HTTP 超时（对每个分块独立生效） |

`.env` 位于 **`~/.text2speech/.env`**，与 `config.json` 并列，启动时加载进进程环境——
因此提供商密钥（`OPENROUTER_API_KEY`、`GEMINI_API_KEY` 等）和 `{env:VAR}` 头引用都能
从它解析。可用 `T2S_ENV_FILE` 指定其他位置；当 `~/.text2speech/.env` 不存在时，仍会
回退到项目根目录的旧位置 `./.env`。**修改后需重启服务生效。** 真实环境变量始终优先于
`.env` 中的值。

## 项目结构

```
src/text2speech/
├── main.py                     # FastAPI 应用、端点、CLI 入口
├── config.py                   # 设置 + 提供商配置加载（首次运行种子化 ~/.text2speech）
├── schemas.py                  # API 模型
├── textio.py                   # txt/markdown → 可朗读文本 + 分句分块
├── audio.py                    # 分块音频合并
├── synthesis.py                # 分块编排（切分 → 排队 → 合并）
├── providers/
│   ├── base.py                 # 提供商协议、SynthesisResult、错误
│   ├── openai_compatible.py    # OpenAI 兼容 /audio/speech 客户端
│   ├── google.py               # Gemini Interactions API 客户端
│   └── registry.py             # 从配置构建提供商
└── static/index.html           # Web UI（无构建步骤）
config/providers.json           # 首次运行时复制到 ~/.text2speech/config.json 的模板
scripts/mock_provider_server.py # 离线测试用假上游（OpenAI + Google 两种形态）
scripts/smoke_test.py           # 针对运行中实例的端到端检查
tests/                          # pytest 测试套件
```

## 开发

```bash
uv sync --dev        # 或：uv sync（已包含 dev 组）
uv run pytest        # 144 个单元/集成测试
uv run python -m scripts.smoke_test   # 针对运行中的实例（见上文）
```

测试覆盖：markdown→文本转换（标题、链接、表格、代码块、编码）、分句分块、音频合并
（WAV 帧级精确、字节拼接）、分块编排（顺序、守恒性、失败中止、事件序列）、两个提供商
客户端（载荷、认证头、错误路径、mock transport）、配置位置与种子化逻辑，以及完整 API
栈（上传、校验错误、SSE 进度流、分块任务、音频服务、删除、路径穿越防护）。

### 关于代理的说明

上游客户端遵循 `HTTP_PROXY`/`HTTPS_PROXY`/`NO_PROXY`。某些环境的 `NO_PROXY` 中含有
IPv6 地址（如 `[::1]`），会导致 `httpx.Client()` 解析时报错；此时客户端会自动回退到
`trust_env=False`，保证合成功能不受影响。
