"""Application settings and provider configuration loading."""

from __future__ import annotations

import json
import os
import re
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Project root = parents[2] of this file (src/text2speech/config.py -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# User-level configuration lives in the home directory, so it survives project
# updates and is shared by every checkout on the machine.
USER_CONFIG_DIR = Path.home() / ".text2speech"
DEFAULT_PROVIDERS_FILE = USER_CONFIG_DIR / "config.json"
DEFAULT_ENV_FILE = USER_CONFIG_DIR / ".env"
LEGACY_ENV_FILE = PROJECT_ROOT / ".env"  # pre-consolidation location
CONFIG_TEMPLATE = PROJECT_ROOT / "config" / "providers.json"


def resolve_env_file(path: Path | str | None = None) -> Path:
    """Locate the .env file: explicit path > $T2S_ENV_FILE > user config dir > legacy project file."""
    if path:
        return Path(path)
    override = os.environ.get("T2S_ENV_FILE")
    if override:
        return Path(override)
    if DEFAULT_ENV_FILE.exists() or not LEGACY_ENV_FILE.exists():
        return DEFAULT_ENV_FILE
    return LEGACY_ENV_FILE


def load_dotenv_file(path: Path | str | None = None) -> Path:
    """Load a .env file into ``os.environ``.

    pydantic-settings only consumes .env for its own (``T2S_``-prefixed) fields and
    never populates ``os.environ`` — but provider API keys and ``{env:VAR}`` header
    references are resolved straight from the environment. Existing environment
    variables always win over .env values.
    """
    env_path = resolve_env_file(path)
    load_dotenv(env_path)
    return env_path


# Make .env values visible to os.environ lookups as soon as the package is imported.
load_dotenv_file()


class Settings(BaseSettings):
    """Runtime settings, overridable via environment variables (prefix ``T2S_``) or a .env file."""

    model_config = SettingsConfigDict(
        env_prefix="T2S_", env_file=resolve_env_file(), extra="ignore"
    )

    host: str = "127.0.0.1"
    port: int = 8000
    outputs_dir: Path = PROJECT_ROOT / "outputs"
    providers_file: Path = DEFAULT_PROVIDERS_FILE
    # Texts longer than chunk_chars are split into sentence-aware chunks, queued and
    # merged back into a single audio file (see synthesis.py). max_text_chars stays
    # the hard cap on the *total* input.
    chunk_chars: int = 20_000
    max_text_chars: int = 200_000
    request_timeout_seconds: float = 180.0
    default_response_format: str = "mp3"


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    if settings.chunk_chars > settings.max_text_chars:
        # A chunk larger than the hard cap could never be sent on its own.
        settings.chunk_chars = settings.max_text_chars
    settings.outputs_dir.mkdir(parents=True, exist_ok=True)
    return settings


_ENV_REF_RE = re.compile(r"\{env:([A-Za-z_][A-Za-z0-9_]*)\}")


def _interpolate(value: Any) -> Any:
    """Replace ``{env:VAR}`` placeholders inside strings with environment values."""
    if not isinstance(value, str):
        return value

    def repl(match: re.Match[str]) -> str:
        return os.environ.get(match.group(1), "")

    return _ENV_REF_RE.sub(repl, value)


class ProviderConfig(BaseModel):
    """One TTS provider entry from the providers config file."""

    id: str
    name: str
    type: str = "openai-compatible"
    base_url: str
    api_key_env: str | None = Field(
        default=None, description="Name of the environment variable holding the API key"
    )
    models: list[str] = Field(default_factory=list)
    default_model: str | None = None
    voices: list[str] = Field(default_factory=list)
    default_voice: str | None = Field(
        default=None, description="Voice used when the request does not specify one"
    )
    style: str | None = Field(
        default=None,
        description="Optional speech-style hint (e.g. Google speech_metadata.style)",
    )
    extra_headers: dict[str, str] = Field(default_factory=dict)
    extra_body: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True

    @property
    def api_key(self) -> str | None:
        if not self.api_key_env:
            return None
        return os.environ.get(self.api_key_env) or None

    def resolved_extra_headers(self) -> dict[str, str]:
        return {k: _interpolate(v) for k, v in self.extra_headers.items()}


class ProvidersFile(BaseModel):
    providers: list[ProviderConfig] = Field(default_factory=list)


def seed_config_file(target: Path, template: Path | None = None) -> bool:
    """Create the user config from the bundled template if it does not exist yet.

    Returns True when the target file exists afterwards.
    """
    target = Path(target)
    if target.exists():
        return True
    template = Path(template) if template else CONFIG_TEMPLATE
    if not template.exists():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(template, target)
    return True


def load_providers(path: Path | str | None = None) -> list[ProviderConfig]:
    """Load provider definitions from the user config file.

    On first run the bundled template (``config/providers.json``) is copied to
    ``~/.text2speech/config.json`` so there is always something to edit.
    """
    settings = get_settings()
    file_path = Path(path) if path else settings.providers_file
    if not seed_config_file(file_path):
        raise FileNotFoundError(
            f"Providers config not found: {file_path}. "
            f"Create it (see the template at {CONFIG_TEMPLATE}) or point "
            "T2S_PROVIDERS_FILE at an existing file."
        )
    data = json.loads(file_path.read_text(encoding="utf-8"))
    return ProvidersFile.model_validate(data).providers
