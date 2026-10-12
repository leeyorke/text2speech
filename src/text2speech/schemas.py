"""Pydantic schemas for the HTTP API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ProviderInfo(BaseModel):
    id: str
    name: str
    type: str
    base_url: str
    models: list[str] = Field(default_factory=list)
    default_model: str | None = None
    voices: list[str] = Field(default_factory=list)
    default_voice: str | None = None
    has_api_key: bool = False
    api_key_env: str | None = None
    supported_formats: list[str] | None = None


class TTSJob(BaseModel):
    filename: str
    url: str
    provider_id: str
    provider_name: str
    model: str
    format: str
    characters: int
    kind: str
    source_name: str
    generation_id: str | None = None
    text_preview: str = ""
    chunked: bool = False
    chunks: int = 1
    generation_ids: list[str] = Field(default_factory=list)


class OutputFile(BaseModel):
    filename: str
    url: str
    size_bytes: int
    modified_at: float


class HealthResponse(BaseModel):
    status: str = "ok"
    providers: int = 0
    version: str = ""
    chunk_chars: int = 0
    max_text_chars: int = 0
