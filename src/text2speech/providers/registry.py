"""Provider registry: builds provider instances from configuration."""

from __future__ import annotations

from functools import lru_cache

from .base import ProviderError, SynthesisResult, TTSProvider
from .google import GoogleInteractionsProvider
from .openai_compatible import OpenAICompatibleProvider

_REGISTRY: dict[str, type[TTSProvider]] = {
    OpenAICompatibleProvider.type: OpenAICompatibleProvider,
    GoogleInteractionsProvider.type: GoogleInteractionsProvider,
    "gemini": GoogleInteractionsProvider,  # friendly alias
}


def register_provider_type(name: str, cls: type[TTSProvider]) -> None:
    """Register a custom provider backend under ``type`` in the providers config."""
    _REGISTRY[name] = cls


def build_provider(config) -> TTSProvider:
    cls = _REGISTRY.get(config.type)
    if cls is None:
        raise ValueError(
            f"Unknown provider type '{config.type}' for provider '{config.id}'. "
            f"Available types: {sorted(_REGISTRY)}"
        )
    return cls(config)


@lru_cache
def get_providers() -> tuple[TTSProvider, ...]:
    """Build all *enabled* providers from the config file (cached)."""
    from ..config import load_providers, get_settings

    configs = [c for c in load_providers(get_settings().providers_file) if c.enabled]
    return tuple(build_provider(c) for c in configs)


def reset_cache() -> None:
    get_providers.cache_clear()


def get_provider(provider_id: str) -> TTSProvider:
    """Return the enabled provider with the given id."""
    for provider in get_providers():
        if provider.config.id == provider_id:
            return provider
    available = ", ".join(p.config.id for p in get_providers()) or "<none>"
    raise KeyError(f"Unknown provider '{provider_id}'. Available: {available}")


__all__ = [
    "SynthesisResult",
    "TTSProvider",
    "ProviderError",
    "OpenAICompatibleProvider",
    "build_provider",
    "get_provider",
    "get_providers",
    "register_provider_type",
    "reset_cache",
]
