"""Provider package: pluggable backends that call TTS model APIs."""

from .base import ProviderError, SynthesisResult, TTSProvider
from .google import GoogleInteractionsProvider
from .openai_compatible import OpenAICompatibleProvider
from .registry import (
    build_provider,
    get_provider,
    get_providers,
    register_provider_type,
    reset_cache,
)

__all__ = [
    "ProviderError",
    "SynthesisResult",
    "TTSProvider",
    "GoogleInteractionsProvider",
    "OpenAICompatibleProvider",
    "build_provider",
    "get_provider",
    "get_providers",
    "register_provider_type",
    "reset_cache",
]
