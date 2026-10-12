"""Tests for settings and provider-config loading (including .env handling)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

import text2speech.config as config_module
from text2speech.config import (
    DEFAULT_ENV_FILE,
    DEFAULT_PROVIDERS_FILE,
    LEGACY_ENV_FILE,
    USER_CONFIG_DIR,
    ProviderConfig,
    get_settings,
    load_dotenv_file,
    load_providers,
    resolve_env_file,
    seed_config_file,
)


class TestDotenvLoading:
    def test_dotenv_values_reach_os_environ(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("T2S_TEST_DOTENV_KEY=from-dotenv\n", encoding="utf-8")
        monkeypatch.delenv("T2S_TEST_DOTENV_KEY", raising=False)
        load_dotenv_file(env_file)
        assert os.environ["T2S_TEST_DOTENV_KEY"] == "from-dotenv"

    def test_real_environment_wins_over_dotenv(self, tmp_path, monkeypatch):
        env_file = tmp_path / ".env"
        env_file.write_text("T2S_TEST_DOTENV_KEY=from-dotenv\n", encoding="utf-8")
        monkeypatch.setenv("T2S_TEST_DOTENV_KEY", "from-real-env")
        load_dotenv_file(env_file)
        assert os.environ["T2S_TEST_DOTENV_KEY"] == "from-real-env"

    def test_missing_dotenv_file_is_tolerated(self, tmp_path):
        # A missing .env must not break imports on fresh checkouts.
        load_dotenv_file(tmp_path / "does-not-exist.env")


class TestEnvFileResolution:
    """The .env lives next to config.json in ~/.text2speech, with a legacy fallback."""

    def test_default_location_is_user_config_dir(self):
        assert DEFAULT_ENV_FILE == USER_CONFIG_DIR / ".env"
        assert USER_CONFIG_DIR == Path.home() / ".text2speech"

    def test_explicit_path_wins(self, tmp_path):
        assert resolve_env_file(tmp_path / "x.env") == tmp_path / "x.env"

    def test_t2s_env_file_override(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom.env"
        monkeypatch.setenv("T2S_ENV_FILE", str(custom))
        assert resolve_env_file() == custom

    def test_user_file_preferred_over_legacy(self, tmp_path, monkeypatch):
        user = tmp_path / "user" / ".env"
        legacy = tmp_path / "legacy" / ".env"
        user.parent.mkdir(parents=True)
        legacy.parent.mkdir(parents=True)
        user.write_text("A=1\n", encoding="utf-8")
        legacy.write_text("A=2\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "DEFAULT_ENV_FILE", user)
        monkeypatch.setattr(config_module, "LEGACY_ENV_FILE", legacy)
        assert resolve_env_file() == user

    def test_legacy_file_used_when_user_file_missing(self, tmp_path, monkeypatch):
        user = tmp_path / "user" / ".env"  # does not exist
        legacy = tmp_path / "legacy" / ".env"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("A=2\n", encoding="utf-8")
        monkeypatch.setattr(config_module, "DEFAULT_ENV_FILE", user)
        monkeypatch.setattr(config_module, "LEGACY_ENV_FILE", legacy)
        assert resolve_env_file() == legacy

    def test_defaults_to_user_location_when_nothing_exists(self, tmp_path, monkeypatch):
        user = tmp_path / "user" / ".env"
        legacy = tmp_path / "legacy" / ".env"
        monkeypatch.setattr(config_module, "DEFAULT_ENV_FILE", user)
        monkeypatch.setattr(config_module, "LEGACY_ENV_FILE", legacy)
        assert resolve_env_file() == user


class TestConfigLocation:
    def test_default_config_lives_in_home_dir(self):
        settings = get_settings()
        assert settings.providers_file == DEFAULT_PROVIDERS_FILE
        assert DEFAULT_PROVIDERS_FILE == USER_CONFIG_DIR / "config.json"
        assert USER_CONFIG_DIR == Path.home() / ".text2speech"

    def test_env_override_still_wins(self, tmp_path, monkeypatch):
        custom = tmp_path / "custom-config.json"
        custom.write_text(json.dumps({"providers": []}), encoding="utf-8")
        monkeypatch.setenv("T2S_PROVIDERS_FILE", str(custom))
        get_settings.cache_clear()
        try:
            assert get_settings().providers_file == custom
        finally:
            get_settings.cache_clear()

    def teardown_method(self, method) -> None:
        get_settings.cache_clear()


class TestConfigSeeding:
    def test_existing_file_is_left_alone(self, tmp_path):
        target = tmp_path / "config.json"
        target.write_text('{"providers": []}', encoding="utf-8")
        assert seed_config_file(target, tmp_path / "template.json") is True
        assert target.read_text(encoding="utf-8") == '{"providers": []}'

    def test_missing_file_is_seeded_from_template(self, tmp_path):
        target = tmp_path / "nested" / "config.json"
        template = tmp_path / "template.json"
        template.write_text('{"providers": [{"id": "t"}]}', encoding="utf-8")
        assert seed_config_file(target, template) is True
        assert target.read_text(encoding="utf-8") == '{"providers": [{"id": "t"}]}'

    def test_no_template_means_no_seed(self, tmp_path):
        assert seed_config_file(tmp_path / "config.json", tmp_path / "nope.json") is False

    def test_load_providers_seeds_from_template(self, tmp_path, monkeypatch):
        target = tmp_path / "home" / "config.json"
        template = tmp_path / "template.json"
        template.write_text(
            json.dumps(
                {
                    "providers": [
                        {
                            "id": "seeded",
                            "name": "Seeded",
                            "base_url": "https://seeded.invalid",
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setenv("T2S_PROVIDERS_FILE", str(target))
        monkeypatch.setattr(config_module, "CONFIG_TEMPLATE", template)
        get_settings.cache_clear()
        try:
            providers = load_providers()
            assert [p.id for p in providers] == ["seeded"]
            assert target.exists(), "load_providers should seed the user config on first run"
        finally:
            get_settings.cache_clear()

    def test_load_providers_without_any_config_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T2S_PROVIDERS_FILE", str(tmp_path / "missing.json"))
        monkeypatch.setattr(config_module, "CONFIG_TEMPLATE", tmp_path / "no-template.json")
        get_settings.cache_clear()
        try:
            with pytest.raises(FileNotFoundError, match="Providers config not found"):
                load_providers()
        finally:
            get_settings.cache_clear()


class TestProviderConfig:
    def _write_config(self, tmp_path, monkeypatch, entry: dict) -> None:
        providers_file = tmp_path / "providers.json"
        providers_file.write_text(json.dumps({"providers": [entry]}), encoding="utf-8")
        monkeypatch.setenv("T2S_PROVIDERS_FILE", str(providers_file))
        get_settings.cache_clear()

    def teardown_method(self, method) -> None:
        get_settings.cache_clear()

    def test_api_key_from_env_var(self, tmp_path, monkeypatch):
        monkeypatch.setenv("T2S_TEST_PROVIDER_KEY", "env-key-123")
        self._write_config(
            tmp_path,
            monkeypatch,
            {
                "id": "p",
                "name": "P",
                "base_url": "https://x.invalid",
                "api_key_env": "T2S_TEST_PROVIDER_KEY",
            },
        )
        providers = load_providers()
        assert providers[0].api_key == "env-key-123"

    def test_api_key_from_dotenv_file(self, tmp_path, monkeypatch):
        monkeypatch.delenv("T2S_TEST_DOTENV_PROVIDER_KEY", raising=False)
        env_file = tmp_path / ".env"
        env_file.write_text(
            "T2S_TEST_DOTENV_PROVIDER_KEY=dotenv-key-456\n", encoding="utf-8"
        )
        load_dotenv_file(env_file)
        self._write_config(
            tmp_path,
            monkeypatch,
            {
                "id": "p",
                "name": "P",
                "base_url": "https://x.invalid",
                "api_key_env": "T2S_TEST_DOTENV_PROVIDER_KEY",
            },
        )
        assert load_providers()[0].api_key == "dotenv-key-456"

    def test_missing_api_key_is_none(self, monkeypatch):
        monkeypatch.delenv("T2S_TEST_ABSENT_KEY", raising=False)
        config = ProviderConfig(
            id="p", name="P", base_url="https://x.invalid", api_key_env="T2S_TEST_ABSENT_KEY"
        )
        assert config.api_key is None

    def test_env_ref_interpolation(self, monkeypatch):
        monkeypatch.setenv("T2S_TEST_SITE_NAME", "my-site")
        config = ProviderConfig(
            id="p",
            name="P",
            base_url="https://x.invalid",
            extra_headers={"X-Title": "{env:T2S_TEST_SITE_NAME}", "X-Plain": "static"},
        )
        assert config.resolved_extra_headers() == {
            "X-Title": "my-site",
            "X-Plain": "static",
        }
