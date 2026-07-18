from pathlib import Path

import pytest
from pydantic import HttpUrl, ValidationError

from news.settings import Settings

ENV_VARS = {
    "MINIFLUX_API_BASE": "http://localhost:4080",
    "MINIFLUX_API_KEY": "mk",
    "LITELLM_API_KEY": "lk",
    "LITELLM_ROUTER": "http://localhost:4000",
    "DIGEST_OUTPUT_DIR": "/tmp/digests",
}


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    for key in ENV_VARS:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.chdir(tmp_path)


def test_settings_loads_from_process_environment(clean_env, monkeypatch):
    for key, value in ENV_VARS.items():
        monkeypatch.setenv(key, value)

    settings = Settings()

    assert settings.miniflux_api_base == HttpUrl(
        ENV_VARS["MINIFLUX_API_BASE"]
    )
    assert settings.miniflux_api_key.get_secret_value() == "mk"
    assert settings.litellm_api_key.get_secret_value() == "lk"
    assert settings.litellm_router == HttpUrl(ENV_VARS["LITELLM_ROUTER"])
    assert settings.digest_output_dir == Path("/tmp/digests")


def test_settings_missing_required_field_raises_validation_error(
    clean_env,
):
    with pytest.raises(ValidationError, match="digest_output_dir"):
        Settings()


def test_settings_loads_from_dotenv_file(clean_env, tmp_path):
    lines = "\n".join(f"{k}={v}" for k, v in ENV_VARS.items())
    (tmp_path / ".env").write_text(lines)

    settings = Settings()

    assert settings.miniflux_api_base == HttpUrl(
        ENV_VARS["MINIFLUX_API_BASE"]
    )
    assert settings.miniflux_api_key.get_secret_value() == "mk"
    assert settings.litellm_api_key.get_secret_value() == "lk"
    assert settings.litellm_router == HttpUrl(ENV_VARS["LITELLM_ROUTER"])
    assert settings.digest_output_dir == Path(
        ENV_VARS["DIGEST_OUTPUT_DIR"]
    )


def test_process_environment_overrides_dotenv_file(
    clean_env, tmp_path, monkeypatch
):
    file_vars = {**ENV_VARS, "MINIFLUX_API_BASE": "http://from-file:1"}
    lines = "\n".join(f"{k}={v}" for k, v in file_vars.items())
    (tmp_path / ".env").write_text(lines)
    monkeypatch.setenv("MINIFLUX_API_BASE", "http://from-env:2")

    settings = Settings()

    assert settings.miniflux_api_base == HttpUrl("http://from-env:2")
