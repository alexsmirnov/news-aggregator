from pathlib import Path
from typing import Any

from pydantic import HttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    miniflux_api_base: HttpUrl
    miniflux_api_key: SecretStr
    litellm_api_key: SecretStr
    litellm_router: HttpUrl
    digest_output_dir: Path

    fetch_lookback_hours: int = 24
    fetch_limit: int = 10000
    entry_content_max_chars: int = 1000
    refine_max_links: int = 20
    model_trending: str = "sonar-reasoning-pro"
    model_grouping: str = "gemini-flash"
    model_refinement: str = "gemini-flash"
    eval_judge_model: str = "gpt"
    retry_attempts: int = 3
    retry_min_wait_s: int = 2
    retry_max_wait_s: int = 30
    schedule_interval_hours: int = 12

    def __init__(self, **kwargs: Any) -> None:
        # Explicit __init__ so pyright does not require the env-sourced
        # fields as call-site arguments (pydantic-settings populates
        # them from the environment/.env at runtime).
        super().__init__(**kwargs)
