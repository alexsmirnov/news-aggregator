from pathlib import Path
from typing import NamedTuple

from pydantic import HttpUrl, SecretStr, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Aggregation(BaseModel):
    name: str
    miniflux_category: str
    focus: str


NEWS_FOCUS: str = (
    "- President Trump's actions, lawsuits, and executive orders\n"
    "- Tariffs and their effects on the U.S. and world economy\n"
    "- Job market, especially related to AI technologies\n"
    "- War in Ukraine\n"
    "- Midterm elections and U.S. political parties\n"
    "- Bay Area news"
)

DEFAULT_AGGREGATIONS: list[Aggregation] = [
    Aggregation(name="news", miniflux_category="news", focus=NEWS_FOCUS),
]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    miniflux_api_base: HttpUrl
    miniflux_api_key: SecretStr
    litellm_api_key: SecretStr
    litellm_router: HttpUrl
    digest_output_dir: Path

    aggregations: list[Aggregation] = DEFAULT_AGGREGATIONS

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
