import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent))

import pytest
import pytest_asyncio
from deepeval.models import LiteLLMModel
from pydantic import ValidationError

from news.digest.llm_client import LlmClient
from news.digest.miniflux_client import MinifluxClient
from news.digest.schemas import DigestRecord, NewsRecord, RssEntry
from news.digest.service import DigestService
from news.settings import Settings

DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def eval_settings() -> Settings:
    try:
        return Settings()
    except ValidationError:
        pytest.skip("evaluation credentials not configured")


@pytest.fixture(scope="module")
def frozen_entries() -> list[RssEntry]:
    path = DATA_DIR / "rss_entries.json"
    if not path.exists():
        pytest.skip(
            "frozen dataset missing - run "
            "tests/evaluation/capture_dataset.py"
        )
    return [RssEntry.model_validate(entry) for entry in _load_json(path)]


@pytest.fixture(scope="module")
def expected_groups() -> list[dict[str, object]]:
    return _load_required_data("expected_groups.json")


@pytest.fixture(scope="module")
def expected_summaries() -> list[dict[str, object]]:
    return _load_required_data("expected_summaries.json")


@pytest.fixture(scope="module")
def judge(eval_settings: Settings) -> LiteLLMModel:
    return LiteLLMModel(
        model=eval_settings.eval_judge_model,
        api_key=eval_settings.litellm_api_key.get_secret_value(),
        base_url=str(eval_settings.litellm_router),
        temperature=0,
    )


@pytest_asyncio.fixture(scope="module")
async def grouping_run(
    eval_settings: Settings,
    frozen_entries: list[RssEntry],
) -> tuple[str, str, list[NewsRecord]]:
    formatted = DigestService.format_entries(
        frozen_entries,
        content_max_chars=eval_settings.grouping_content_max_chars,
    )
    llm = LlmClient(
        eval_settings.litellm_api_key, str(eval_settings.litellm_router)
    )
    service = DigestService(
        eval_settings,
        cast(MinifluxClient, object()),
        llm,
    )
    try:
        records = await service.extract_groups(
            formatted,
            focus=eval_settings.aggregations[0].focus,
        )
    finally:
        await llm.aclose()
    actual_json = json.dumps(
        [record.model_dump(mode="json") for record in records], indent=2
    )
    return formatted, actual_json, records


@pytest_asyncio.fixture(scope="module")
async def refined_run(
    eval_settings: Settings,
    frozen_entries: list[RssEntry],
    grouping_run: tuple[str, str, list[NewsRecord]],
) -> list[DigestRecord]:
    _, _, records = grouping_run
    llm = LlmClient(
        eval_settings.litellm_api_key, str(eval_settings.litellm_router)
    )
    service = DigestService(
        eval_settings,
        cast(MinifluxClient, object()),
        llm,
    )
    try:
        return await service.refine_all(
            records, frozen_entries, today=datetime.now(UTC).date()
        )
    finally:
        await llm.aclose()


def _load_json(path: Path) -> list[dict[str, object]]:
    return json.loads(path.read_text())


def _load_required_data(filename: str) -> list[dict[str, object]]:
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip(f"frozen evaluation data missing: {path}")
    return _load_json(path)
