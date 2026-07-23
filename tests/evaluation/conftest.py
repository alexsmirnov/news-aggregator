import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

sys.path.insert(0, str(Path(__file__).parent))

import pytest
import pytest_asyncio
from deepeval.models import GPTModel
from pydantic import ValidationError

from news.digest.llm_client import LlmClient
from news.digest.miniflux_client import MinifluxClient
from news.digest.schemas import DigestRecord, NewsRecord, RssEntry
from news.digest.service import DigestService
from news.settings import Settings

DATA_DIR = Path(__file__).parent / "data"

_ENTRY_PREFIX = "rss_entries_"
_ENTRY_SUFFIX = ".json"


def _collect_dataset_ids() -> list[str]:
    """Discover dataset suffixes from rss_entries_*.json files."""
    if not DATA_DIR.exists():
        return []
    ids: list[str] = []
    pattern = f"{_ENTRY_PREFIX}*{_ENTRY_SUFFIX}"
    for path in sorted(DATA_DIR.glob(pattern)):
        suffix = path.name.removeprefix(
            _ENTRY_PREFIX
        ).removesuffix(_ENTRY_SUFFIX)
        ids.append(suffix)
    return ids


@pytest.fixture(scope="module")
def eval_settings() -> Settings:
    try:
        return Settings()  # type: ignore
    except ValidationError:
        pytest.skip("evaluation credentials not configured")


@pytest.fixture(scope="module", params=_collect_dataset_ids())
def dataset_id(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="module")
def frozen_entries(dataset_id: str) -> list[RssEntry]:
    path = DATA_DIR / f"{_ENTRY_PREFIX}{dataset_id}{_ENTRY_SUFFIX}"
    if not path.exists():
        pytest.skip(
            f"frozen dataset missing: {path} - run "
            "tests/evaluation/capture_dataset.py"
        )
    return [RssEntry.model_validate(entry) for entry in _load_json(path)]


@pytest.fixture(scope="module")
def expected_groups(dataset_id: str) -> list[dict[str, object]]:
    return _load_required_data(f"expected_groups_{dataset_id}.json")


@pytest.fixture(scope="module")
def expected_summaries(dataset_id: str) -> list[dict[str, object]]:
    return _load_required_data(f"expected_summaries_{dataset_id}.json")


@pytest.fixture(scope="module")
def judge(eval_settings: Settings) -> GPTModel:
    return GPTModel(
        model=eval_settings.eval_judge_model,
        api_key=eval_settings.litellm_api_key.get_secret_value(),
        base_url=str(eval_settings.litellm_router),
        temperature=1,
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
