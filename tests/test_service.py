import json
import types
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import HttpUrl

from news.config import Aggregation
from news.digest.llm_client import LlmClient
from news.digest.miniflux_client import MinifluxClient
from news.digest.prompts import (
    grouping_user_prompt,
    refinement_system_prompt,
    refinement_user_prompt,
)
from news.digest.schemas import (
    Digest,
    NewsRecord,
    NewsResponse,
    RssEntry,
)
from news.digest.service import (
    DigestService,
    PipelineError,
    extract_groups,
    fetch_entries,
    format_entries,
    format_entry,
    refine_all,
    refine_record,
    strip_html,
    write_digest,
)
from news.settings import Settings

NOW = datetime(2026, 7, 17, 12, 0, 0)


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: object | None = None) -> datetime:
        return NOW


class FakeMiniflux:
    def __init__(
        self,
        entries: list[dict[str, Any]] | None = None,
        category_id: int = 7,
        category_error: Exception | None = None,
        entries_error: Exception | None = None,
    ) -> None:
        self.entries = entries if entries is not None else []
        self.category_id = category_id
        self.category_error = category_error
        self.entries_error = entries_error
        self.calls: list[dict[str, Any]] = []

    async def get_category_id(self, _title: str) -> int:
        if self.category_error is not None:
            raise self.category_error
        return self.category_id

    async def get_entries(
        self, category_id: int, **kwargs: Any
    ) -> list[dict[str, Any]]:
        self.calls.append({"category_id": category_id, **kwargs})
        if self.entries_error is not None:
            raise self.entries_error
        return self.entries


class FakeLlm:
    def __init__(
        self,
        chat_results: list[Any] | None = None,
        chat_parsed_results: list[Any] | None = None,
    ) -> None:
        self.chat_results = list(chat_results or [])
        self.chat_parsed_results = list(chat_parsed_results or [])
        self.chat_calls: list[tuple[Any, ...]] = []
        self.chat_parsed_calls: list[tuple[Any, ...]] = []

    async def chat(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> Any:
        self.chat_calls.append((model, messages, kwargs))
        result = self.chat_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result

    async def chat_parsed(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_format: Any,
        **kwargs: Any,
    ) -> Any:
        self.chat_parsed_calls.append(
            (model, messages, response_format, kwargs)
        )
        result = self.chat_parsed_results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


@pytest.fixture
def entry() -> RssEntry:
    return RssEntry(
        id=42,
        title="T",
        content="C",
        link="http://a",
        published_at="2026-07-16T10:00:00",
        source="F",
    )


@pytest.fixture
def fake_miniflux() -> type[FakeMiniflux]:
    return FakeMiniflux


@pytest.fixture
def fake_llm() -> type[FakeLlm]:
    return FakeLlm


@pytest.fixture
def settings_stub(tmp_path: Path) -> Settings:
    return cast(
        Settings,
        types.SimpleNamespace(
            miniflux_api_base="http://m.test",
            miniflux_api_key="k",
            litellm_api_key="l",
            litellm_router="http://r.test",
            digest_output_dir=tmp_path,
            fetch_lookback_hours=24,
            fetch_limit=10000,
            entry_content_max_chars=1000,
            refine_max_links=20,
            model_trending="sonar-reasoning-pro",
            model_grouping="gemini-flash",
            model_refinement="gemini-flash",
        ),
    )


@pytest.fixture
def news_agg() -> Aggregation:
    return Aggregation(name="news", miniflux_category="news", focus="FOCUS")


@pytest.fixture
def three_aggs(monkeypatch: pytest.MonkeyPatch) -> tuple[Aggregation, ...]:
    aggs = (
        Aggregation(name="a", miniflux_category="news", focus="fa"),
        Aggregation(name="b", miniflux_category="tech", focus="fb"),
        Aggregation(name="c", miniflux_category="economy", focus="fc"),
    )
    monkeypatch.setattr("news.config.AGGREGATIONS", aggs)
    return aggs


def test_strip_html_extracts_text() -> None:
    # Act
    text = strip_html("<p>Hello <b>World</b></p>")

    # Assert
    assert text == "Hello World"


def test_format_entry_block(entry: RssEntry) -> None:
    # Act
    text = format_entry(42, entry)

    # Assert
    assert text == (
        "# Entity 42\nTitle: T\nContent: C\nSource: F\nLink: http://a\n"
    )


def test_format_entries_joins_blocks() -> None:
    # Arrange
    e1 = RssEntry(
        id=1,
        title="T1",
        content="C1",
        link="http://a",
        published_at="2026-07-16T10:00:00",
        source="F1",
    )
    e2 = RssEntry(
        id=2,
        title="T2",
        content="C2",
        link="http://b",
        published_at="2026-07-16T10:00:00",
        source="F2",
    )

    # Act
    text = format_entries([e1, e2])

    # Assert
    assert text == format_entry(1, e1) + "\n" + format_entry(2, e2)


async def test_fetch_entries_maps_and_truncates(
    fake_miniflux: type[FakeMiniflux],
) -> None:
    # Arrange
    raw = {
        "id": 9,
        "title": "T",
        "url": "http://x",
        "content": "<p>" + "a" * 1500 + "</p>",
        "published_at": "2026-07-16",
        "feed": {"title": "Feed"},
    }
    client = fake_miniflux(entries=[raw])

    # Act
    entries = await fetch_entries(
        cast(MinifluxClient, client),
        category="news",
        lookback_hours=24,
        limit=10000,
        max_chars=1000,
        now=NOW,
    )

    # Assert
    assert len(entries) == 1
    result = entries[0]
    assert result.id == 9
    assert result.link == "http://x"
    assert result.source == "Feed"
    assert result.content == "a" * 1000
    assert client.calls == [
        {
            "category_id": 7,
            "published_after": int(
                (NOW - timedelta(hours=24)).timestamp()
            ),
            "order": "published_at",
            "limit": 10000,
        }
    ]


async def test_extract_groups_calls_trending_then_grouping(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    response = NewsResponse(
        records=[NewsRecord(title="T", summary="S", links=[])]
    )
    llm = fake_llm(chat_results=["TRENDS"], chat_parsed_results=[response])

    # Act
    records = await extract_groups(
        cast(LlmClient, llm),
        "FORMATTED",
        trending_model="sonar-reasoning-pro",
        grouping_model="gemini-flash",
        focus="FOCUS",
    )

    # Assert
    assert records == response.records
    assert llm.chat_calls[0][0] == "sonar-reasoning-pro"
    model, messages, response_format, _ = llm.chat_parsed_calls[0]
    assert model == "gemini-flash"
    assert response_format is NewsResponse
    assert "TRENDS" in messages[0]["content"]
    assert "FOCUS" in messages[0]["content"]
    assert messages[1]["content"] == grouping_user_prompt("FORMATTED")


async def test_extract_groups_raises_on_none_trending(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    llm = fake_llm(chat_results=[None])

    # Act / Assert
    with pytest.raises(PipelineError):
        await extract_groups(
            cast(LlmClient, llm),
            "FORMATTED",
            trending_model="sonar-reasoning-pro",
            grouping_model="gemini-flash",
            focus="FOCUS",
        )
    assert llm.chat_parsed_calls == []


async def test_extract_groups_raises_on_none_grouping(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    llm = fake_llm(chat_results=["TRENDS"], chat_parsed_results=[None])

    # Act / Assert
    with pytest.raises(PipelineError):
        await extract_groups(
            cast(LlmClient, llm),
            "FORMATTED",
            trending_model="sonar-reasoning-pro",
            grouping_model="gemini-flash",
            focus="FOCUS",
        )


async def test_refine_record_calls_with_tools_and_thinking_budget(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    links = [f"http://l{i}" for i in range(25)]
    record = NewsRecord(
        title="T", summary="S", links=cast(list[HttpUrl], links)
    )
    llm = fake_llm(chat_results=["REFINED"])

    # Act
    text = await refine_record(
        cast(LlmClient, llm),
        record,
        model="gemini-flash",
        max_links=20,
        today=date(2026, 7, 17),
    )

    # Assert
    assert text == "REFINED"
    _, messages, kwargs = llm.chat_calls[0]
    assert kwargs["tools"] == [{"url_context": {}}]
    assert kwargs["extra_body"] == {"thinkingBudget": -1}
    assert messages[1]["content"] == refinement_user_prompt(
        record.title or "",
        record.summary or "",
        [str(link) for link in record.links[:20]],
    )
    assert messages[0]["content"] == refinement_system_prompt(
        date(2026, 7, 17)
    )


async def test_refine_record_passes_links_unchanged(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    record = NewsRecord(
        title="T",
        summary="S",
        links=cast(
            list[HttpUrl], ["http://public.example", "http://127.0.0.1"]
        ),
    )
    llm = fake_llm(chat_results=["REFINED"])

    # Act
    await refine_record(
        cast(LlmClient, llm),
        record,
        model="gemini-flash",
        max_links=20,
        today=date(2026, 7, 17),
    )

    # Assert
    _, messages, _ = llm.chat_calls[0]
    assert "public.example" in messages[1]["content"]
    assert "127.0.0.1" in messages[1]["content"]


async def test_refine_record_returns_none_on_llm_failure(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    record = NewsRecord(title="T", summary="S", links=[])
    llm = fake_llm(chat_results=[RuntimeError("boom")])

    # Act
    text = await refine_record(
        cast(LlmClient, llm),
        record,
        model="gemini-flash",
        max_links=20,
        today=date(2026, 7, 17),
    )

    # Assert
    assert text is None


async def test_refine_record_returns_none_on_none_content(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    record = NewsRecord(title="T", summary="S", links=[])
    llm = fake_llm(chat_results=[None])

    # Act
    text = await refine_record(
        cast(LlmClient, llm),
        record,
        model="gemini-flash",
        max_links=20,
        today=date(2026, 7, 17),
    )

    # Assert
    assert text is None


async def test_refine_all_preserves_order_and_marks_failures(
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    records = [
        NewsRecord(title="T1", summary="S1", links=[]),
        NewsRecord(title="T2", summary="S2", links=[]),
        NewsRecord(title="T3", summary="S3", links=[]),
    ]
    llm = fake_llm(chat_results=["R1", RuntimeError("boom"), "R3"])

    # Act
    out = await refine_all(
        cast(LlmClient, llm),
        records,
        model="gemini-flash",
        max_links=20,
        today=date(2026, 7, 17),
    )

    # Assert
    assert len(out) == 3
    assert out[0].refined_summary == "R1"
    assert out[1].refined_summary is None
    assert out[2].refined_summary == "R3"
    assert [r.title for r in out] == ["T1", "T2", "T3"]


async def test_write_digest_creates_date_tree(tmp_path: Path) -> None:
    # Arrange
    digest = Digest(generated_at="2026-07-17T12:00:00", records=[])

    # Act
    path = await write_digest(
        digest, tmp_path, date(2026, 7, 17), name="news"
    )

    # Assert
    assert path == tmp_path / "2026" / "07" / "news-17.json"
    assert json.loads(path.read_text()) == {
        "generated_at": "2026-07-17T12:00:00",
        "records": [],
    }


async def test_run_pipeline_happy_path_writes_refined_digest(
    settings_stub: Settings,
    news_agg: Aggregation,
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    raw = {
        "id": 1,
        "title": "T",
        "url": "http://x",
        "content": "C",
        "published_at": "2026-07-16",
        "feed": {"title": "Feed"},
    }
    miniflux = fake_miniflux(entries=[raw])
    llm = fake_llm(
        chat_results=["TRENDS", "REFINED"],
        chat_parsed_results=[
            NewsResponse(
                records=[
                    NewsRecord(
                        title="T",
                        summary="S",
                        links=cast(list[HttpUrl], ["http://a"]),
                    )
                ]
            )
        ],
    )

    # Act
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, miniflux),
        cast(LlmClient, llm),
    )
    path = await service._run_pipeline(news_agg, NOW)

    # Assert
    expected_path = (
        settings_stub.digest_output_dir / "2026" / "07" / "news-17.json"
    )
    assert path == expected_path
    data = json.loads(path.read_text())
    assert data["generated_at"] == NOW.isoformat()
    assert data["records"] == [
        {
            "title": "T",
            "summary": "S",
            "refined_summary": "REFINED",
            "links": ["http://a/"],
        }
    ]


async def test_run_pipeline_partial_refinement_failure_still_writes(
    settings_stub: Settings,
    news_agg: Aggregation,
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    raw = {
        "id": 1,
        "title": "T",
        "url": "http://x",
        "content": "C",
        "published_at": "2026-07-16",
        "feed": {"title": "Feed"},
    }
    miniflux = fake_miniflux(entries=[raw])
    llm = fake_llm(
        chat_results=["TRENDS", "R1", RuntimeError("boom")],
        chat_parsed_results=[
            NewsResponse(
                records=[
                    NewsRecord(title="T1", summary="S1", links=[]),
                    NewsRecord(title="T2", summary="S2", links=[]),
                ]
            )
        ],
    )

    # Act
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, miniflux),
        cast(LlmClient, llm),
    )
    path = await service._run_pipeline(news_agg, NOW)

    # Assert
    data = json.loads(path.read_text())
    assert len(data["records"]) == 2
    assert data["records"][0]["refined_summary"] == "R1"
    assert data["records"][1]["refined_summary"] is None


async def test_run_pipeline_empty_window_writes_empty_digest(
    settings_stub: Settings,
    news_agg: Aggregation,
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    miniflux = fake_miniflux(entries=[])
    llm = fake_llm()

    # Act
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, miniflux),
        cast(LlmClient, llm),
    )
    path = await service._run_pipeline(news_agg, NOW)

    # Assert
    data = json.loads(path.read_text())
    assert data["records"] == []
    assert llm.chat_calls == []


async def test_run_pipeline_fetch_failure_writes_nothing(
    settings_stub: Settings,
    news_agg: Aggregation,
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    miniflux = fake_miniflux(category_error=httpx.ConnectError("down"))
    llm = fake_llm()

    # Act / Assert
    with pytest.raises(httpx.ConnectError):
        service = DigestService(
            settings_stub,
            cast(MinifluxClient, miniflux),
            cast(LlmClient, llm),
        )
        await service._run_pipeline(news_agg, NOW)
    assert list(settings_stub.digest_output_dir.rglob("*.json")) == []


async def test_run_pipeline_grouping_failure_writes_nothing(
    settings_stub: Settings,
    news_agg: Aggregation,
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    raw = {
        "id": 1,
        "title": "T",
        "url": "http://x",
        "content": "C",
        "published_at": "2026-07-16",
        "feed": {"title": "Feed"},
    }
    miniflux = fake_miniflux(entries=[raw])
    llm = fake_llm(chat_results=[RuntimeError("boom")])

    # Act / Assert
    with pytest.raises(RuntimeError):
        service = DigestService(
            settings_stub,
            cast(MinifluxClient, miniflux),
            cast(LlmClient, llm),
        )
        await service._run_pipeline(news_agg, NOW)
    assert list(settings_stub.digest_output_dir.rglob("*.json")) == []


async def test_run_all_aggregations_runs_each_in_sequence(
    settings_stub: Settings,
    three_aggs: tuple[Aggregation, ...],
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    miniflux = fake_miniflux()
    llm = fake_llm()
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, miniflux),
        cast(LlmClient, llm),
    )
    calls: list[tuple[Aggregation, datetime]] = []

    async def run_pipeline(aggregation: Aggregation, now: datetime) -> Path:
        calls.append((aggregation, now))
        return Path(f"/p/{aggregation.name}.json")

    monkeypatch.setattr(service, "_run_pipeline", run_pipeline)
    monkeypatch.setattr("news.digest.service.datetime", FixedDatetime)

    # Act
    paths = await service()

    # Assert
    assert paths == [
        Path("/p/a.json"),
        Path("/p/b.json"),
        Path("/p/c.json"),
    ]
    assert calls == [(aggregation, NOW) for aggregation in three_aggs]


async def test_run_all_aggregations_failed_one_does_not_block_remaining(
    settings_stub: Settings,
    three_aggs: tuple[Aggregation, ...],
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Arrange
    miniflux = fake_miniflux()
    llm = fake_llm()
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, miniflux),
        cast(LlmClient, llm),
    )
    calls: list[tuple[Aggregation, datetime]] = []

    async def run_pipeline(aggregation: Aggregation, now: datetime) -> Path:
        calls.append((aggregation, now))
        if aggregation.name == "b":
            raise RuntimeError("boom")
        return Path(f"/p/{aggregation.name}.json")

    monkeypatch.setattr(service, "_run_pipeline", run_pipeline)
    monkeypatch.setattr("news.digest.service.datetime", FixedDatetime)

    # Act
    paths = await service()

    # Assert
    assert paths == [Path("/p/a.json"), Path("/p/c.json")]
    assert calls == [(aggregation, NOW) for aggregation in three_aggs]
