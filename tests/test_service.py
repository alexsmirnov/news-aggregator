import json
import types
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from pydantic import HttpUrl

from news.settings import Aggregation
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
from news.digest.service import DigestService, PipelineError
from news.settings import Settings

NOW = datetime(2026, 7, 17, 12, 0, 0)


class FixedDatetime(datetime):
    @classmethod
    def now(cls, tz: object | None = None) -> datetime:
        return NOW


class FakeMiniflux:
    def __init__(
        self,
        entries: list[RssEntry] | None = None,
        category_error: Exception | None = None,
        entries_error: Exception | None = None,
    ) -> None:
        self.entries = entries if entries is not None else []
        self.category_error = category_error
        self.entries_error = entries_error
        self.calls: list[dict[str, Any]] = []

    async def get_entries(
        self, category_name: str, **kwargs: Any
    ) -> list[RssEntry]:
        self.calls.append({"category_name": category_name, **kwargs})
        if self.category_error is not None:
            raise self.category_error
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
            grouping_content_max_chars=300,
            refine_max_links=10,
            model_trending="sonar-reasoning-pro",
            model_grouping="gemini-flash",
            model_refinement="gemini-flash",
        ),
    )


@pytest.fixture
def news_agg() -> Aggregation:
    return Aggregation(name="news", miniflux_category="news", focus="FOCUS")


@pytest.fixture
def three_aggs(settings_stub: Settings) -> tuple[Aggregation, ...]:
    aggs = (
        Aggregation(name="a", miniflux_category="news", focus="fa"),
        Aggregation(name="b", miniflux_category="tech", focus="fb"),
        Aggregation(name="c", miniflux_category="economy", focus="fc"),
    )
    settings_stub.aggregations = aggs  # type: ignore[attr-defined]
    return aggs


def test_strip_html_extracts_text() -> None:
    # Act
    text = DigestService.strip_html("<p>Hello <b>World</b></p>")

    # Assert
    assert text == "Hello World"


def test_format_entry_block(entry: RssEntry) -> None:
    # Act
    text = DigestService.format_entry(42, entry, content_max_chars=1000)

    # Assert
    assert text == (
        "# Entity 42\nTitle: T\nContent: C\nSource: F\nLink: http://a\n"
    )


def test_format_entry_truncates_content_without_mutating_entry(
    entry: RssEntry,
) -> None:
    # Arrange
    entry.content = "abcdef"

    # Act
    text = DigestService.format_entry(42, entry, content_max_chars=3)

    # Assert
    assert "Content: abc\n" in text
    assert entry.content == "abcdef"


def test_format_full_entry_block(entry: RssEntry) -> None:
    # Act
    text = DigestService.format_full_entry(entry)

    # Assert
    assert text == "Title: T\nContent: C\nLink: http://a\n"


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
    text = DigestService.format_entries([e1, e2], content_max_chars=1000)

    # Assert
    assert text == DigestService.format_entry(
        1, e1, content_max_chars=1000
    ) + "\n" + DigestService.format_entry(2, e2, content_max_chars=1000)


async def test_fetch_entries_maps_and_truncates(
    settings_stub: Settings,
    fake_miniflux: type[FakeMiniflux],
) -> None:
    # Arrange
    raw = RssEntry(
        id=9,
        title="T",
        link="http://x",
        content="<p>" + "a" * 1500 + "</p>",
        published_at="2026-07-16",
        source="Feed",
    )
    client = fake_miniflux(entries=[raw])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, client),
        cast(LlmClient, object()),
    )

    # Act
    entries = await service.fetch_entries(
        category="news",
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
            "category_name": "news",
            "published_after": int(
                (NOW - timedelta(hours=24)).timestamp()
            ),
            "published_before": int(NOW.timestamp()),
            "order": "published_at",
            "limit": 10000,
        }
    ]


async def test_extract_groups_calls_trending_then_grouping(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    response = NewsResponse(records=[NewsRecord(title="T", links=[])])
    llm = fake_llm(chat_results=["TRENDS"], chat_parsed_results=[response])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act
    records = await service.extract_groups(
        "FORMATTED",
        focus="FOCUS",
    )

    # Assert
    assert records == response.records
    assert llm.chat_calls[0][0] == settings_stub.model_trending
    model, messages, response_format, _ = llm.chat_parsed_calls[0]
    assert model == settings_stub.model_grouping
    assert response_format is NewsResponse
    assert "TRENDS" in messages[0]["content"]
    assert "FOCUS" in messages[0]["content"]
    assert messages[1]["content"] == grouping_user_prompt("FORMATTED")


async def test_extract_groups_raises_on_none_trending(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    llm = fake_llm(chat_results=[None])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act / Assert
    with pytest.raises(PipelineError):
        await service.extract_groups(
            "FORMATTED",
            focus="FOCUS",
        )
    assert llm.chat_parsed_calls == []


async def test_extract_groups_raises_on_none_grouping(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    llm = fake_llm(chat_results=["TRENDS"], chat_parsed_results=[None])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act / Assert
    with pytest.raises(PipelineError):
        await service.extract_groups(
            "FORMATTED",
            focus="FOCUS",
        )


async def test_refine_record_combines_full_content_and_limits_fetch_links(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    settings_stub.refine_max_links = 2  # type: ignore[attr-defined]
    entries = [
        RssEntry(
            id=i,
            title=f"T{i}",
            content=content,
            link=f"http://l{i}",
            published_at="2026-07-16",
            source="F",
        )
        for i, content in enumerate(["aaaaa", "a", "a" * 10, "aaa"])
    ]
    entries_by_link = {e.link: e for e in entries}
    record = NewsRecord(
        title="T", links=cast(list[HttpUrl], [e.link for e in entries])
    )
    llm = fake_llm(chat_results=["REFINED"])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act
    text = await service.refine_record(
        record,
        entries_by_link,
        today=date(2026, 7, 17),
    )

    # Assert
    assert text == "REFINED"
    _, messages, kwargs = llm.chat_calls[0]
    assert kwargs["tools"] == [{"url_context": {}}]
    assert kwargs["extra_body"] == {"thinkingBudget": -1}
    full_content = "\n".join(
        DigestService.format_full_entry(e) for e in entries
    )
    assert messages[1]["content"] == refinement_user_prompt(
        "T", full_content, ["http://l1", "http://l3"]
    )
    assert messages[0]["content"] == refinement_system_prompt(
        date(2026, 7, 17)
    )


async def test_refine_record_includes_link_in_full_content_unchanged(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    matched_entry = RssEntry(
        id=1,
        title="T",
        content="C",
        link="http://127.0.0.1",
        published_at="2026-07-16",
        source="F",
    )
    entries_by_link = {matched_entry.link: matched_entry}
    record = NewsRecord(
        title="T", links=cast(list[HttpUrl], [matched_entry.link])
    )
    llm = fake_llm(chat_results=["REFINED"])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act
    await service.refine_record(
        record,
        entries_by_link,
        today=date(2026, 7, 17),
    )

    # Assert
    _, messages, _ = llm.chat_calls[0]
    assert "127.0.0.1" in messages[1]["content"]


async def test_refine_record_returns_none_on_llm_failure(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    record = NewsRecord(title="T", links=[])
    llm = fake_llm(chat_results=[RuntimeError("boom")])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act
    text = await service.refine_record(
        record,
        {},
        today=date(2026, 7, 17),
    )

    # Assert
    assert text is None


async def test_refine_record_returns_none_on_none_content(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    record = NewsRecord(title="T", links=[])
    llm = fake_llm(chat_results=[None])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act
    text = await service.refine_record(
        record,
        {},
        today=date(2026, 7, 17),
    )

    # Assert
    assert text is None


async def test_refine_all_preserves_order_and_marks_failures(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    records = [
        NewsRecord(title="T1", links=[]),
        NewsRecord(title="T2", links=[]),
        NewsRecord(title="T3", links=[]),
    ]
    llm = fake_llm(chat_results=["R1", RuntimeError("boom"), "R3"])
    service = DigestService(
        settings_stub,
        cast(MinifluxClient, object()),
        cast(LlmClient, llm),
    )

    # Act
    out = await service.refine_all(
        records,
        [],
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
    path = await DigestService.write_digest(
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
    raw = RssEntry(
        id=1,
        title="T",
        link="http://x",
        content="C",
        published_at="2026-07-16",
        source="Feed",
    )
    miniflux = fake_miniflux(entries=[raw])
    llm = fake_llm(
        chat_results=["TRENDS", "REFINED"],
        chat_parsed_results=[
            NewsResponse(
                records=[
                    NewsRecord(
                        title="T",
                        links=cast(list[HttpUrl], ["http://x"]),
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
            "refined_summary": "REFINED",
            "links": ["http://x/"],
        }
    ]


async def test_run_pipeline_partial_refinement_failure_still_writes(
    settings_stub: Settings,
    news_agg: Aggregation,
    fake_miniflux: type[FakeMiniflux],
    fake_llm: type[FakeLlm],
) -> None:
    # Arrange
    raw = RssEntry(
        id=1,
        title="T",
        link="http://x",
        content="C",
        published_at="2026-07-16",
        source="Feed",
    )
    miniflux = fake_miniflux(entries=[raw])
    llm = fake_llm(
        chat_results=["TRENDS", "R1", RuntimeError("boom")],
        chat_parsed_results=[
            NewsResponse(
                records=[
                    NewsRecord(title="T1", links=[]),
                    NewsRecord(title="T2", links=[]),
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
    raw = RssEntry(
        id=1,
        title="T",
        link="http://x",
        content="C",
        published_at="2026-07-16",
        source="Feed",
    )
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
