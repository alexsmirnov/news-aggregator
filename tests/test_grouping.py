from typing import cast

import pytest
from tests.conftest import FakeLlm

from news.digest.grouping import (
    LlmGrouping,
    PipelineError,
    format_entries,
    format_entry,
)
from news.digest.llm_client import LlmClient
from news.digest.prompts import grouping_user_prompt
from news.digest.schemas import NewsRecord, NewsResponse, RssEntry
from news.settings import Settings


def test_format_entry_block(entry: RssEntry) -> None:
    # Act
    text = format_entry(42, entry, content_max_chars=1000)

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
    text = format_entry(42, entry, content_max_chars=3)

    # Assert
    assert "Content: abc\n" in text
    assert entry.content == "abcdef"


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
    text = format_entries([e1, e2], content_max_chars=1000)

    # Assert
    assert text == format_entry(
        1, e1, content_max_chars=1000
    ) + "\n" + format_entry(2, e2, content_max_chars=1000)


async def test_llm_grouping_calls_grouping(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
    entry: RssEntry,
) -> None:
    # Arrange
    response = NewsResponse(records=[NewsRecord(title="T", links=[])])
    llm = fake_llm(chat_parsed_results=[response])
    grouping = LlmGrouping(settings_stub, cast(LlmClient, llm))

    # Act
    records = await grouping([entry], focus="FOCUS")

    # Assert
    assert records == response.records
    model, messages, response_format, _ = llm.chat_parsed_calls[0]
    assert model == settings_stub.model_grouping
    assert response_format is NewsResponse
    assert "FOCUS" in messages[0]["content"]
    assert messages[1]["content"] == grouping_user_prompt(
        format_entries(
            [entry],
            content_max_chars=settings_stub.grouping_content_max_chars,
        )
    )


async def test_llm_grouping_raises_on_none_response(
    settings_stub: Settings,
    fake_llm: type[FakeLlm],
    entry: RssEntry,
) -> None:
    # Arrange
    llm = fake_llm(chat_parsed_results=[None])
    grouping = LlmGrouping(settings_stub, cast(LlmClient, llm))

    # Act / Assert
    with pytest.raises(PipelineError):
        await grouping([entry], focus="FOCUS")
