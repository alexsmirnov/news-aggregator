import math
from typing import cast

import pytest
from tests.conftest import FakeLlm

from news.digest.llm_client import LlmClient
from news.digest.map_reduce import (
    EMBED_BATCH_SIZE,
    EMBED_MAX_INPUT_TOKENS,
    MapReduceGrouping,
)
from news.digest.schemas import RssEntry
from news.settings import Settings


def make_entry(
    entry_id: int, *, title: str = "T", content: str = "C"
) -> RssEntry:
    return RssEntry(
        id=entry_id,
        title=title,
        content=content,
        link=f"http://a/{entry_id}",
        published_at="2026-07-16T10:00:00",
        source="F",
    )


def make_grouping(
    settings_stub: Settings, llm: FakeLlm
) -> MapReduceGrouping:
    return MapReduceGrouping(settings_stub, cast(LlmClient, llm))


async def test_embed_entries_binds_title_and_content_vectors(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [
        make_entry(1, title="T1", content="C1"),
        make_entry(2, title="T2", content="C2"),
    ]
    title_vectors = [[1.0], [2.0]]
    content_vectors = [[10.0], [20.0]]
    llm = fake_llm(
        embeddings_results=[title_vectors + content_vectors]
    )
    grouping = make_grouping(settings_stub, llm)

    # Act
    embedded = await grouping.embed_entries(entries)

    # Assert
    assert [e.entry for e in embedded] == entries
    assert [e.title_vector for e in embedded] == [[1.0], [1.0]]
    assert [e.content_vector for e in embedded] == [[1.0], [1.0]]


async def test_embed_entries_normalizes_title_and_content_separately(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [make_entry(1), make_entry(2)]
    title_vectors = [[3.0, 4.0], [1.0, 0.0]]
    content_vectors = [[0.0, 5.0], [6.0, 8.0]]
    llm = fake_llm(embeddings_results=[title_vectors + content_vectors])
    grouping = make_grouping(settings_stub, llm)

    # Act
    embedded = await grouping.embed_entries(entries)

    # Assert
    assert [e.title_vector for e in embedded] == [[0.6, 0.8], [1.0, 0.0]]
    assert [e.content_vector for e in embedded] == [[0.0, 1.0], [0.6, 0.8]]


async def test_embed_entries_leaves_zero_vector_unchanged(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [make_entry(1)]
    llm = fake_llm(embeddings_results=[[[0.0, 0.0], [1.0, 0.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    embedded = await grouping.embed_entries(entries)

    # Assert
    assert embedded[0].title_vector == [0.0, 0.0]


async def test_embed_entries_batches_by_size_and_stays_under_limit(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entry_count = EMBED_BATCH_SIZE + 1
    entries = [make_entry(i) for i in range(entry_count)]
    total_texts = 2 * entry_count
    expected_batches = math.ceil(total_texts / EMBED_BATCH_SIZE)
    llm = fake_llm(
        embeddings_results=[
            [[float(i)] for i in range(len(batch))]
            for batch in (
                range(min(EMBED_BATCH_SIZE, total_texts - start))
                for start in range(0, total_texts, EMBED_BATCH_SIZE)
            )
        ]
    )
    grouping = make_grouping(settings_stub, llm)

    # Act
    await grouping.embed_entries(entries)

    # Assert
    assert len(llm.embeddings_calls) == expected_batches
    assert all(
        len(inputs) <= EMBED_BATCH_SIZE
        for _, inputs, _ in llm.embeddings_calls
    )


async def test_embed_entries_truncates_long_content(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    long_content = "x" * (EMBED_MAX_INPUT_TOKENS * 10)
    entries = [make_entry(1, content=long_content)]
    llm = fake_llm(embeddings_results=[[[1.0], [2.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    await grouping.embed_entries(entries)

    # Assert
    _, inputs, _ = llm.embeddings_calls[0]
    assert len(inputs[1]) == EMBED_MAX_INPUT_TOKENS * 4


async def test_embed_entries_falls_back_to_title_for_empty_content(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [make_entry(1, title="Only title", content="   ")]
    llm = fake_llm(embeddings_results=[[[1.0], [2.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    await grouping.embed_entries(entries)

    # Assert
    _, inputs, _ = llm.embeddings_calls[0]
    assert inputs[1] == "Only title"


async def test_embed_entries_falls_back_to_sentinel_when_both_blank(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [make_entry(1, title="   ", content="   ")]
    llm = fake_llm(embeddings_results=[[[1.0], [2.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    await grouping.embed_entries(entries)

    # Assert
    _, inputs, _ = llm.embeddings_calls[0]
    assert inputs == ["(empty)", "(empty)"]


async def test_call_raises_not_implemented(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    grouping = make_grouping(settings_stub, fake_llm())

    # Act / Assert
    with pytest.raises(NotImplementedError):
        await grouping([make_entry(1)], focus="FOCUS")
