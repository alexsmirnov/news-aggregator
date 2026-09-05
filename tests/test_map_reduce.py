import math
from typing import cast

import numpy as np
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


def unit_vector(degrees: float) -> list[float]:
    radians = math.radians(degrees)
    return [math.cos(radians), math.sin(radians)]


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


def test_calibrate_threshold_identical_corpus_is_zero() -> None:
    # Arrange
    embeddings = np.tile(np.asarray(unit_vector(30.0)), (10, 1))

    # Act
    threshold = MapReduceGrouping.calibrate_threshold(embeddings)

    # Assert
    assert threshold == 0.0


def test_calibrate_threshold_orthogonal_corpus_is_one() -> None:
    # Arrange
    embeddings = np.asarray([unit_vector(0.0), unit_vector(90.0)])

    # Act
    threshold = MapReduceGrouping.calibrate_threshold(embeddings)

    # Assert: cos(90 degrees) via math.radians is not exactly 0.0.
    assert threshold == pytest.approx(1.0)


def test_calibrate_threshold_larger_k_sigma_tightens_threshold() -> None:
    # Arrange: opposing vectors give a wide, unclamped similarity range.
    embeddings = np.asarray(
        [unit_vector(0.0), unit_vector(90.0), unit_vector(180.0)]
    )

    # Act
    loose = MapReduceGrouping.calibrate_threshold(embeddings, k_sigma=2.0)
    tight = MapReduceGrouping.calibrate_threshold(embeddings, k_sigma=4.0)

    # Assert
    assert tight < loose


def test_calibrate_threshold_clamps_at_zero() -> None:
    # Arrange
    embeddings = np.asarray(
        [unit_vector(0.0), unit_vector(45.0), unit_vector(90.0)]
    )

    # Act
    threshold = MapReduceGrouping.calibrate_threshold(
        embeddings, k_sigma=100.0
    )

    # Assert
    assert threshold == 0.0


def test_calibrate_threshold_single_embedding_is_zero_not_nan() -> None:
    # Arrange
    embeddings = np.asarray([unit_vector(0.0)])

    # Act
    threshold = MapReduceGrouping.calibrate_threshold(embeddings)

    # Assert
    assert threshold == 0.0


def test_cluster_separates_two_tight_blobs() -> None:
    # Arrange
    embeddings = np.asarray(
        [
            unit_vector(0.0),
            unit_vector(1.0),
            unit_vector(2.0),
            unit_vector(90.0),
            unit_vector(91.0),
            unit_vector(92.0),
        ]
    )

    # Act
    labels = MapReduceGrouping.cluster(embeddings, distance_threshold=0.1)

    # Assert
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] == labels[4] == labels[5]
    assert labels[0] != labels[3]


def test_cluster_keeps_singletons_without_noise_label() -> None:
    # Arrange
    embeddings = np.asarray(
        [unit_vector(0.0), unit_vector(1.0), unit_vector(180.0)]
    )

    # Act
    labels = MapReduceGrouping.cluster(embeddings, distance_threshold=0.05)

    # Assert
    assert len(labels) == 3
    assert all(label >= 0 for label in labels)
    assert labels[2] not in (labels[0], labels[1])


def test_cluster_average_linkage_resists_chaining() -> None:
    # Arrange: pairwise cosine distances are ~0.234, 0.234, 0.826 — a
    # single-linkage / connected-components cut at 0.35 would chain all
    # three together via the middle point.
    embeddings = np.asarray(
        [unit_vector(0.0), unit_vector(40.0), unit_vector(80.0)]
    )

    # Act
    labels = MapReduceGrouping.cluster(embeddings, distance_threshold=0.35)

    # Assert
    assert labels[0] != labels[2]


def test_cluster_single_entry_is_its_own_label() -> None:
    # Arrange
    embeddings = np.asarray([unit_vector(0.0)])

    # Act
    labels = MapReduceGrouping.cluster(embeddings, distance_threshold=0.5)

    # Assert
    assert list(labels) == [0]


def test_cluster_empty_input_is_empty_output() -> None:
    # Arrange
    embeddings = np.zeros((0, 2))

    # Act
    labels = MapReduceGrouping.cluster(embeddings, distance_threshold=0.5)

    # Assert
    assert len(labels) == 0


def test_calibrate_and_cluster_recover_synthetic_blobs() -> None:
    # Arrange: three tight, well-separated blobs. The default k_sigma=3.0
    # clamps to 0.0 on this small synthetic corpus (over-tight); k_sigma
    # is exactly the constant the note says must be tuned per corpus, so
    # a working value is recorded here rather than in production code.
    rng = np.random.default_rng(1)
    centers = [unit_vector(0.0), unit_vector(120.0), unit_vector(240.0)]
    blobs = [
        center + rng.normal(scale=0.01, size=2) for center in centers
        for _ in range(5)
    ]
    embeddings = np.asarray(blobs)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Act
    threshold = MapReduceGrouping.calibrate_threshold(
        embeddings, k_sigma=1.5, seed=1
    )
    labels = MapReduceGrouping.cluster(embeddings, threshold)

    # Assert: each blob's 5 members share a label, and the three blobs
    # are not merged into one.
    blob_labels = [set(labels[i : i + 5]) for i in range(0, 15, 5)]
    assert all(len(labels_set) == 1 for labels_set in blob_labels)
    assert len({next(iter(s)) for s in blob_labels}) == 3
