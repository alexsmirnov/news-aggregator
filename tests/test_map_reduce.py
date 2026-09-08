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
    entry_id: int, *, title: str = "T", content: str = "C",
    link: str | None = None,
    published_at: str = "2026-07-16T10:00:00",
    source: str = "F",
) -> RssEntry:
    return RssEntry(
        id=entry_id,
        title=title,
        content=content,
        link=link if link is not None else f"http://a/{entry_id}",
        published_at=published_at,
        source=source,
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


def test_scoring_singleton_has_positive_score() -> None:
    # Arrange
    records = [make_entry(1)]

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([0]), np.array([[1.0, 0.0]])
    )

    # Assert
    assert scored == [{
        "records": records, "size": 1, "unique_domains": 1,
        "source_entropy": 0.0, "effective_sources": 1.0,
        "burst_z": 0.0, "novelty": 1.0, "trend_score": 0.520,
    }]


def test_scoring_diverse_group_ranks_above_same_host_group() -> None:
    # Arrange
    records = [
        make_entry(1, link="https://A.example:443/1", source="Feed A"),
        make_entry(2, link="http://a.example/2", source="Feed B"),
        make_entry(3, link="https://a.example/3"),
        make_entry(4, link="https://b.example/4"),
    ]

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([9, 9, 3, 3]), np.tile([1.0, 0.0], (4, 1))
    )

    # Assert
    diverse, same_host = scored
    assert diverse["records"] == records[2:]
    assert same_host["records"] == records[:2]
    assert diverse["size"] == same_host["size"] == 2
    assert diverse["unique_domains"] == 2
    assert diverse["source_entropy"] == .693
    assert diverse["effective_sources"] == 2.0
    assert diverse["burst_z"] == 0.0
    assert diverse["trend_score"] == 2.472
    assert same_host["unique_domains"] == 1
    assert same_host["source_entropy"] == 0.0
    assert same_host["trend_score"] == .520


def test_scoring_skewed_host_shares_use_shannon_entropy() -> None:
    # Arrange
    records = [make_entry(i) for i in range(3)] + [
        make_entry(3, link="https://b/3")
    ]

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.zeros(4, dtype=np.int64), np.tile([1.0, 0.0], (4, 1))
    )

    # Assert
    assert scored[0]["source_entropy"] == .562
    assert scored[0]["effective_sources"] == 1.75


@pytest.mark.parametrize("aware", [False, True])
def test_scoring_arrival_span_uses_six_hour_floor(aware: bool) -> None:
    # Arrange
    stamps = ["00:00:00", "00:00:00", "03:00:00", "00:00:00",
              "12:00:00", "00:00:00"]
    records = [
        make_entry(i, published_at=f"2026-07-16T{stamp}{'Z' if aware else ''}")
        for i, stamp in enumerate(stamps)
    ]
    if aware:
        records[2].published_at = "2026-07-16T05:00:00+02:00"

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([0, 0, 1, 1, 2, 2]), np.tile([1.0, 0.0], (6, 1))
    )

    # Assert
    assert [group["burst_z"] for group in scored] == [.71, .71, -1.41]


def test_scoring_baseline_uses_normalized_centroid_nearest_match() -> None:
    # Arrange
    records = [make_entry(1), make_entry(2)]
    embeddings = np.array([[1.0, 0.0], [0.0, 1.0]])
    baseline = np.array([[1.0, 0.0], [-1.0, 0.0]])

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([0, 0]), embeddings, baseline
    )

    # Assert
    assert scored[0]["novelty"] == .293


@pytest.mark.parametrize(
    ("baseline", "novelty", "trend"),
    [
        (None, 1.0, .520),
        (np.empty((0, 2)), 1.0, .520),
        (np.array([[1.0, 0.0]]), 0.0, .173),
        (np.array([[0.0, 1.0]]), 1.0, .520),
        (np.array([[-1.0, 0.0]]), 2.0, .866),
    ],
)
def test_scoring_novelty_baseline_boundaries(
    baseline: np.ndarray | None, novelty: float, trend: float,
) -> None:
    # Arrange
    records = [make_entry(1)]

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([0]), np.array([[1.0, 0.0]]), baseline
    )

    # Assert
    assert scored[0]["novelty"] == novelty
    assert scored[0]["trend_score"] == trend


def test_scoring_zero_centroid_is_finite() -> None:
    # Arrange
    records = [make_entry(1), make_entry(2)]

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([0, 0]), np.array([[1.0, 0.0], [-1.0, 0.0]]),
        np.array([[1.0, 0.0]]),
    )

    # Assert
    assert scored[0]["novelty"] == 1.0
    assert scored[0]["trend_score"] == .520


def test_scoring_ties_keep_group_encounter_order() -> None:
    # Arrange
    records = [make_entry(i) for i in range(3)]

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([9, 2, 5]), np.tile([1.0, 0.0], (3, 1))
    )

    # Assert
    assert [group["records"] for group in scored] == [[r] for r in records]


def test_scoring_empty_input_returns_no_groups() -> None:
    # Arrange
    labels = np.array([], dtype=np.int64)
    embeddings = np.empty((0, 2))

    # Act
    scored = MapReduceGrouping.score_clusters([], labels, embeddings)

    # Assert
    assert scored == []


@pytest.mark.parametrize(("label_count", "vector_count"), [(1, 2), (2, 1)])
def test_scoring_misaligned_inputs_raise(
    label_count: int, vector_count: int,
) -> None:
    # Arrange
    records = [make_entry(1), make_entry(2)]

    # Act / Assert
    with pytest.raises(ValueError):
        MapReduceGrouping.score_clusters(
            records, np.zeros(label_count, dtype=np.int64),
            np.tile([1.0, 0.0], (vector_count, 1)),
        )


def test_scoring_incompatible_baseline_width_raises() -> None:
    # Arrange
    records = [make_entry(1)]

    # Act / Assert
    with pytest.raises(ValueError):
        MapReduceGrouping.score_clusters(
            records, np.array([0]), np.array([[1.0, 0.0]]),
            np.array([[1.0, 0.0, 0.0]]),
        )


def test_scoring_missing_hostnames_share_unknown_source() -> None:
    # Arrange
    records = [
        make_entry(1, link="/one", source="A"),
        make_entry(2, link="two", source="B"),
    ]

    # Act
    scored = MapReduceGrouping.score_clusters(
        records, np.array([0, 0]), np.tile([1.0, 0.0], (2, 1))
    )

    # Assert
    assert scored[0]["unique_domains"] == 1
    assert scored[0]["source_entropy"] == 0.0
