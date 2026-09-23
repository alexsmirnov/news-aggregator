import math
import types
from datetime import timedelta
from typing import cast

import numpy as np
import pytest
from tests.conftest import FakeLlm

from news.digest.llm_client import LlmClient
from news.digest.map_reduce import (
    EMBED_BATCH_SIZE,
    EMBED_MAX_INPUT_TOKENS,
    Candidate,
    MapReduceGrouping,
    RateBaseline,
    ScoredCluster,
)
from news.digest.schemas import ClusterSummary, RssEntry
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


def make_settings(settings_stub: Settings, **overrides: object) -> Settings:
    return cast(
        Settings,
        types.SimpleNamespace(**{**vars(settings_stub), **overrides}),
    )


def make_scored_cluster(
    records: list[RssEntry], *, trend_score: float = 1.0
) -> ScoredCluster:
    return ScoredCluster(
        records=records,
        indices=list(range(len(records))),
        size=len(records),
        unique_domains=1,
        source_entropy=0.0,
        effective_sources=1.0,
        burst_z=0.0,
        novelty=1.0,
        trend_score=trend_score,
    )


def make_candidate(
    entry: RssEntry,
    *,
    title: str | None = None,
    summary: str = "",
    entities: tuple[str, ...] = (),
    index: int = 0,
    trend_score: float = 1.0,
) -> Candidate:
    return Candidate(
        title=title or entry.title,
        summary=summary,
        entities=entities,
        entries=(entry,),
        indices=(index,),
        trend_score=trend_score,
    )


def summary(title: str, text: str = "s") -> ClusterSummary:
    return ClusterSummary(title=title, summary=text, entities=[])


async def test_embed_entries_returns_normalized_matrix_in_entry_order(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [
        make_entry(1, title="T1", content="C1"),
        make_entry(2, title="T2", content="C2"),
    ]
    llm = fake_llm(embeddings_results=[[[3.0, 4.0], [1.0, 0.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    vectors = await grouping.embed_entries(entries)

    # Assert
    assert vectors.dtype == np.float64
    np.testing.assert_allclose(vectors, [[0.6, 0.8], [1.0, 0.0]])


async def test_embed_entries_leaves_zero_vector_unchanged(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [make_entry(1)]
    llm = fake_llm(embeddings_results=[[[0.0, 0.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    vectors = await grouping.embed_entries(entries)

    # Assert
    np.testing.assert_allclose(vectors, [[0.0, 0.0]])


async def test_embed_entries_empty_input_returns_empty_matrix_without_llm_call(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    llm = fake_llm()
    grouping = make_grouping(settings_stub, llm)

    # Act
    vectors = await grouping.embed_entries([])

    # Assert
    assert vectors.shape == (0, settings_stub.embedding_dimensions)
    assert llm.embeddings_calls == []


async def test_embed_entries_batches_by_size_and_stays_under_limit(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entry_count = EMBED_BATCH_SIZE + 1
    entries = [make_entry(i) for i in range(entry_count)]
    expected_batches = math.ceil(entry_count / EMBED_BATCH_SIZE)
    llm = fake_llm(
        embeddings_results=[
            [[1.0] for _ in range(min(EMBED_BATCH_SIZE, entry_count - start))]
            for start in range(0, entry_count, EMBED_BATCH_SIZE)
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
    llm = fake_llm(embeddings_results=[[[1.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    await grouping.embed_entries(entries)

    # Assert
    _, inputs, _ = llm.embeddings_calls[0]
    assert len(inputs[0]) == EMBED_MAX_INPUT_TOKENS * 4


async def test_embed_entries_falls_back_to_title_for_empty_content(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [make_entry(1, title="Only title", content="   ")]
    llm = fake_llm(embeddings_results=[[[1.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    await grouping.embed_entries(entries)

    # Assert
    _, inputs, _ = llm.embeddings_calls[0]
    assert inputs[0] == "Only title"


async def test_embed_entries_falls_back_to_sentinel_when_both_blank(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [make_entry(1, title="   ", content="   ")]
    llm = fake_llm(embeddings_results=[[[1.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    await grouping.embed_entries(entries)

    # Assert
    _, inputs, _ = llm.embeddings_calls[0]
    assert inputs == ["(empty)"]


async def test_call_clusters_embeds_and_titles_each_group(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange: two identical-vector pairs cluster together at the
    # calibrated threshold; two orthogonal pairs stay apart (see plan
    # note: threshold clamps to 0.0 on this corpus, so only exact-distance
    # -0 duplicates merge).
    entries = [
        make_entry(1, link="https://a.example/1"),
        make_entry(2, link="https://a.example/2"),
        make_entry(3, link="https://b.example/3"),
        make_entry(4, link="https://b.example/4"),
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    llm = fake_llm(
        embeddings_results=[vectors, [[1.0, 0.0], [0.0, 1.0]]],
        chat_parsed_results=[summary("Group 1"), summary("Group 2")],
    )
    # ponytail: k_sigma=3.0 (the settings_stub default) clamps the
    # threshold to 0.0 on this tiny corpus, and 0.0 rejects even exact
    # duplicates because their cosine distance is float noise (~1e-16),
    # not exactly 0. k_sigma=1.0 keeps the threshold a wide, comfortably
    # positive margin between the duplicate pairs (~0) and the orthogonal
    # pairs (1.0).
    # grouping_window_hours=0: all entries share one published_at, so a
    # non-zero window would exclude every entry from the clustering slice.
    settings = make_settings(
        settings_stub, grouping_k_sigma=1.0, grouping_window_hours=0
    )
    grouping = make_grouping(settings, llm)

    # Act
    result = await grouping(entries, focus="FOCUS")

    # Assert
    assert len(result) == 2
    assert {link for record in result for link in record.links} == {
        entry.link for entry in entries
    }
    assert all(len(record.links) == 2 for record in result)
    assert {record.title for record in result} == {"Group 1", "Group 2"}


async def test_call_limits_map_calls_to_grouping_map_clusters(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange: group B has higher source diversity so it always outranks
    # group A regardless of clustering label order (deterministic top-1).
    entries = [
        make_entry(1, link="https://a.example/1", source="A"),
        make_entry(2, link="https://a.example/2", source="A"),
        make_entry(3, link="https://b.example/3", source="B"),
        make_entry(4, link="https://c.example/4", source="C"),
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    llm = fake_llm(
        embeddings_results=[vectors, [[1.0, 0.0]]],
        chat_parsed_results=[summary("Top")],
    )
    # grouping_window_hours=0: all entries share one published_at, so a
    # non-zero window would exclude every entry from the clustering slice.
    settings = make_settings(
        settings_stub,
        grouping_map_clusters=1,
        grouping_k_sigma=1.0,
        grouping_window_hours=0,
    )
    grouping = make_grouping(settings, llm)

    # Act
    result = await grouping(entries, focus="FOCUS")

    # Assert
    assert len(llm.chat_parsed_calls) == 1
    assert len(result) == 1
    assert set(result[0].links) == {entries[2].link, entries[3].link}


async def test_call_excludes_baseline_only_entries_from_clustering(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange: an entry from 12h ago is outside the 4h clustering window
    # (settings_stub.grouping_window_hours) but still inside the baseline
    # window, so it must be embedded (for baseline centroid comparisons)
    # yet never appear in a returned NewsRecord's links.
    old_entry = make_entry(
        1, link="https://old.example/1",
        published_at="2026-07-16T00:00:00",
    )
    recent_a = make_entry(
        2, link="https://a.example/2",
        published_at="2026-07-16T06:00:00",
    )
    recent_b = make_entry(
        3, link="https://a.example/3",
        published_at="2026-07-16T06:00:00",
    )
    newest = make_entry(
        4, link="https://b.example/4",
        published_at="2026-07-16T12:00:00",
    )
    entries = [old_entry, recent_a, recent_b, newest]
    # Vectors are given in published_at-ascending order (sort_by_published_at
    # is stable, and these four are already sorted by construction).
    vectors = [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    llm = fake_llm(
        embeddings_results=[vectors, [[1.0, 0.0], [0.0, 1.0]]],
        chat_parsed_results=[summary("Recent"), summary("Newest")],
    )
    settings = make_settings(settings_stub, grouping_k_sigma=1.0)
    grouping = make_grouping(settings, llm)

    # Act
    result = await grouping(entries, focus="FOCUS")

    # Assert
    all_links = {link for record in result for link in record.links}
    assert old_entry.link not in all_links
    assert all_links == {recent_a.link, recent_b.link, newest.link}


async def test_map_clusters_falls_back_to_entry_title_on_failure(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entry = make_entry(1, title="Fallback Title")
    cluster = make_scored_cluster([entry], trend_score=3.5)
    llm = fake_llm(chat_parsed_results=[RuntimeError("boom")])
    grouping = make_grouping(settings_stub, llm)

    # Act
    candidates = await grouping._map_clusters([cluster], focus="FOCUS")

    # Assert
    assert candidates == [Candidate(
        title="Fallback Title", summary="", entities=(), entries=(entry,),
        indices=(0,), trend_score=3.5,
    )]


def test_to_news_records_orders_by_trend_score_and_truncates() -> None:
    # Arrange
    low = make_candidate(make_entry(1, link="https://a"), trend_score=1.0)
    high = make_candidate(make_entry(2, link="https://b"), trend_score=3.0)
    mid = make_candidate(make_entry(3, link="https://c"), trend_score=2.0)

    # Act
    records = MapReduceGrouping._to_news_records(
        [low, high, mid], max_records=2
    )

    # Assert
    assert [record.title for record in records] == [high.title, mid.title]


def test_sort_by_published_at_orders_ascending_and_is_stable() -> None:
    # Arrange
    first = make_entry(1, published_at="2026-07-16T09:00:00")
    tied_a = make_entry(2, published_at="2026-07-16T10:00:00")
    tied_b = make_entry(3, published_at="2026-07-16T10:00:00")
    last = make_entry(4, published_at="2026-07-16T11:00:00")
    entries = [last, tied_a, tied_b, first]

    # Act
    sorted_entries = MapReduceGrouping.sort_by_published_at(entries)

    # Assert
    assert sorted_entries == [first, tied_a, tied_b, last]


def test_split_offsets_bisects_baseline_and_clustering_windows() -> None:
    # Arrange: hourly entries 00:00..05:00, window=2h. Baseline keeps
    # everything up to (last - 2h) = 03:00 inclusive -> indexes 0..3.
    # Clustering keeps everything from (first + 2h) = 02:00 inclusive
    # -> indexes 2..5.
    entries = [
        make_entry(i, published_at=f"2026-07-16T0{i}:00:00")
        for i in range(6)
    ]

    # Act
    baseline_cut, clustering_cut = MapReduceGrouping.split_offsets(
        entries, timedelta(hours=2)
    )

    # Assert
    assert baseline_cut == 4
    assert clustering_cut == 2


def test_split_offsets_zero_window_keeps_full_range_both_sides() -> None:
    # Arrange
    entries = [make_entry(i) for i in range(3)]

    # Act
    baseline_cut, clustering_cut = MapReduceGrouping.split_offsets(
        entries, timedelta(hours=0)
    )

    # Assert
    assert (baseline_cut, clustering_cut) == (3, 0)


def test_split_offsets_empty_input_returns_zero_offsets() -> None:
    # Act
    baseline_cut, clustering_cut = MapReduceGrouping.split_offsets(
        [], timedelta(hours=4)
    )

    # Assert
    assert (baseline_cut, clustering_cut) == (0, 0)


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
        "records": records, "indices": [0], "size": 1, "unique_domains": 1,
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


def test_compute_metrics_rescores_pooled_fragments_above_each_part() -> None:
    # Arrange: two single-host fragments of one event from different hosts;
    # pooled, they gain source diversity neither fragment has alone.
    entry_a = make_entry(1, link="https://a.example/1")
    entry_b = make_entry(2, link="https://b.example/2")
    vector = np.array([[1.0, 0.0]])
    rate = RateBaseline(mean=0.0, std=1.0)

    # Act
    part_a = MapReduceGrouping.compute_metrics([entry_a], vector, rate)
    part_b = MapReduceGrouping.compute_metrics([entry_b], vector, rate)
    pooled = MapReduceGrouping.compute_metrics(
        [entry_a, entry_b], np.tile([1.0, 0.0], (2, 1)), rate
    )

    # Assert
    assert pooled["effective_sources"] == 2.0
    assert pooled["trend_score"] > max(
        part_a["trend_score"], part_b["trend_score"]
    )


def test_rate_baseline_is_mean_and_floored_std_of_group_rates() -> None:
    # Arrange: two equal-size groups at one timestamp have equal rates.
    records = [make_entry(i) for i in range(4)]

    # Act
    rate = MapReduceGrouping.rate_baseline(records, np.array([0, 0, 1, 1]))

    # Assert
    assert rate == RateBaseline(mean=2 / 6, std=1e-6)


def test_merge_candidates_leads_with_longest_summary_and_unions() -> None:
    # Arrange
    entry_a = make_entry(1, link="https://a.example/1")
    entry_b = make_entry(2, link="https://b.example/2")
    short = make_candidate(
        entry_a, title="Short", summary="s", entities=("X",), index=0,
        trend_score=9.0,
    )
    rich = make_candidate(
        entry_b, title="Rich", summary="longer", entities=("Y", "X"), index=1,
    )
    vectors = np.tile([1.0, 0.0], (2, 1))
    rate = RateBaseline(mean=0.0, std=1.0)

    # Act
    merged = MapReduceGrouping._merge_candidates([short, rich], vectors, rate)

    # Assert
    assert merged == Candidate(
        title="Rich",
        summary="longer",
        entities=("X", "Y"),
        entries=(entry_a, entry_b),
        indices=(0, 1),
        trend_score=MapReduceGrouping.compute_metrics(
            [entry_a, entry_b], vectors, rate
        )["trend_score"],
    )


def test_merge_candidates_singleton_keeps_content() -> None:
    # Arrange
    candidate = make_candidate(
        make_entry(1), title="Only", summary="s", entities=("X",),
    )

    # Act
    merged = MapReduceGrouping._merge_candidates(
        [candidate], np.array([[1.0, 0.0]]), RateBaseline(0.0, 1.0)
    )

    # Assert
    assert (merged.title, merged.summary, merged.entities) == (
        "Only", "s", ("X",)
    )
    assert (merged.entries, merged.indices) == (
        candidate.entries, candidate.indices
    )


async def test_merge_fragments_merges_similar_summaries_only(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    entries = [
        make_entry(i, link=f"https://h{i}.example/{i}") for i in range(3)
    ]
    candidates = [
        make_candidate(entries[0], title="A", summary="sa", entities=("X",)),
        make_candidate(entries[1], title="B", index=1),
        make_candidate(entries[2], title="C", summary="sc", index=2),
    ]
    llm = fake_llm(embeddings_results=[[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]])
    grouping = make_grouping(settings_stub, llm)

    # Act
    merged = await grouping._merge_fragments(
        candidates, np.tile([1.0, 0.0], (3, 1)), RateBaseline(0.0, 1.0)
    )

    # Assert
    assert llm.embeddings_calls[0][1] == ["A X sa", "B", "C  sc"]
    assert [m.entries for m in merged] == [
        (entries[0], entries[1]), (entries[2],)
    ]


async def test_merge_fragments_empty_input_skips_embedding(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange
    llm = fake_llm()
    grouping = make_grouping(settings_stub, llm)

    # Act
    merged = await grouping._merge_fragments(
        [], np.empty((0, 2)), RateBaseline(0.0, 1.0)
    )

    # Assert
    assert merged == []
    assert llm.embeddings_calls == []


async def test_call_merges_map_outputs_with_matching_summaries(
    settings_stub: Settings, fake_llm: type[FakeLlm]
) -> None:
    # Arrange: two article clusters (orthogonal entry vectors) whose map
    # summaries embed identically, i.e. two fragments of one event.
    entries = [
        make_entry(1, link="https://a.example/1"),
        make_entry(2, link="https://a.example/2"),
        make_entry(3, link="https://b.example/3"),
        make_entry(4, link="https://b.example/4"),
    ]
    vectors = [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
    llm = fake_llm(
        embeddings_results=[vectors, [[1.0, 0.0], [1.0, 0.0]]],
        chat_parsed_results=[summary("Part 1"), summary("Part 2")],
    )
    settings = make_settings(
        settings_stub, grouping_k_sigma=1.0, grouping_window_hours=0
    )
    grouping = make_grouping(settings, llm)

    # Act
    result = await grouping(entries, focus="FOCUS")

    # Assert
    assert len(llm.chat_parsed_calls) == 2
    assert len(result) == 1
    assert set(result[0].links) == {entry.link for entry in entries}
