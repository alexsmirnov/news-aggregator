import asyncio
import logging
import math
from bisect import bisect_left, bisect_right
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import batched
from typing import TypedDict
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize

from news.digest.grouping import PipelineError, format_entry
from news.digest.llm_client import LlmClient
from news.digest.prompts import (
    cluster_summary_system_prompt,
    cluster_summary_user_prompt,
    merge_system_prompt,
    merge_user_prompt,
)
from news.digest.schemas import (
    ClusterSummary,
    MergeResponse,
    NewsRecord,
    RssEntry,
)
from news.settings import Settings

logger = logging.getLogger(__name__)

EMBED_MAX_INPUT_TOKENS = 8192
CHARS_PER_TOKEN = 4  # ponytail: heuristic; BGE tokenizer not available
EMBED_MAX_INPUT_CHARS = EMBED_MAX_INPUT_TOKENS * CHARS_PER_TOKEN
EMBED_BATCH_SIZE = 64
MAX_CONCURRENT_EMBED_REQUESTS = 8
MAX_CONCURRENT_MAP_CALLS = 8
REDUCE_BATCH_SIZE = 8
MAX_COSINE_DISTANCE = 2.0


def _embedding_input(text: str, *, fallback: str = "") -> str:
    # ponytail: embeddings APIs reject empty input; "(empty)" is a last
    # resort when both title and content are blank.
    non_empty = text.strip() or fallback.strip() or "(empty)"
    return non_empty[:EMBED_MAX_INPUT_CHARS]


def _arrival_rate(group: list[RssEntry], floor_hours: float = 6.0) -> float:
    stamps = [datetime.fromisoformat(record.published_at) for record in group]
    span_hours = (max(stamps) - min(stamps)).total_seconds() / 3600.0
    return len(group) / max(span_hours, floor_hours)


class ScoredCluster(TypedDict):
    records: list[RssEntry]
    size: int
    unique_domains: int
    source_entropy: float
    effective_sources: float
    burst_z: float
    novelty: float
    trend_score: float


@dataclass(frozen=True, slots=True)
class Candidate:
    title: str
    summary: str
    entries: tuple[RssEntry, ...]
    trend_score: float


class MapReduceGrouping:
    def __init__(self, settings: Settings, llm: LlmClient) -> None:
        self.settings = settings
        self.llm = llm

    async def __call__(
        self, entries: list[RssEntry], *, focus: str
    ) -> list[NewsRecord]:
        logger.info(
            "map-reduce grouping started entries_count=%s focus_chars=%s",
            len(entries),
            len(focus),
        )
        sorted_entries = self.sort_by_published_at(entries)
        window = timedelta(hours=self.settings.grouping_window_hours)
        baseline_cut, clustering_cut = self.split_offsets(
            sorted_entries, window
        )
        vectors_all = await self.embed_entries(sorted_entries)
        baseline_vectors = vectors_all[:baseline_cut]
        clustering_entries = sorted_entries[clustering_cut:]
        vectors = vectors_all[clustering_cut:]
        threshold = self.calibrate_threshold(
            vectors, k_sigma=self.settings.grouping_k_sigma
        )
        labels = self.cluster(vectors, threshold)
        scored = self.score_clusters(
            clustering_entries, labels, vectors, baseline_vectors
        )
        selected = scored[: self.settings.grouping_map_clusters]
        logger.info(
            "selected clusters for mapping selected_count=%s "
            "total_clusters=%s",
            len(selected),
            len(scored),
        )
        candidates = await self._map_clusters(selected, focus)
        collapsed = await self._collapse(candidates)
        records = self._to_news_records(
            collapsed, self.settings.grouping_max_records
        )
        logger.info(
            "map-reduce grouping completed records_count=%s", len(records)
        )
        return records

    async def _map_one(
        self,
        cluster: ScoredCluster,
        focus: str,
        semaphore: asyncio.Semaphore,
    ) -> Candidate:
        block = "\n---\n".join(
            format_entry(
                entry.id,
                entry,
                content_max_chars=self.settings.grouping_content_max_chars,
            )
            for entry in cluster["records"]
        )
        parsed: ClusterSummary | None = None
        async with semaphore:
            try:
                parsed = await self.llm.chat_parsed(
                    self.settings.model_grouping,
                    [
                        {
                            "role": "system",
                            "content": cluster_summary_system_prompt(focus),
                        },
                        {
                            "role": "user",
                            "content": cluster_summary_user_prompt(block),
                        },
                    ],
                    response_format=ClusterSummary,
                    reasoning_effort="medium",
                    temperature=0.1,
                )
            except Exception:
                logger.warning(
                    "cluster summary failed, falling back to entry title",
                    exc_info=True,
                )
        if parsed is None:
            fallback_entry = cluster["records"][0]
            return Candidate(
                title=fallback_entry.title,
                summary="",
                entries=tuple(cluster["records"]),
                trend_score=cluster["trend_score"],
            )
        return Candidate(
            title=parsed.title,
            summary=parsed.summary,
            entries=tuple(cluster["records"]),
            trend_score=cluster["trend_score"],
        )

    async def _map_clusters(
        self, clusters: list[ScoredCluster], focus: str
    ) -> list[Candidate]:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_MAP_CALLS)
        return list(
            await asyncio.gather(
                *(
                    self._map_one(cluster, focus, semaphore)
                    for cluster in clusters
                )
            )
        )

    @staticmethod
    def _merge_batch(
        batch: list[Candidate], response: MergeResponse
    ) -> list[Candidate]:
        merged: list[Candidate] = []
        referenced: set[int] = set()
        for group in response.groups:
            valid_indexes = [
                index
                for index in group.member_indexes
                if 1 <= index <= len(batch)
            ]
            members = [batch[index - 1] for index in valid_indexes]
            if not members:
                continue
            referenced.update(valid_indexes)
            entries_by_link: dict[str, RssEntry] = {}
            for candidate in members:
                for entry in candidate.entries:
                    entries_by_link.setdefault(entry.link, entry)
            merged.append(Candidate(
                title=group.title,
                summary=group.summary,
                entries=tuple(entries_by_link.values()),
                trend_score=max(c.trend_score for c in members),
            ))
        leftovers = [
            candidate
            for index, candidate in enumerate(batch, start=1)
            if index not in referenced
        ]
        return merged + leftovers

    async def _reduce_one_batch(
        self, batch: list[Candidate]
    ) -> list[Candidate]:
        numbered = "\n---\n".join(
            f"# Entity {index}\nTitle: {candidate.title}\n"
            f"Summary: {candidate.summary}\n"
            for index, candidate in enumerate(batch, start=1)
        )
        response = await self.llm.chat_parsed(
            self.settings.model_grouping,
            [
                {"role": "system", "content": merge_system_prompt()},
                {"role": "user", "content": merge_user_prompt(numbered)},
            ],
            response_format=MergeResponse,
            reasoning_effort="medium",
            temperature=0.1,
        )
        if response is None:
            logger.warning("merge query returned empty content")
            raise PipelineError("merge query returned no content")
        return self._merge_batch(batch, response)

    async def _reduce_level(self, level: list[Candidate]) -> list[Candidate]:
        batches = [list(batch) for batch in batched(level, REDUCE_BATCH_SIZE)]
        merged_batches = await asyncio.gather(
            *(self._reduce_one_batch(batch) for batch in batches)
        )
        return [candidate for batch in merged_batches for candidate in batch]

    async def _collapse(self, level: list[Candidate]) -> list[Candidate]:
        while len(level) > REDUCE_BATCH_SIZE:
            next_level = await self._reduce_level(level)
            logger.info(
                "reduce pass completed previous_count=%s next_count=%s",
                len(level),
                len(next_level),
            )
            if len(next_level) >= len(level):
                logger.warning(
                    "reduce pass did not shrink level, stopping "
                    "level_count=%s",
                    len(level),
                )
                break
            level = next_level
        return level

    @staticmethod
    def _to_news_records(
        candidates: list[Candidate], max_records: int
    ) -> list[NewsRecord]:
        ranked = sorted(
            candidates,
            key=lambda candidate: candidate.trend_score,
            reverse=True,
        )
        return [
            NewsRecord(
                title=candidate.title,
                links=[entry.link for entry in candidate.entries],
            )
            for candidate in ranked[:max_records]
        ]

    @staticmethod
    def sort_by_published_at(entries: list[RssEntry]) -> list[RssEntry]:
        return sorted(
            entries,
            key=lambda entry: datetime.fromisoformat(entry.published_at),
        )

    @staticmethod
    def split_offsets(
        sorted_entries: list[RssEntry], window: timedelta
    ) -> tuple[int, int]:
        """Index boundaries into a published_at-ascending entry list.

        baseline window = sorted_entries[:baseline_cut] (<= last - window)
        clustering window = sorted_entries[clustering_cut:] (>= first + window)
        The two windows overlap; this only computes offsets, no embedding.
        """
        if not sorted_entries:
            # ponytail: no entries means no timestamps to bisect; nothing
            # to put in either window.
            return 0, 0
        stamps = [
            datetime.fromisoformat(entry.published_at)
            for entry in sorted_entries
        ]
        baseline_cut = bisect_right(stamps, stamps[-1] - window)
        clustering_cut = bisect_left(stamps, stamps[0] + window)
        return baseline_cut, clustering_cut

    @staticmethod
    def calibrate_threshold(
        embeddings: NDArray[np.float64],
        *,
        n_pairs: int = 20_000,
        k_sigma: float = 3.0,
        seed: int = 0,
    ) -> float:
        """Cosine *distance* cutoff, `k_sigma` std devs above this corpus's
        random-pair similarity baseline (the note's calibration constant
        `k`, renamed for clarity). Embedding score distributions shift per
        corpus, so a hardcoded constant does not transfer between runs;
        `k_sigma` is the knob an evaluation harness tunes.

        `embeddings` must be L2-normalized so dot product == cosine
        similarity.
        """
        rng = np.random.default_rng(seed)
        n = len(embeddings)
        if n < 2:
            # ponytail: no pairs means no baseline and no evidence to
            # merge anything on; 0.0 keeps every entry a singleton.
            return 0.0
        left = rng.integers(0, n, n_pairs)
        right = rng.integers(0, n, n_pairs)
        keep = left != right
        similarities = np.einsum(
            "ij,ij->i", embeddings[left[keep]], embeddings[right[keep]]
        )
        mean_similarity = float(similarities.mean())
        std_similarity = float(similarities.std())
        threshold = 1.0 - (mean_similarity + k_sigma * std_similarity)
        # ponytail: sklearn rejects a negative distance_threshold; 2.0 is
        # the maximum possible cosine distance.
        clamped = min(max(threshold, 0.0), MAX_COSINE_DISTANCE)
        logger.info(
            "calibrated threshold mean_similarity=%.4f "
            "std_similarity=%.4f k_sigma=%.2f threshold=%.4f",
            mean_similarity,
            std_similarity,
            k_sigma,
            clamped,
        )
        return clamped

    @staticmethod
    def cluster(
        embeddings: NDArray[np.float64], distance_threshold: float
    ) -> NDArray[np.int64]:
        """Average-linkage agglomerative clustering. No noise class: every
        record lands in a group, singletons included — the defect this
        design exists to prevent. Average linkage avoids the
        single-linkage chaining where one multi-topic bridge article
        collapses unrelated stories into one cluster.

        `embeddings` must be L2-normalized and non-zero (guaranteed by
        `_embedding_input`'s "(empty)" sentinel upstream).
        """
        n = len(embeddings)
        if n < 2:
            # ponytail: sklearn requires >=2 samples; 0 or 1 entries have
            # exactly one trivial labeling.
            return np.zeros(n, dtype=np.int64)
        model = AgglomerativeClustering(
            n_clusters=None,  # type: ignore[arg-type]
            distance_threshold=distance_threshold,
            metric="cosine",
            linkage="average",
        )
        labels = model.fit_predict(embeddings)
        logger.info(
            "clustered entries entries_count=%s clusters_count=%s "
            "singleton_count=%s distance_threshold=%.4f",
            n,
            model.n_clusters_,
            int(np.sum(np.bincount(labels) == 1)),
            distance_threshold,
        )
        return labels

    @staticmethod
    def score_clusters(
        records: list[RssEntry],
        labels: NDArray[np.int64],
        embeddings: NDArray[np.float64],
        baseline_embeddings: NDArray[np.float64] | None = None,
    ) -> list[ScoredCluster]:
        """Rank every group, including singletons, without filtering.

        Embedding rows must be aligned with records and L2-normalized;
        baseline vectors must have the same width. ISO timestamps within
        each group must have consistent timezone awareness.
        """
        if not len(records) == len(labels) == len(embeddings):
            raise ValueError("Records, labels and embeddings must be aligned")
        if not records:
            return []

        groups: dict[int, list[int]] = {}
        for index, label in enumerate(labels):
            groups.setdefault(int(label), []).append(index)
        rates = {
            label: _arrival_rate([records[i] for i in indices])
            for label, indices in groups.items()
        }
        # ponytail: corpus rates substitute for a previous-window baseline.
        rate_values = np.fromiter(rates.values(), dtype=float)
        mean_rate = float(rate_values.mean())
        std_rate = max(float(rate_values.std()), 1e-6)

        scored: list[ScoredCluster] = []
        for label, indices in groups.items():
            group = [records[i] for i in indices]
            size = len(group)
            # Unknown hostnames share one bucket, never one per feed title.
            counts = Counter(
                urlsplit(record.link).hostname or "" for record in group
            )
            entropy = -sum(
                (count / size) * math.log(count / size)
                for count in counts.values()
            )
            effective_sources = math.exp(entropy)
            normalized_entropy = entropy / math.log(size) if size > 1 else 0.0
            burst_z = (rates[label] - mean_rate) / std_rate
            novelty = 1.0
            if baseline_embeddings is not None and len(baseline_embeddings):
                centroid = normalize(
                    embeddings[indices].mean(axis=0).reshape(1, -1),
                    norm="l2",
                )[0]
                similarity = float((baseline_embeddings @ centroid).max())
                novelty = 1.0 - min(max(similarity, -1.0), 1.0)
            trend = (
                (1.0 + max(burst_z, 0.0))
                * (0.5 + normalized_entropy)
                * math.log1p(effective_sources)
                * (0.5 + novelty)
            )
            scored.append(ScoredCluster(
                records=group,
                size=size,
                unique_domains=len(counts),
                source_entropy=round(entropy, 3),
                effective_sources=round(effective_sources, 2),
                burst_z=round(burst_z, 2),
                novelty=round(novelty, 3),
                trend_score=round(trend, 3),
            ))
        return sorted(
            scored, key=lambda group: group["trend_score"], reverse=True
        )

    async def embed_entries(
        self, entries: list[RssEntry]
    ) -> NDArray[np.float64]:
        """Rows aligned with `entries`, L2-normalized so dot product ==
        cosine similarity (the precondition `calibrate_threshold` and
        `cluster` require).
        """
        if not entries:
            # ponytail: sklearn's normalize rejects 0 samples.
            return np.zeros(
                (0, self.settings.embedding_dimensions), dtype=np.float64
            )
        content_texts = [
            _embedding_input(
                f"{entry.title}\n\n{entry.content}", fallback=entry.title
            )
            for entry in entries
        ]
        vectors = await self._embed_texts(content_texts)
        logger.info(
            "embedded entries entries_count=%s dimensions=%s",
            len(entries),
            self.settings.embedding_dimensions,
        )
        return normalize(vectors, norm="l2")  # type: ignore[return-value]

    async def _embed_with_limit(
        self, batch: list[str], semaphore: asyncio.Semaphore
    ) -> list[list[float]]:
        async with semaphore:
            return await self.llm.embeddings(
                self.settings.model_embedding,
                batch,
                dimensions=self.settings.embedding_dimensions,
            )

    async def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_EMBED_REQUESTS)
        vectors_per_batch = await asyncio.gather(
            *(
                self._embed_with_limit(list(batch), semaphore)
                for batch in batched(texts, EMBED_BATCH_SIZE)
            )
        )
        return [
            vector
            for batch_vectors in vectors_per_batch
            for vector in batch_vectors
        ]
