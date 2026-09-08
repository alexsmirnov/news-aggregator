import asyncio
import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from itertools import batched
from typing import TypedDict
from urllib.parse import urlsplit

import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

from news.digest.llm_client import LlmClient
from news.digest.schemas import NewsRecord, RssEntry
from news.settings import Settings

logger = logging.getLogger(__name__)

EMBED_MAX_INPUT_TOKENS = 8192
CHARS_PER_TOKEN = 4  # ponytail: heuristic; BGE tokenizer not available
EMBED_MAX_INPUT_CHARS = EMBED_MAX_INPUT_TOKENS * CHARS_PER_TOKEN
EMBED_BATCH_SIZE = 64
MAX_CONCURRENT_EMBED_REQUESTS = 8
MAX_COSINE_DISTANCE = 2.0


def _embedding_input(text: str, *, fallback: str = "") -> str:
    # ponytail: embeddings APIs reject empty input; "(empty)" is a last
    # resort when both title and content are blank.
    non_empty = text.strip() or fallback.strip() or "(empty)"
    return non_empty[:EMBED_MAX_INPUT_CHARS]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(component * component for component in vector))
    if norm == 0:
        # ponytail: a zero vector has no direction; leave it unchanged
        # rather than dividing by zero.
        return vector
    return [component / norm for component in vector]


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
class EmbeddedEntry:
    entry: RssEntry
    title_vector: list[float]
    content_vector: list[float]


class MapReduceGrouping:
    def __init__(self, settings: Settings, llm: LlmClient) -> None:
        self.settings = settings
        self.llm = llm

    async def __call__(
        self, entries: list[RssEntry], *, focus: str
    ) -> list[NewsRecord]:
        logger.info(
            "map-reduce grouping not implemented "
            "entries_count=%s focus_chars=%s",
            len(entries),
            len(focus),
        )
        raise NotImplementedError(
            "MapReduceGrouping map/reduce is not implemented"
        )

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
                centroid = _normalize(
                    embeddings[indices].mean(axis=0).tolist()
                )
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
    ) -> list[EmbeddedEntry]:
        title_texts = [_embedding_input(entry.title) for entry in entries]
        content_texts = [
            _embedding_input(entry.content, fallback=entry.title)
            for entry in entries
        ]
        vectors = await self._embed_texts(title_texts + content_texts)
        title_vectors = [
            _normalize(vector) for vector in vectors[: len(entries)]
        ]
        content_vectors = [
            _normalize(vector) for vector in vectors[len(entries) :]
        ]
        logger.info(
            "embedded entries entries_count=%s dimensions=%s",
            len(entries),
            self.settings.embedding_dimensions,
        )
        return [
            EmbeddedEntry(
                entry=entry,
                title_vector=title_vector,
                content_vector=content_vector,
            )
            for entry, title_vector, content_vector in zip(
                entries, title_vectors, content_vectors, strict=True
            )
        ]

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
