import asyncio
import logging
import math
from dataclasses import dataclass
from itertools import batched

from news.digest.llm_client import LlmClient
from news.digest.schemas import NewsRecord, RssEntry
from news.settings import Settings

logger = logging.getLogger(__name__)

EMBED_MAX_INPUT_TOKENS = 8192
CHARS_PER_TOKEN = 4  # ponytail: heuristic; BGE tokenizer not available
EMBED_MAX_INPUT_CHARS = EMBED_MAX_INPUT_TOKENS * CHARS_PER_TOKEN
EMBED_BATCH_SIZE = 64
MAX_CONCURRENT_EMBED_REQUESTS = 8


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
            "MapReduceGrouping only computes embeddings so far; "
            "clustering, scoring, and map/reduce are not implemented"
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
