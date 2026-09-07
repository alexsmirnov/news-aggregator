import logging
from collections import defaultdict
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

import httpx
from pydantic import SecretStr
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from news.digest.schemas import RssEntry
from news.settings import Settings

logger = logging.getLogger(__name__)


def is_transient_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.RequestError)


def merge_duplicate_entries(entries: list[RssEntry]) -> list[RssEntry]:
    """Collapse repeated captures of one page into a single entry.

    Miniflux re-captures an article when its page is updated, emitting a new
    id and published_at for the same url. The update carries the most accurate
    text, the first capture carries the real publication time, so the survivor
    is the newest capture stamped with the oldest published_at.
    """
    captures: dict[str, list[RssEntry]] = defaultdict(list)
    for entry in entries:
        # ponytail: mirrors DigestService._normalize_link (service.py), which
        # keys the downstream lookup; not imported, that would invert layering.
        captures[entry.link.rstrip("/")].append(entry)
    return [_collapse_captures(group) for group in captures.values()]


def _collapse_captures(captures: list[RssEntry]) -> RssEntry:
    """Keep the newest capture, stamped with the earliest capture time.

    Newest and earliest are picked from the whole group instead of being
    folded together capture by capture: a running stamp would overwrite the
    published_at that the remaining comparisons are made against.
    """
    # ponytail: miniflux serializes published_at as RFC3339 UTC ("...Z"), so
    # string order is time order. Parse ISO if non-UTC offsets ever appear.
    newest = max(captures, key=lambda capture: capture.published_at)
    earliest = min(capture.published_at for capture in captures)
    if newest.published_at == earliest:
        return newest
    return newest.model_copy(update={"published_at": earliest})


class MinifluxClient:
    def __init__(
        self,
        base_url: str,
        api_key: SecretStr,
        client: httpx.AsyncClient,
        *,
        attempts: int = 3,
        min_wait: int = 2,
        max_wait: int = 30,
        max_page_limit: int = 1000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
        self.max_page_limit = max_page_limit
        self.client.headers["X-Auth-Token"] = api_key.get_secret_value()
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=1, min=min_wait, max=max_wait, exp_base=2
            )
            + wait_random(0, 1),
            retry=retry_if_exception(is_transient_http_error),
            reraise=True,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def get_category_id(self, title: str) -> int:
        attempt = 0

        async def _do() -> int:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                logger.warning(
                    "miniflux category retry attempt=%s title=%s",
                    attempt,
                    title,
                )
            try:
                response = await self.client.get(
                    f"{self.base_url}/v1/categories"
                )
                response.raise_for_status()
            except Exception:
                logger.error(
                    "miniflux category request failed attempt=%s title=%s",
                    attempt,
                    title,
                    exc_info=True,
                )
                raise
            payload = response.json()
            if not isinstance(payload, list):
                logger.warning(
                    "unexpected miniflux categories payload type=%s",
                    type(payload).__name__,
                )
                raise TypeError("invalid miniflux categories payload")
            category = next(
                (c for c in payload if c["title"] == title), None
            )
            if category is None:
                raise LookupError(f"unknown miniflux category: {title}")
            return category["id"]

        return await self._retrying(_do)

    async def get_entries(
        self,
        category_name: str,
        *,
        published_after: int,
        published_before: int,
        order: str,
        limit: int,
    ) -> list[RssEntry]:
        category_id = await self.get_category_id(category_name)
        entries: list[RssEntry] = []
        offset = 0
        while len(entries) < limit:
            page_limit = min(self.max_page_limit, limit - len(entries))
            page = await self._get_entries_page(
                category_name,
                category_id,
                published_after=published_after,
                published_before=published_before,
                order=order,
                limit=page_limit,
                offset=offset,
            )
            entries.extend(page)
            if len(page) < page_limit:
                break
            offset += page_limit

        # limit is an upper bound: dedup runs once paging stopped, extra pages
        # are not fetched to refill what duplicates removed.
        deduplicated = merge_duplicate_entries(entries)
        if len(deduplicated) < len(entries):
            logger.info(
                "collapsed duplicate entries category=%s before=%s after=%s",
                category_name,
                len(entries),
                len(deduplicated),
            )
        entries = deduplicated

        if not entries:
            logger.warning(
                "miniflux returned empty entries category=%s category_id=%s",
                category_name,
                category_id,
            )
        return entries

    async def _get_entries_page(
        self,
        category_name: str,
        category_id: int,
        *,
        published_after: int,
        published_before: int,
        order: str,
        limit: int,
        offset: int,
    ) -> list[RssEntry]:
        attempt = 0

        async def _do() -> list[RssEntry]:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                logger.warning(
                    "miniflux entries retry attempt=%s category=%s "
                    "category_id=%s offset=%s",
                    attempt,
                    category_name,
                    category_id,
                    offset,
                )
            try:
                response = await self.client.get(
                    f"{self.base_url}/v1/categories/{category_id}/entries",
                    params={
                        "published_after": published_after,
                        "published_before": published_before,
                        "order": order,
                        "limit": limit,
                        "offset": offset,
                    },
                )
                response.raise_for_status()
            except Exception:
                logger.error(
                    "miniflux entries request failed attempt=%s category=%s "
                    "category_id=%s offset=%s",
                    attempt,
                    category_name,
                    category_id,
                    offset,
                    exc_info=True,
                )
                raise

            payload = response.json()
            if not isinstance(payload, dict) or "entries" not in payload:
                logger.warning(
                    "unexpected miniflux entries payload type=%s",
                    type(payload).__name__,
                )
                raise TypeError("invalid miniflux entries payload")

            raw_entries = payload["entries"]
            if not isinstance(raw_entries, list):
                logger.warning(
                    "unexpected miniflux entries field type=%s",
                    type(raw_entries).__name__,
                )
                raise TypeError("invalid miniflux entries field")

            return [
                RssEntry(
                    id=raw["id"],
                    title=raw["title"],
                    link=raw["url"],
                    content=raw["content"],
                    published_at=raw["published_at"],
                    source=raw["feed"]["title"],
                )
                for raw in raw_entries
            ]

        return await self._retrying(_do)


@asynccontextmanager
async def miniflux_client(settings: Settings) -> AsyncGenerator[MinifluxClient]:
    client = MinifluxClient(
        str(settings.miniflux_api_base),
        settings.miniflux_api_key,
        httpx.AsyncClient(timeout=httpx.Timeout(30.0, connect=10.0)),
    )
    try:
        yield client
    finally:
        await client.aclose()
