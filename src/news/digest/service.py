import ipaddress
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles
from bs4 import BeautifulSoup
from pydantic import HttpUrl

from news import config
from news.config import Aggregation
from news.digest.llm_client import LlmClient
from news.digest.miniflux_client import MinifluxClient
from news.digest.prompts import (
    grouping_system_prompt,
    grouping_user_prompt,
    refinement_system_prompt,
    refinement_user_prompt,
    trending_query,
)
from news.digest.schemas import (
    Digest,
    DigestRecord,
    NewsRecord,
    NewsResponse,
    RssEntry,
)
from news.settings import Settings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


@asynccontextmanager
async def _miniflux_client(
    settings: Settings, provided: MinifluxClient | None
) -> AsyncIterator[MinifluxClient]:
    """Yield an injected client untouched, else create and close our own."""
    if provided is not None:
        yield provided
        return
    client = MinifluxClient(
        str(settings.miniflux_api_base), settings.miniflux_api_key
    )
    try:
        yield client
    finally:
        await client.aclose()


@asynccontextmanager
async def _llm_client(
    settings: Settings, provided: LlmClient | None
) -> AsyncIterator[LlmClient]:
    """Yield an injected client untouched, else create and close our own."""
    if provided is not None:
        yield provided
        return
    client = LlmClient(
        settings.litellm_api_key, str(settings.litellm_router)
    )
    try:
        yield client
    finally:
        await client.aclose()


def strip_html(entry_content: str) -> str:
    return BeautifulSoup(entry_content, "html.parser").get_text(
        " ", strip=True
    )


def format_entry(index: int, entry: RssEntry) -> str:
    return (
        f"# Entity {index}\n"
        f"Title: {entry.title}\n"
        f"Content: {entry.content}\n"
        f"Source: {entry.source}\n"
        f"Link: {entry.link}\n"
    )


def format_entries(entries: list[RssEntry]) -> str:
    return "\n".join(format_entry(e.id, e) for e in entries)


async def fetch_entries(
    client: MinifluxClient,
    *,
    category: str,
    lookback_hours: int,
    limit: int,
    max_chars: int,
    now: datetime,
) -> list[RssEntry]:
    category_id = await client.get_category_id(category)
    published_after = int(
        (now - timedelta(hours=lookback_hours)).timestamp()
    )
    raw_entries = await client.get_entries(
        category_id,
        published_after=published_after,
        order="published_at",
        limit=limit,
    )
    return [
        RssEntry(
            id=raw["id"],
            title=raw["title"],
            link=raw["url"],
            content=strip_html(raw["content"])[:max_chars],
            published_at=raw["published_at"],
            source=raw["feed"]["title"],
        )
        for raw in raw_entries
    ]


async def extract_groups(
    llm: LlmClient,
    formatted_entries: str,
    *,
    trending_model: str,
    grouping_model: str,
    focus: str,
) -> list[NewsRecord]:
    trending = await llm.chat(
        trending_model, [{"role": "user", "content": trending_query()}]
    )
    if trending is None:
        raise PipelineError("trending query returned no content")

    parsed_response = await llm.chat_parsed(
        grouping_model,
        [
            {
                "role": "system",
                "content": grouping_system_prompt(trending, focus),
            },
            {
                "role": "user",
                "content": grouping_user_prompt(formatted_entries),
            },
        ],
        response_format=NewsResponse,
        reasoning_effort="high",
        temperature=1.0,
    )
    if parsed_response is None:
        raise PipelineError("grouping query returned no content")
    return parsed_response.records


# ponytail: static IP/scheme check only, does not resolve hostnames.
# A DNS-rebinding domain (e.g. one that resolves to 127.0.0.1) passes
# through as "safe" here. Closing that gap would require controlling
# DNS resolution at request time, but the actual fetch happens inside
# the LLM provider's remote url_context tool, not in this process, so
# resolving here would not close the gap anyway (TOCTOU). Upgrade path
# if this becomes a real threat: drop url_context in favor of fetching
# links ourselves through a transport that validates the resolved IP.
def _is_safe_link(url: HttpUrl) -> bool:
    if url.scheme not in ("http", "https"):
        return False
    host = url.host
    if host is None or host == "localhost":
        return False
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return True
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


async def refine_record(
    llm: LlmClient,
    record: NewsRecord,
    *,
    model: str,
    max_links: int,
    today: date,
) -> str | None:
    safe_links = [
        str(link) for link in record.links if _is_safe_link(link)
    ][:max_links]
    messages = [
        {"role": "system", "content": refinement_system_prompt(today)},
        {
            "role": "user",
            "content": refinement_user_prompt(
                record.title or "", record.summary or "", safe_links
            ),
        },
    ]
    try:
        return await llm.chat(
            model,
            messages,
            tools=[{"url_context": {}}],
            extra_body={"thinkingBudget": -1},
        )
    except Exception:
        logger.warning(
            "refinement failed for %s", record.title, exc_info=True
        )
        return None


async def refine_all(
    llm: LlmClient,
    records: list[NewsRecord],
    *,
    model: str,
    max_links: int,
    today: date,
) -> list[DigestRecord]:
    digest_records = []
    for record in records:
        refined_summary = await refine_record(
            llm, record, model=model, max_links=max_links, today=today
        )
        digest_records.append(
            DigestRecord(
                title=record.title,
                summary=record.summary,
                refined_summary=refined_summary,
                links=record.links,
            )
        )
    return digest_records


async def write_digest(
    digest: Digest, output_dir: Path, today: date, *, name: str
) -> Path:
    path = (
        output_dir / f"{today:%Y}" / f"{today:%m}" / f"{name}-{today:%d}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(path, "w") as f:
        await f.write(digest.model_dump_json(indent=2))
    return path


async def run_pipeline(
    settings: Settings,
    aggregation: Aggregation,
    miniflux: MinifluxClient | None = None,
    llm: LlmClient | None = None,
    *,
    now: datetime | None = None,
) -> Path:
    now = now or datetime.now(UTC)
    today = now.date()
    async with (
        _miniflux_client(settings, miniflux) as miniflux,
        _llm_client(settings, llm) as llm,
    ):
        entries = await fetch_entries(
            miniflux,
            category=aggregation.miniflux_category,
            lookback_hours=settings.fetch_lookback_hours,
            limit=settings.fetch_limit,
            max_chars=settings.entry_content_max_chars,
            now=now,
        )
        records: list[Any] = []
        if entries:
            formatted = format_entries(entries)
            parsed_records = await extract_groups(
                llm,
                formatted,
                trending_model=settings.model_trending,
                grouping_model=settings.model_grouping,
                focus=aggregation.focus,
            )
            records = await refine_all(
                llm,
                parsed_records,
                model=settings.model_refinement,
                max_links=settings.refine_max_links,
                today=today,
            )
        digest = Digest(generated_at=now.isoformat(), records=records)
        return await write_digest(
            digest, settings.digest_output_dir, today, name=aggregation.name
        )


async def run_all_aggregations(
    settings: Settings,
    miniflux: MinifluxClient | None = None,
    llm: LlmClient | None = None,
    *,
    now: datetime | None = None,
) -> list[Path]:
    now = now or datetime.now(UTC)
    paths = []
    async with (
        _miniflux_client(settings, miniflux) as miniflux,
        _llm_client(settings, llm) as llm,
    ):
        for aggregation in config.AGGREGATIONS:
            try:
                paths.append(
                    await run_pipeline(
                        settings,
                        aggregation,
                        miniflux=miniflux,
                        llm=llm,
                        now=now,
                    )
                )
            except Exception as exc:
                logger.error(
                    "aggregation %s failed: %s", aggregation.name, exc
                )
        return paths
