import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import aiofiles
from bs4 import BeautifulSoup

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


async def refine_record(
    llm: LlmClient,
    record: NewsRecord,
    *,
    model: str,
    max_links: int,
    today: date,
) -> str | None:
    links = [str(link) for link in record.links[:max_links]]
    messages = [
        {"role": "system", "content": refinement_system_prompt(today)},
        {
            "role": "user",
            "content": refinement_user_prompt(
                record.title or "", record.summary or "", links
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


class DigestService:
    def __init__(
        self, settings: Settings, miniflux: MinifluxClient, llm: LlmClient
    ) -> None:
        self.settings = settings
        self.miniflux = miniflux
        self.llm = llm

    async def _run_pipeline(
        self, aggregation: Aggregation, now: datetime
    ) -> Path:
        today = now.date()
        entries = await fetch_entries(
            self.miniflux,
            category=aggregation.miniflux_category,
            lookback_hours=self.settings.fetch_lookback_hours,
            limit=self.settings.fetch_limit,
            max_chars=self.settings.entry_content_max_chars,
            now=now,
        )
        records: list[Any] = []
        if entries:
            formatted = format_entries(entries)
            parsed_records = await extract_groups(
                self.llm,
                formatted,
                trending_model=self.settings.model_trending,
                grouping_model=self.settings.model_grouping,
                focus=aggregation.focus,
            )
            records = await refine_all(
                self.llm,
                parsed_records,
                model=self.settings.model_refinement,
                max_links=self.settings.refine_max_links,
                today=today,
            )
        digest = Digest(generated_at=now.isoformat(), records=records)
        return await write_digest(
            digest,
            self.settings.digest_output_dir,
            today,
            name=aggregation.name,
        )

    async def __call__(self) -> list[Path]:
        now = datetime.now(UTC)
        paths = []
        for aggregation in config.AGGREGATIONS:
            try:
                paths.append(await self._run_pipeline(aggregation, now))
            except Exception as exc:
                logger.error(
                    "aggregation %s failed: %s", aggregation.name, exc
                )
        logger.info("news digests written: %s", paths)
        return paths
