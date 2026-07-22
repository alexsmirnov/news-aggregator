import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import aiofiles
from bs4 import BeautifulSoup

from news.settings import Aggregation
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


class DigestService:
    def __init__(
        self, settings: Settings, miniflux: MinifluxClient, llm: LlmClient
    ) -> None:
        self.settings = settings
        self.miniflux = miniflux
        self.llm = llm

    @staticmethod
    def strip_html(entry_content: str) -> str:
        return BeautifulSoup(entry_content, "html.parser").get_text(
            " ", strip=True
        )

    @staticmethod
    def format_entry(
        index: int, entry: RssEntry, *, content_max_chars: int
    ) -> str:
        return (
            f"# Entity {index}\n"
            f"Title: {entry.title}\n"
            f"Content: {entry.content[:content_max_chars]}\n"
            f"Source: {entry.source}\n"
            f"Link: {entry.link}\n"
        )

    @staticmethod
    def format_entries(
        entries: list[RssEntry], *, content_max_chars: int
    ) -> str:
        return "\n".join(
            DigestService.format_entry(
                e.id, e, content_max_chars=content_max_chars
            )
            for e in entries
        )

    @staticmethod
    def format_full_entry(entry: RssEntry) -> str:
        return (
            f"Title: {entry.title}\n"
            f"Content: {entry.content}\n"
            f"Link: {entry.link}\n"
        )

    @staticmethod
    def _normalize_link(link: str) -> str:
        # ponytail: rstrip("/") only; scheme/host normalization
        # mismatches between RssEntry.link and pydantic-normalized
        # HttpUrl are out of scope.
        return link.rstrip("/")

    async def fetch_entries(
        self,
        *,
        category: str,
        now: datetime,
    ) -> list[RssEntry]:
        published_after = int(
            (
                now - timedelta(hours=self.settings.fetch_lookback_hours)
            ).timestamp()
        )
        entries = await self.miniflux.get_entries(
            category,
            published_after=published_after,
            order="published_at",
            limit=self.settings.fetch_limit,
        )
        for entry in entries:
            entry.content = self.strip_html(entry.content)[
                : self.settings.entry_content_max_chars
            ]
        return entries

    async def extract_groups(
        self,
        formatted_entries: str,
        *,
        focus: str,
    ) -> list[NewsRecord]:
        trending = await self.llm.chat(
            self.settings.model_trending,
            [{"role": "user", "content": trending_query()}],
        )
        if trending is None:
            raise PipelineError("trending query returned no content")

        parsed_response = await self.llm.chat_parsed(
            self.settings.model_grouping,
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
        self,
        record: NewsRecord,
        entries_by_link: dict[str, RssEntry],
        *,
        today: date,
    ) -> str | None:
        group_entries = [
            entries_by_link[key]
            for link in record.links
            if (key := self._normalize_link(str(link))) in entries_by_link
        ]
        fetch_links = [
            entry.link
            for entry in sorted(group_entries, key=lambda e: len(e.content))[
                : self.settings.refine_max_links
            ]
        ]
        full_content = "\n".join(
            self.format_full_entry(entry) for entry in group_entries
        )
        messages = [
            {"role": "system", "content": refinement_system_prompt(today)},
            {
                "role": "user",
                "content": refinement_user_prompt(
                    record.title or "", full_content, fetch_links
                ),
            },
        ]
        try:
            return await self.llm.chat(
                self.settings.model_refinement,
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
        self,
        records: list[NewsRecord],
        entries: list[RssEntry],
        *,
        today: date,
    ) -> list[DigestRecord]:
        entries_by_link = {
            self._normalize_link(entry.link): entry for entry in entries
        }
        return [
            DigestRecord(
                title=record.title,
                refined_summary=await self.refine_record(
                    record, entries_by_link, today=today
                ),
                links=record.links,
            )
            for record in records
        ]

    @staticmethod
    async def write_digest(
        digest: Digest,
        output_dir: Path,
        today: date,
        *,
        name: str,
    ) -> Path:
        path = (
            output_dir
            / f"{today:%Y}"
            / f"{today:%m}"
            / f"{name}-{today:%d}.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        async with aiofiles.open(path, "w") as f:
            await f.write(digest.model_dump_json(indent=2))
        return path

    async def _run_pipeline(
        self, aggregation: Aggregation, now: datetime
    ) -> Path:
        today = now.date()
        entries = await self.fetch_entries(
            category=aggregation.miniflux_category,
            now=now,
        )
        records: list[DigestRecord] = []
        if entries:
            formatted_entries = self.format_entries(
                entries,
                content_max_chars=self.settings.grouping_content_max_chars,
            )
            news_records = await self.extract_groups(
                formatted_entries,
                focus=aggregation.focus,
            )
            records = await self.refine_all(
                news_records,
                entries,
                today=today,
            )
        digest = Digest(generated_at=now.isoformat(), records=records)
        return await self.write_digest(
            digest,
            self.settings.digest_output_dir,
            today,
            name=aggregation.name,
        )

    async def __call__(self) -> list[Path]:
        now = datetime.now(UTC)
        paths = []
        for aggregation in self.settings.aggregations:
            try:
                paths.append(await self._run_pipeline(aggregation, now))
            except Exception as exc:
                logger.error(
                    "aggregation %s failed: %s", aggregation.name, exc
                )
        logger.info("news digests written: %s", paths)
        return paths
