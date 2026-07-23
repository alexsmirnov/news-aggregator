import asyncio
import logging
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import aiofiles
from bs4 import BeautifulSoup

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
from news.settings import Aggregation, Settings

logger = logging.getLogger(__name__)

MAX_CONCURRENT_REFINEMENTS = 8


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
        published_before = int(now.timestamp())
        logger.info(
            "fetching entries category=%s published_after=%s "
            "published_before=%s limit=%s",
            category,
            published_after,
            published_before,
            self.settings.fetch_limit,
        )
        entries = await self.miniflux.get_entries(
            category,
            published_after=published_after,
            published_before=published_before,
            order="published_at",
            limit=self.settings.fetch_limit,
        )
        for entry in entries:
            entry.content = self.strip_html(entry.content)[
                : self.settings.entry_content_max_chars
            ]
        if not entries:
            logger.warning("no entries fetched category=%s", category)
        logger.info(
            "fetched entries category=%s entries_count=%s",
            category,
            len(entries),
        )
        return entries

    async def extract_groups(
        self,
        formatted_entries: str,
        *,
        focus: str,
    ) -> list[NewsRecord]:
        logger.info(
            "extracting groups formatted_entries_chars=%s focus_chars=%s",
            len(formatted_entries),
            len(focus),
        )
        trending = await self.llm.chat(
            self.settings.model_trending,
            [{"role": "user", "content": trending_query()}],
        )
        if trending is None:
            logger.warning("trending query returned empty content")
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
            logger.warning("grouping query returned empty content")
            raise PipelineError("grouping query returned no content")
        if not parsed_response.records:
            logger.warning("grouping produced empty records")
        logger.info(
            "extracted groups records_count=%s trending_chars=%s",
            len(parsed_response.records),
            len(trending),
        )
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
        unmatched_links = [
            str(link)
            for link in record.links
            if self._normalize_link(str(link)) not in entries_by_link
        ]
        if unmatched_links:
            logger.warning(
                "refine record has unmatched links title=%s links_count=%s "
                "matched_count=%s unmatched_links=%s",
                record.title,
                len(record.links),
                len(group_entries),
                unmatched_links,
            )
        fetch_links = [
            entry.link
            for entry in sorted(group_entries, key=lambda e: len(e.content))[
                : self.settings.refine_max_links
            ]
        ]
        full_content = "\n".join(
            self.format_full_entry(entry) for entry in group_entries
        )
        logger.info(
            "refining record title=%s links_count=%s "
            "matched_entries_count=%s fetch_links_count=%s "
            "full_content_chars=%s",
            record.title,
            len(record.links),
            len(group_entries),
            len(fetch_links),
            len(full_content),
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
            refined_summary = await self.llm.chat(
                self.settings.model_refinement,
                messages,
                tools=[{"url_context": {}}],
                extra_body={"thinkingBudget": -1},
            )
        except Exception:
            logger.error(
                "refinement failed title=%s", record.title, exc_info=True
            )
            return None

        if refined_summary is None:
            logger.warning(
                "refinement returned empty summary title=%s", record.title
            )
        else:
            logger.info(
                "refinement completed title=%s summary_chars=%s",
                record.title,
                len(refined_summary),
            )
        return refined_summary

    async def _refine_with_limit(
        self,
        record: NewsRecord,
        entries_by_link: dict[str, RssEntry],
        semaphore: asyncio.Semaphore,
        *,
        today: date,
    ) -> DigestRecord:
        async with semaphore:
            refined_summary = await self.refine_record(
                record, entries_by_link, today=today
            )
        return DigestRecord(
            title=record.title,
            refined_summary=refined_summary,
            links=record.links,
        )

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
        logger.info(
            "refining all records_count=%s entries_count=%s",
            len(records),
            len(entries),
        )
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_REFINEMENTS)
        digest_records = await asyncio.gather(
            *(
                self._refine_with_limit(
                    record,
                    entries_by_link,
                    semaphore,
                    today=today,
                )
                for record in records
            )
        )
        logger.info(
            "refine all completed digest_records_count=%s",
            len(digest_records),
        )
        return digest_records

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
        logger.info(
            "writing digest path=%s records_count=%s",
            path,
            len(digest.records),
        )
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            async with aiofiles.open(path, "w") as f:
                await f.write(digest.model_dump_json(indent=2))
        except Exception:
            logger.error("failed writing digest path=%s", path, exc_info=True)
            raise
        logger.info("digest written path=%s", path)
        return path

    async def _run_pipeline(
        self, aggregation: Aggregation, now: datetime
    ) -> Path:
        today = now.date()
        logger.info(
            "starting digest pipeline aggregation=%s category=%s",
            aggregation.name,
            aggregation.miniflux_category,
        )
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
            logger.info(
                "formatted entries aggregation=%s formatted_entries_chars=%s",
                aggregation.name,
                len(formatted_entries),
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
        else:
            logger.warning(
                "no entries to process aggregation=%s", aggregation.name
            )
        digest = Digest(generated_at=now.isoformat(), records=records)
        logger.info(
            "digest constructed aggregation=%s digest_records_count=%s",
            aggregation.name,
            len(records),
        )
        return await self.write_digest(
            digest,
            self.settings.digest_output_dir,
            today,
            name=aggregation.name,
        )

    async def __call__(self) -> list[Path]:
        now = datetime.now(UTC)
        logger.info(
            "starting digest service aggregations_count=%s",
            len(self.settings.aggregations)
        )
        paths = []
        for aggregation in self.settings.aggregations:
            try:
                paths.append(await self._run_pipeline(aggregation, now))
            except Exception as exc:
                logger.error(
                    "aggregation %s failed: %s", aggregation.name, exc
                )
        logger.info(
            "news digests written count=%s paths=%s", len(paths), paths
        )
        return paths
