import logging
from typing import Protocol

from news.digest.llm_client import LlmClient
from news.digest.prompts import grouping_system_prompt, grouping_user_prompt
from news.digest.schemas import NewsRecord, NewsResponse, RssEntry
from news.settings import Settings

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    pass


class Grouping(Protocol):
    async def __call__(
        self, entries: list[RssEntry], *, focus: str
    ) -> list[NewsRecord]: ...


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


def format_entries(entries: list[RssEntry], *, content_max_chars: int) -> str:
    return "\n".join(
        format_entry(e.id, e, content_max_chars=content_max_chars)
        for e in entries
    )


class LlmGrouping:
    def __init__(self, settings: Settings, llm: LlmClient) -> None:
        self.settings = settings
        self.llm = llm

    async def __call__(
        self, entries: list[RssEntry], *, focus: str
    ) -> list[NewsRecord]:
        formatted_entries = format_entries(
            entries,
            content_max_chars=self.settings.grouping_content_max_chars,
        )
        logger.info(
            "extracting groups formatted_entries_chars=%s focus_chars=%s",
            len(formatted_entries),
            len(focus),
        )

        parsed_response = await self.llm.chat_parsed(
            self.settings.model_grouping,
            [
                {
                    "role": "system",
                    "content": grouping_system_prompt(focus),
                },
                {
                    "role": "user",
                    "content": grouping_user_prompt(formatted_entries),
                },
            ],
            response_format=NewsResponse,
            reasoning_effort="medium",
            temperature=0.1,
        )
        if parsed_response is None:
            logger.warning("grouping query returned empty content")
            raise PipelineError("grouping query returned no content")
        if not parsed_response.records:
            logger.warning("grouping produced empty records")
        logger.info(
            "extracted groups records_count=%s",
            len(parsed_response.records),
        )
        return parsed_response.records
