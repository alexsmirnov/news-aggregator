import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_random,
)

from news.settings import Settings

T = TypeVar("T", bound=BaseModel)

logger = logging.getLogger(__name__)


class LlmClient:
    def __init__(
        self,
        client: openai.AsyncOpenAI,
        *,
        attempts: int = 3,
        min_wait: int = 2,
        max_wait: int = 30,
    ) -> None:
        self.client = client
        self._retrying = AsyncRetrying(
            stop=stop_after_attempt(attempts),
            wait=wait_exponential(
                multiplier=1, min=min_wait, max=max_wait, exp_base=2
            )
            + wait_random(0, 1),
            retry=retry_if_exception_type(
                (
                    openai.APIConnectionError,
                    openai.RateLimitError,
                    openai.InternalServerError,
                )
            ),
            reraise=True,
        )

    async def aclose(self) -> None:
        await self.client.close()

    async def chat(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> str | None:
        sdk_messages = cast(list[ChatCompletionMessageParam], messages)
        attempt = 0

        async def _do() -> str | None:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                logger.warning(
                    "llm chat retry attempt=%s model=%s", attempt, model
                )
            try:
                response = await self.client.chat.completions.create(
                    model=model, messages=sdk_messages, **kwargs
                )
            except Exception:
                logger.error(
                    "llm chat failed attempt=%s model=%s",
                    attempt,
                    model,
                    exc_info=True,
                )
                raise
            return response.choices[0].message.content

        return await self._retrying(_do)

    async def chat_parsed(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_format: type[T],
        **kwargs: Any,
    ) -> T | None:
        sdk_messages = cast(list[ChatCompletionMessageParam], messages)
        attempt = 0

        async def _do() -> T | None:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                logger.warning(
                    "llm parsed retry attempt=%s model=%s response_format=%s",
                    attempt,
                    model,
                    response_format.__name__,
                )
            try:
                response = await self.client.chat.completions.parse(
                    model=model,
                    messages=sdk_messages,
                    response_format=response_format,
                    **kwargs,
                )
            except Exception:
                logger.error(
                    "llm parsed failed attempt=%s model=%s response_format=%s",
                    attempt,
                    model,
                    response_format.__name__,
                    exc_info=True,
                )
                raise
            return response.choices[0].message.parsed

        return await self._retrying(_do)


@asynccontextmanager
async def llm_client(settings: Settings) -> AsyncGenerator[LlmClient]:
    openai_client = openai.AsyncOpenAI(
        api_key=settings.litellm_api_key.get_secret_value(),
        base_url=str(settings.litellm_router),
        timeout=180.0,
    )
    client = LlmClient(
        openai_client,
        attempts=settings.retry_attempts,
        min_wait=settings.retry_min_wait_s,
        max_wait=settings.retry_max_wait_s,
    )
    try:
        yield client
    finally:
        await client.aclose()
