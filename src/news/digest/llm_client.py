import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any, TypeVar, cast

import openai
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, SecretStr
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
        api_key: str | SecretStr,
        base_url: str,
        client: openai.AsyncOpenAI | None = None,
        *,
        attempts: int = 3,
        min_wait: int = 2,
        max_wait: int = 30,
    ) -> None:
        key = (
            api_key.get_secret_value()
            if isinstance(api_key, SecretStr)
            else api_key
        )
        self.client = client or openai.AsyncOpenAI(
            api_key=key, base_url=str(base_url), timeout=60.0
        )
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
    client = LlmClient(settings.litellm_api_key, str(settings.litellm_router))
    try:
        yield client
    finally:
        await client.aclose()
