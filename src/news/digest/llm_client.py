import logging
from collections.abc import AsyncGenerator, Awaitable, Callable
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
R = TypeVar("R")

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

    async def _with_retry(
        self,
        call: Callable[[], Awaitable[R]],
        *,
        retry_message: str,
        error_message: str,
        log_args: tuple[Any, ...],
    ) -> R:
        attempt = 0

        async def _do() -> R:
            nonlocal attempt
            attempt += 1
            if attempt > 1:
                logger.warning(retry_message, attempt, *log_args)
            try:
                return await call()
            except Exception:
                logger.error(
                    error_message, attempt, *log_args, exc_info=True
                )
                raise

        return await self._retrying(_do)

    async def chat(
        self, model: str, messages: list[dict[str, Any]], **kwargs: Any
    ) -> str | None:
        sdk_messages = cast(list[ChatCompletionMessageParam], messages)

        async def _call() -> str | None:
            response = await self.client.chat.completions.create(
                model=model, messages=sdk_messages, **kwargs
            )
            return response.choices[0].message.content

        return await self._with_retry(
            _call,
            retry_message="llm chat retry attempt=%s model=%s",
            error_message="llm chat failed attempt=%s model=%s",
            log_args=(model,),
        )

    async def chat_parsed(
        self,
        model: str,
        messages: list[dict[str, Any]],
        response_format: type[T],
        **kwargs: Any,
    ) -> T | None:
        sdk_messages = cast(list[ChatCompletionMessageParam], messages)

        async def _call() -> T | None:
            response = await self.client.chat.completions.parse(
                model=model,
                messages=sdk_messages,
                response_format=response_format,
                **kwargs,
            )
            return response.choices[0].message.parsed

        return await self._with_retry(
            _call,
            retry_message=(
                "llm parsed retry attempt=%s model=%s response_format=%s"
            ),
            error_message=(
                "llm parsed failed attempt=%s model=%s response_format=%s"
            ),
            log_args=(model, response_format.__name__),
        )

    async def embeddings(
        self, model: str, inputs: list[str], **kwargs: Any
    ) -> list[list[float]]:
        async def _call() -> list[list[float]]:
            response = await self.client.embeddings.create(
                model=model, input=inputs, **kwargs
            )
            # ponytail: the API does not guarantee response order, so
            # callers can only zip vectors back to inputs by index.
            return [
                item.embedding
                for item in sorted(
                    response.data, key=lambda item: item.index
                )
            ]

        return await self._with_retry(
            _call,
            retry_message="llm embeddings retry attempt=%s model=%s",
            error_message="llm embeddings failed attempt=%s model=%s",
            log_args=(model,),
        )


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
