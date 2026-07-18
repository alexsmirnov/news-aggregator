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

T = TypeVar("T", bound=BaseModel)


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

        async def _do() -> str | None:
            response = await self.client.chat.completions.create(
                model=model, messages=sdk_messages, **kwargs
            )
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

        async def _do() -> T | None:
            response = await self.client.chat.completions.parse(
                model=model,
                messages=sdk_messages,
                response_format=response_format,
                **kwargs,
            )
            return response.choices[0].message.parsed

        return await self._retrying(_do)
