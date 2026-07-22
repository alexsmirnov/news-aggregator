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


def is_transient_http_error(exc: BaseException) -> bool:
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        return status == 429 or status >= 500
    return isinstance(exc, httpx.RequestError)


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
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = client
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
        async def _do() -> int:
            response = await self.client.get(
                f"{self.base_url}/v1/categories"
            )
            response.raise_for_status()
            category = next(
                (c for c in response.json() if c["title"] == title), None
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
        order: str,
        limit: int,
    ) -> list[RssEntry]:
        category_id = await self.get_category_id(category_name)

        async def _do() -> list[RssEntry]:
            response = await self.client.get(
                f"{self.base_url}/v1/categories/{category_id}/entries",
                params={
                    "published_after": published_after,
                    "order": order,
                    "limit": limit,
                },
            )
            response.raise_for_status()
            return [
                RssEntry(
                    id=raw["id"],
                    title=raw["title"],
                    link=raw["url"],
                    content=raw["content"],
                    published_at=raw["published_at"],
                    source=raw["feed"]["title"],
                )
                for raw in response.json()["entries"]
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
