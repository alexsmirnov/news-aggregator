import types
from unittest.mock import AsyncMock

import httpx
import openai
import pytest

from news.digest.llm_client import LlmClient, llm_client
from news.digest.schemas import NewsResponse
from news.settings import Settings


@pytest.fixture
def fake_openai() -> types.SimpleNamespace:
    return types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(
                create=AsyncMock(), parse=AsyncMock()
            )
        )
    )


def make_response(content: str | None) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(content=content)
            )
        ]
    )


@pytest.fixture
def make_client(fake_openai: types.SimpleNamespace) -> LlmClient:
    return LlmClient(
        api_key="k",
        base_url="http://router.test",
        client=fake_openai,
        attempts=3,
        max_wait=0,
    )


@pytest.fixture
def rate_limit_error() -> openai.RateLimitError:
    request = httpx.Request("POST", "http://router.test/v1/chat/completions")
    return openai.RateLimitError(
        "rate", response=httpx.Response(429, request=request), body=None
    )


@pytest.fixture
def connection_error() -> openai.APIConnectionError:
    request = httpx.Request("POST", "http://router.test/v1/chat/completions")
    return openai.APIConnectionError(request=request)


async def test_chat_returns_message_content_and_passes_kwargs(
    fake_openai, make_client
):
    # Arrange
    fake_openai.chat.completions.create.return_value = make_response("hello")

    # Act
    text = await make_client.chat(
        model="gemini-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )

    # Assert
    assert text == "hello"
    fake_openai.chat.completions.create.assert_awaited_once_with(
        model="gemini-flash",
        messages=[{"role": "user", "content": "hi"}],
        reasoning_effort="high",
    )


async def test_chat_parsed_returns_parsed_model(fake_openai, make_client):
    # Arrange
    parsed = NewsResponse(records=[])
    fake_openai.chat.completions.parse.return_value = types.SimpleNamespace(
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(parsed=parsed)
            )
        ]
    )

    # Act
    result = await make_client.chat_parsed(
        model="gemini-flash", messages=[], response_format=NewsResponse
    )

    # Assert
    assert result is parsed
    fake_openai.chat.completions.parse.assert_awaited_once_with(
        model="gemini-flash", messages=[], response_format=NewsResponse
    )


async def test_chat_retries_transient_error_then_succeeds(
    fake_openai, make_client, rate_limit_error
):
    # Arrange
    fake_openai.chat.completions.create.side_effect = [
        rate_limit_error,
        make_response("ok"),
    ]

    # Act
    text = await make_client.chat(model="m", messages=[])

    # Assert
    assert text == "ok"
    assert fake_openai.chat.completions.create.await_count == 2


async def test_chat_reraises_after_retry_exhaustion(
    fake_openai, make_client, connection_error
):
    # Arrange
    fake_openai.chat.completions.create.side_effect = [
        connection_error,
        connection_error,
        connection_error,
    ]

    # Act / Assert
    with pytest.raises(openai.APIConnectionError):
        await make_client.chat(model="m", messages=[])
    assert fake_openai.chat.completions.create.await_count == 3


async def test_chat_does_not_retry_non_transient_errors(
    fake_openai, make_client
):
    # Arrange
    fake_openai.chat.completions.create.side_effect = ValueError("bad")

    # Act / Assert
    with pytest.raises(ValueError):
        await make_client.chat(model="m", messages=[])
    assert fake_openai.chat.completions.create.await_count == 1


async def test_llm_client_factory_configures_and_closes_client(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    # Arrange
    openai_client = types.SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        "news.digest.llm_client.openai.AsyncOpenAI", lambda **_: openai_client
    )
    settings = Settings(
        miniflux_api_base="http://miniflux.test",
        miniflux_api_key="unused",
        litellm_api_key="secret",
        litellm_router="http://router.test",
        digest_output_dir=tmp_path,
    )

    # Act
    async with llm_client(settings) as client:
        # Assert
        assert client.client is openai_client

    # Assert
    openai_client.close.assert_awaited_once()


async def test_llm_client_factory_closes_client_when_context_body_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path,
):
    # Arrange
    openai_client = types.SimpleNamespace(close=AsyncMock())
    monkeypatch.setattr(
        "news.digest.llm_client.openai.AsyncOpenAI", lambda **_: openai_client
    )
    settings = Settings(
        miniflux_api_base="http://miniflux.test",
        miniflux_api_key="unused",
        litellm_api_key="secret",
        litellm_router="http://router.test",
        digest_output_dir=tmp_path,
    )

    # Act / Assert
    with pytest.raises(RuntimeError, match="boom"):
        async with llm_client(settings):
            raise RuntimeError("boom")

    openai_client.close.assert_awaited_once()
