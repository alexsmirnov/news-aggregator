import openai
import pytest

from news.digest.llm_client import LlmClient
from news.settings import Settings


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    return dot / (norm_a * norm_b)


@pytest.mark.integration
async def test_embedding_model_returns_vectors_of_configured_dimension(
    eval_settings: Settings,
) -> None:
    llm = LlmClient(
        openai.AsyncOpenAI(
            api_key=eval_settings.litellm_api_key.get_secret_value(),
            base_url=str(eval_settings.litellm_router),
        )
    )
    try:
        vectors = await llm.embeddings(
            eval_settings.model_embedding,
            ["Federal Reserve raises interest rates", "banana bread recipe"],
            dimensions=eval_settings.embedding_dimensions,
        )
    finally:
        await llm.aclose()

    assert len(vectors) == 2
    assert all(
        len(vector) == eval_settings.embedding_dimensions
        for vector in vectors
    )


@pytest.mark.integration
async def test_embedding_model_ranks_similar_titles_closer(
    eval_settings: Settings,
) -> None:
    llm = LlmClient(
        openai.AsyncOpenAI(
            api_key=eval_settings.litellm_api_key.get_secret_value(),
            base_url=str(eval_settings.litellm_router),
        )
    )
    try:
        vectors = await llm.embeddings(
            eval_settings.model_embedding,
            [
                "Federal Reserve raises interest rates",
                "Fed hikes benchmark rate again",
                "banana bread recipe",
            ],
            dimensions=eval_settings.embedding_dimensions,
        )
    finally:
        await llm.aclose()

    related, unrelated = vectors[0], vectors[1]
    control = vectors[2]
    assert _cosine(related, unrelated) > _cosine(related, control)
