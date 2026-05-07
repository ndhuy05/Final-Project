"""
Embedding service: text embeddings via OpenRouter API.
Model: qwen/qwen3-embedding-8b (4096-dim) by default.
Both embed_text and embed_texts are async — callers must await them.
Failed calls are retried up to 3 times with exponential backoff (2 s, 4 s, 8 s).
"""
from typing import List
import openai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from app.core.config import settings

_RETRYABLE = (openai.APIError, openai.APIConnectionError, openai.RateLimitError)


def _get_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
async def embed_text(text: str) -> List[float]:
    """Embed a single string. Returns a flat 4096-dim vector."""
    client = _get_client()
    response = await client.embeddings.create(
        model=settings.RAG_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=8),
    retry=retry_if_exception_type(_RETRYABLE),
    reraise=True,
)
async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings in one batched API call. Returns one vector per text."""
    client = _get_client()
    response = await client.embeddings.create(
        model=settings.RAG_EMBEDDING_MODEL,
        input=texts,
    )
    # API returns items sorted by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
