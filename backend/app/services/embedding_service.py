"""
Embedding service: text embeddings via OpenRouter API.
Model: openai/text-embedding-3-small (1536-dim) by default.
Both embed_text and embed_texts are async — callers must await them.
"""
from typing import List
import openai
from app.core.config import settings


def _get_client() -> openai.AsyncOpenAI:
    return openai.AsyncOpenAI(
        api_key=settings.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
    )


async def embed_text(text: str) -> List[float]:
    """Embed a single string. Returns a flat 1536-dim vector."""
    client = _get_client()
    response = await client.embeddings.create(
        model=settings.OPENROUTER_EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings in one batched API call. Returns one vector per text."""
    client = _get_client()
    response = await client.embeddings.create(
        model=settings.OPENROUTER_EMBEDDING_MODEL,
        input=texts,
    )
    # API returns items sorted by index
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
