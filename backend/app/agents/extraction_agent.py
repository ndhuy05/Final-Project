"""
ExtractionAgent — converts raw PDF page images into searchable text chunks,
structured metadata, and vector embeddings stored in Qdrant.

Replaces: backend/app/services/extracting_service.py

Lifecycle (called from the papers router on every upload):
    agent = ExtractionAgent()
    pages  = await agent.extract_pages(paper_id, image_paths, extract_metadata=True)
    chunks = agent.chunk_text(pages)
    vectors = await agent.embed_chunks(chunks)
    await agent.index_to_vector_store(paper_id, chunks, vectors)
"""
import base64
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.agents.prompts.extraction_prompts import EXTRACTION_PROMPT, METADATA_SUFFIX

logger = logging.getLogger(__name__)


class ExtractionAgent:
    """
    Ingests PDF page images and produces:
    - plain text per page (via vision LLM)
    - structured bibliographic metadata (from page 0)
    - text chunks ready for embedding
    - embeddings + Qdrant upsert
    """

    model: str = settings.RAG_VISION_MODEL

    def __init__(self) -> None:
        self._client: AsyncOpenAI | None = None

    # --- Private helpers ---

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=settings.OPENROUTER_API_KEY,
                base_url="https://openrouter.ai/api/v1",
            )
        return self._client

    @staticmethod
    def _encode_image(path: str) -> str:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    @staticmethod
    def _extract_json(raw: str) -> dict[str, Any]:
        """
        Robustly extract a JSON object from a model response.
        3-tier fallback: direct parse → strip markdown fences → regex.
        Raises json.JSONDecodeError if all attempts fail.
        """
        text = raw.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        if "```" in text:
            for part in text.split("```"):
                candidate = part.strip()
                if candidate.startswith("json"):
                    candidate = candidate[4:].strip()
                try:
                    return json.loads(candidate)
                except json.JSONDecodeError:
                    continue
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise json.JSONDecodeError("No JSON object found", text, 0)

    def _parse_batch_response(
        self,
        raw: str,
        pages: list[tuple[int, str]],
        extract_metadata: bool,
    ) -> list[dict[str, Any]]:
        """
        Parse VLM response for a batch of pages (currently always size 1).
        Strips ---METADATA--- block from the end when extract_metadata=True.
        Returns [{"page_num": int, "text": str, "metadata": dict}].
        """
        global_metadata: dict[str, Any] = {}
        if extract_metadata and "---METADATA---" in raw:
            parts = raw.split("---METADATA---", 1)
            raw = parts[0].strip()
            try:
                global_metadata = self._extract_json(parts[1].strip())
            except Exception as e:
                logger.warning("Failed to parse metadata JSON: %s", e)

        results = []
        for i, (page_num, _) in enumerate(pages):
            results.append({
                "page_num": page_num,
                "text": raw if i == 0 else "",
                "metadata": global_metadata if (i == 0 and extract_metadata) else {},
            })
        return results

    # --- Public API ---

    async def extract_pages(
        self,
        pages: list[tuple[int, str]],
        extract_metadata: bool = False,
    ) -> list[dict[str, Any]]:
        """
        Vision LLM call to extract content from one or more page images.

        pages: list of (0-based page_num, image_path) pairs.
        extract_metadata: set True when the batch contains page 0.
                          The model appends ---METADATA--- JSON at the end.

        Returns [{"page_num": int, "text": str, "metadata": dict}].
        metadata is populated only for the first entry when extract_metadata=True.
        """
        client = self._get_client()
        prompt = EXTRACTION_PROMPT + (METADATA_SUFFIX if extract_metadata else "")

        content: list[dict] = [{"type": "text", "text": prompt}]
        for _, path in pages:
            b64 = self._encode_image(path)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
            })

        max_tokens = min(4096 * len(pages), 8192)

        response = await client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            max_tokens=max_tokens,
        )
        raw = (response.choices[0].message.content or "").strip()
        return self._parse_batch_response(raw, pages, extract_metadata)

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks using RecursiveCharacterTextSplitter.
        500-char chunks with 75-char overlap — matches Qdrant index expectations.
        """
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=75)
        return splitter.split_text(text)

    async def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        """
        Embed a list of text chunks via OpenRouter embeddings API.
        Returns a parallel list of dense vectors.
        """
        from app.services.embedding_service import embed_texts
        return await embed_texts(chunks)

    async def index_to_vector_store(
        self,
        paper_id: str,
        page_num: int,
        chunks: list[str],
        vectors: list[list[float]],
    ) -> None:
        """
        Upsert chunk vectors into Qdrant with payload:
            {type, paper_id, page_num, content}
        """
        from app.services.qdrant_service import upsert_chunks
        await upsert_chunks(paper_id=paper_id, page_num=page_num, chunks=chunks, vectors=vectors)
