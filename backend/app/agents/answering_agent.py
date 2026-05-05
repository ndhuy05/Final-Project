"""
AnsweringAgent — generates the final answer from retrieved context and page images.

Replaces: backend/app/services/answering_service.py

Called after PlannerAgent has executed all retrieval actions:
    agent = AnsweringAgent()
    answer = await agent.generate_answer(question, image_paths, results)
    citations = agent.format_citations(results)
"""
import base64
import logging
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.agents.prompts.answering_prompts import ANSWER_PROMPT, METADATA_ANSWER_PROMPT

logger = logging.getLogger(__name__)


class AnsweringAgent:
    """
    Generates the final VLM answer from retrieved page images and text context.
    Uses OPENROUTER_ANSWER_MODEL (vision-capable).
    """

    model: str = settings.OPENROUTER_ANSWER_MODEL

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

    # --- Public API ---

    async def generate_answer(
        self,
        question: str,
        image_paths: list[str] | None = None,
        results: list[dict[str, Any]] | None = None,
    ) -> str:
        """
        Generate an answer using OPENROUTER_ANSWER_MODEL.

        With images: vision call — model reads page images alongside the question.
        Without images: text-only call — question already contains any metadata context.

        results: used for debug logging only.
        """
        if image_paths is None:
            image_paths = []
        if results is None:
            results = []

        logger.debug("=== AnsweringAgent.generate_answer ===")
        for i, r in enumerate(results, 1):
            rtype = r.get("type", "?")
            page = r.get("page_num", "?")
            title = r.get("paper_title", "Unknown")
            score = round(r.get("score", 0), 4)
            logger.debug("  [%d] %-5s | page %s | score %s | %s", i, rtype, page, score, title)
        logger.debug(
            "  images sent: %s",
            ["/".join(p.replace("\\", "/").split("/")[-2:]) for p in image_paths],
        )
        logger.debug("======================================")

        if image_paths:
            user_content: list[dict] = [{"type": "text", "text": question}]
            for path in image_paths:
                b64 = self._encode_image(path)
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{b64}"},
                    "detail": "high",
                })
            system = ANSWER_PROMPT
            user_msg: Any = user_content
        else:
            system = METADATA_ANSWER_PROMPT
            user_msg = question

        response = await self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=4096,
        )
        return (response.choices[0].message.content or "").strip()

    def format_citations(self, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Convert raw retrieval results into frontend-ready citation objects.

        Returns: [{"id": str, "title": str, "page": int, "excerpt": str, "score": float}]
        """
        citations = []
        for chunk in chunks:
            citations.append({
                "id": chunk.get("id", ""),
                "title": chunk.get("paper_title", ""),
                "page": chunk.get("page_num"),
                "excerpt": (chunk.get("content") or "")[:300],
                "score": round(chunk.get("score", 0.0), 4),
            })
        return citations
