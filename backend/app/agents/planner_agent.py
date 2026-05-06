"""
PlannerAgent — decides which RAG actions to execute for a given question.

Replaces: backend/app/services/planning_service.py

At the start of every chat turn the router calls:
    agent = PlannerAgent()
    actions = await agent.plan_actions(question, papers)
    # actions is a list of {"action": "read_metadata"|"retrieve", "paper_id": ..., "query": ...}
"""
import json
import logging
import re
from typing import Any

from openai import AsyncOpenAI

from app.core.config import settings
from app.agents.prompts.planner_prompts import PLANNER_PROMPT, PLANNER_USER

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    Decides which retrieval actions to run before answering a question.
    Uses a text-only LLM (RAG_PLANNER_MODEL) that outputs a JSON action list.
    Falls back to a single retrieve-all action on any error.
    """

    model: str = settings.RAG_PLANNER_MODEL

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
    def _extract_json_array(raw: str) -> list:
        """
        3-tier fallback to extract a JSON array from a model response:
        direct parse → strip markdown fences → regex.
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
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise json.JSONDecodeError("No JSON array found", text, 0)

    # --- Public API ---

    async def plan_actions(
        self,
        question: str,
        papers: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Plan a list of retrieval actions to answer the question.

        Returns a list of action dicts:
          {"action": "read_metadata", "paper_id": str | None}
          {"action": "retrieve",      "paper_id": str | None, "query": str}

        Falls back to [{"action": "retrieve", "paper_id": None, "query": question}] on any error.
        """
        fallback = [{"action": "retrieve", "paper_id": None, "query": question}]

        lines = []
        for p in papers:
            meta = p.get("metadata") or {}
            pid = p["id"]
            title = meta.get("title") or p.get("title", "Unknown")
            desc = meta.get("description") or "(no description available)"
            lines.append(f"- ID: {pid}\n  Title: {title}\n  Description: {desc}")
        papers_context = "\n\n".join(lines) or "(no papers available)"

        try:
            user_msg = PLANNER_USER.format(papers_context=papers_context, question=question)
            response = await self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": PLANNER_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=512,
            )
            raw = (response.choices[0].message.content or "").strip()
            logger.debug("PlannerAgent raw response: %s", raw[:300])

            result = self._extract_json_array(raw)
            if not isinstance(result, list) or not result:
                return fallback

            valid = []
            for action in result:
                if not isinstance(action, dict):
                    continue
                if action.get("action") not in ("read_metadata", "retrieve"):
                    continue
                if action["action"] == "retrieve" and not action.get("query"):
                    action["query"] = question
                valid.append(action)

            logger.debug("PlannerAgent actions for %r: %s", question[:60], valid)
            return valid if valid else fallback

        except Exception as e:
            logger.warning("PlannerAgent failed (%s), falling back to retrieve", e)
            return fallback

    def read_metadata(self, paper: dict[str, Any]) -> dict[str, Any]:
        """
        Return the bibliographic metadata dict for a paper (no LLM call).
        Accepts the raw paper dict as stored in memory_store.
        """
        meta = paper.get("metadata") or {}
        return {
            "id": paper.get("id"),
            "title": meta.get("title") or paper.get("title"),
            "authors": meta.get("authors"),
            "year": meta.get("year"),
            "venue": meta.get("venue"),
            "abstract": meta.get("abstract"),
            "keywords": meta.get("keywords"),
            "description": meta.get("description"),
        }

    async def retrieve(
        self,
        query: str,
        paper_id: str | None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Embed query → Qdrant cosine search (top 50) → cross-encoder rerank → top_k chunks.
        Delegates to the infrastructure services.
        """
        from app.services.embedding_service import embed_texts
        from app.services.qdrant_service import search_chunks
        from app.services.reranker_service import rerank

        vectors = await embed_texts([query])
        query_vector = vectors[0]
        candidates = await search_chunks(
            query_vector=query_vector,
            paper_id=paper_id,
            top_k=50,
        )
        return rerank(query=query, chunks=candidates, top_k=top_k)
