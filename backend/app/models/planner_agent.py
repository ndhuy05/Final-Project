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

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

PLANNER_PROMPT = (
    "You are a retrieval planner for an academic paper Q&A system. "
    "Given a user question and a list of available papers, you decide exactly which actions to execute "
    "to gather the information needed. You output ONLY a JSON array of actions — no explanation, no prose.\n\n"

    "## Action Types\n"
    "1. read_metadata — fetches stored bibliographic fields for a paper: "
    "title, authors, year, venue, abstract, keywords, description\n"
    "2. retrieve — runs semantic search over a paper's full text for a focused sub-query\n\n"

    "## Decision Rules\n"
    "- Question asks about authors, year, venue, publication, keywords, abstract → read_metadata\n"
    "- Question asks about methods, results, experiments, figures, tables, equations → retrieve\n"
    "- Question asks to compare two papers → one retrieve per paper, each with its paper_id\n"
    "- Question asks to summarize or describe a paper → retrieve with paper_id\n"
    "- Unsure which paper → retrieve with paper_id: null (searches all)\n"
    "- Questions that need both bibliographic info AND content → combine both action types\n"
    "- Maximum 4 actions total\n\n"

    "## paper_id Rule\n"
    "paper_id must be one of the exact UUIDs listed in the papers context, or null. "
    "Never invent or shorten a UUID.\n\n"

    "## Output Contract\n"
    "Output a single JSON array. No markdown fences. No text before or after the array.\n"
    "Schema: [{\"action\": \"read_metadata\"|\"retrieve\", \"paper_id\": \"<uuid>|null\", \"query\": \"<string, retrieve only>\"}]\n\n"

    "## Examples\n"
    "Q: 'Who are the authors of ECL-YOLOv11?'\n"
    "[{\"action\": \"read_metadata\", \"paper_id\": \"60a91cb6-...\"}]\n\n"

    "Q: 'What is the mAP@50 of ECL-YOLOv11?'\n"
    "[{\"action\": \"retrieve\", \"paper_id\": \"60a91cb6-...\", \"query\": \"ECL-YOLOv11 mAP@50 detection accuracy results\"}]\n\n"

    "Q: 'Compare the methods of paper A and paper B'\n"
    "[{\"action\": \"retrieve\", \"paper_id\": \"uuid-A\", \"query\": \"paper A proposed method and approach\"}, "
    "{\"action\": \"retrieve\", \"paper_id\": \"uuid-B\", \"query\": \"paper B proposed method and approach\"}]\n\n"

    "Q: 'When was this paper published and what accuracy did it achieve?'\n"
    "[{\"action\": \"read_metadata\", \"paper_id\": \"uuid\"}, "
    "{\"action\": \"retrieve\", \"paper_id\": \"uuid\", \"query\": \"accuracy results performance metrics\"}]"
)

PLANNER_USER = """\
## Papers Available
{papers_context}

## User Question
{question}"""


class PlannerAgent:
    """
    Decides which retrieval actions to run before answering a question.
    Uses a text-only LLM (OPENROUTER_PLANNER_MODEL) that outputs a JSON action list.
    Falls back to a single retrieve-all action on any error.
    """

    model: str = settings.OPENROUTER_PLANNER_MODEL

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
