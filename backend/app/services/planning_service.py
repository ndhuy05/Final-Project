"""
OpenRouter planning service: retrieval action planning.
"""
import json
import logging
import re
from typing import List, Dict, Any

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

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

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


def _extract_json_array(raw: str) -> list:
    """
    Robustly extract a JSON array from a model response.
    Same 3-tier fallback as _extract_json but targets [...] instead of {...}.
    """
    text = raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    if "```" in text:
        parts = text.split("```")
        for part in parts:
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


async def plan_actions(question: str, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Plan a list of retrieval actions to answer the question.

    Returns a list of action dicts:
      {"action": "read_metadata", "paper_id": str | None}
      {"action": "retrieve",      "paper_id": str | None, "query": str}

    Falls back to [{"action": "retrieve", "paper_id": None, "query": question}] on any error.
    """
    client = _get_client()
    fallback = [{"action": "retrieve", "paper_id": None, "query": question}]

    # Build papers context: id + title + description
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
        response = await client.chat.completions.create(
            model=settings.OPENROUTER_PLANNER_MODEL,
            messages=[
                {"role": "system", "content": PLANNER_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=512,
        )
        raw = (response.choices[0].message.content or "").strip()
        logger.debug("Planner raw response: %s", raw[:300])

        result = _extract_json_array(raw)
        if not isinstance(result, list) or not result:
            return fallback

        # Validate and clean up each action
        valid = []
        for action in result:
            if not isinstance(action, dict):
                continue
            if action.get("action") not in ("read_metadata", "retrieve"):
                continue
            if action["action"] == "retrieve" and not action.get("query"):
                action["query"] = question
            valid.append(action)

        logger.debug("Planner actions for '%s': %s", question[:60], valid)
        return valid if valid else fallback
    except Exception as e:
        logger.warning("Planning failed (%s), falling back to retrieve", e)
        return fallback
