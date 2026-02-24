"""
OpenRouter service: vision extraction (text + tables) and answer generation.
"""
import base64
import re
from typing import List, Dict, Any

from openai import AsyncOpenAI

from app.config import settings

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """You are a document extraction assistant. Extract ALL content from the provided page image(s).

Output your response in exactly this format:

===TEXT===
<all prose, headings, captions, footnotes, and any non-table text verbatim>

===TABLES===
<each table as GitHub-flavored Markdown, separated by a blank line; if no tables write NONE>

Rules:
- Preserve the reading order of the text.
- Do not summarize or paraphrase — extract verbatim.
- For tables: include header rows and alignment markers.
- If content spans two pages, treat them as continuous."""

TABLE_SUMMARY_PROMPT = (
    "Summarize the following table in 2-3 sentences. "
    "Describe what the table is about and highlight key values or trends.\n\n"
    "{markdown_table}"
)

ANSWER_SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer the user's question based solely on "
    "the provided context from research papers. Cite sources using [1], [2], etc. "
    "If the context does not contain enough information, say so honestly."
)

ANSWER_USER_PROMPT = "Context:\n\n{context_block}\n\nQuestion: {question}"

# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _parse_extraction(raw: str) -> Dict[str, Any]:
    """Parse ===TEXT=== / ===TABLES=== delimited output."""
    text_match = re.search(r"===TEXT===(.*?)(?===TABLES===|$)", raw, re.DOTALL)
    tables_match = re.search(r"===TABLES===(.*?)$", raw, re.DOTALL)

    text = text_match.group(1).strip() if text_match else raw.strip()

    tables = []
    if tables_match:
        raw_tables = tables_match.group(1).strip()
        if raw_tables.upper() != "NONE":
            for block in re.split(r"\n{2,}", raw_tables):
                block = block.strip()
                if block and "|" in block:
                    tables.append(block)

    return {"text": text, "tables": tables}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_page_content(image_paths: List[str]) -> Dict[str, Any]:
    """
    Single vision call to extract text + tables from 1 or 2 page images.
    Returns {"text": str, "tables": [markdown_str, ...]}.
    """
    client = _get_client()

    content = [{"type": "text", "text": EXTRACTION_PROMPT}]
    for path in image_paths:
        b64 = _encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}",
                        "detail": "high"},
        })

    response = await client.chat.completions.create(
        model=settings.OPENROUTER_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=4096,
    )
    raw = response.choices[0].message.content or ""
    return _parse_extraction(raw)


async def summarize_table(markdown_table: str) -> str:
    """Summarize a Markdown table into a descriptive sentence for embedding."""
    client = _get_client()
    response = await client.chat.completions.create(
        model=settings.OPENROUTER_TEXT_MODEL,
        messages=[{
            "role": "user",
            "content": TABLE_SUMMARY_PROMPT.format(markdown_table=markdown_table),
        }],
        max_tokens=256,
    )
    return (response.choices[0].message.content or "").strip()


async def generate_answer(question: str, results: List[Dict[str, Any]]) -> str:
    """Generate an answer using retrieved text chunks and/or tables as context."""
    client = _get_client()

    context_parts = []
    for i, r in enumerate(results, 1):
        source = f"[{i}] {r.get('paper_title', 'Unknown')} (page {r.get('page_num', '?')})"
        if r.get("type") == "table":
            context_parts.append(f"{source} — Table:\n{r.get('markdown_table', '')}")
        else:
            # context_window = pages N-1, N, N+1 fetched at query time
            context = r.get("context_window") or r.get("page_text") or r.get("content", "")
            context_parts.append(f"{source}:\n{context}")

    context_block = "\n\n---\n\n".join(context_parts)

    # Debug: log each source passed to the LLM
    import logging
    logger = logging.getLogger(__name__)
    logger.debug("=== generate_answer sources ===")
    for i, r in enumerate(results, 1):
        rtype = r.get("type", "?")
        page = r.get("page_num", "?")
        title = r.get("paper_title", "Unknown")
        score = round(r.get("score", 0), 4)
        if rtype == "table":
            chars = len(r.get("markdown_table", ""))
        else:
            chars = len(r.get("context_window") or r.get("page_text") or r.get("content", ""))
        logger.debug(f"  [{i}] {rtype:5s} | page {page} | score {score} | {chars} chars | {title}")
    logger.debug("================================")

    response = await client.chat.completions.create(
        model=settings.OPENROUTER_TEXT_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM_PROMPT},
            {"role": "user", "content": ANSWER_USER_PROMPT.format(
                context_block=context_block,
                question=question,
            )},
        ],
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()
