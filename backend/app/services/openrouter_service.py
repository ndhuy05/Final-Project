"""
OpenRouter service: vision-based page extraction, answer generation,
metadata extraction, and query routing/decomposition.
"""
import base64
import json
import logging
import re
from typing import List, Dict, Any, Tuple

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = (
    "You are an expert document extraction assistant. Extract ALL content from the provided page image with maximum accuracy and completeness.\n\n"

    "## Extraction Rules\n"
    "- Extract content in natural reading order (top-to-bottom, left-to-right, multi-column aware)\n"
    "- Preserve ALL text verbatim: headings, subheadings, body text, captions, footnotes, headers, footers\n"
    "- Do NOT summarize, paraphrase, or omit any content — even if it seems repetitive\n"
    "- If text is partially cut off at page edges, extract what is visible and mark with [TRUNCATED]\n"
    "- If any content is unclear or illegible, mark with [ILLEGIBLE]\n\n"

    "## Tables\n"
    "When you encounter a table, output the following in order:\n"
    "1. **[TABLE DESCRIPTION]**: A detailed prose description covering:\n"
    "   - The table's purpose and what it represents\n"
    "   - Column headers and their meaning\n"
    "   - Row structure and groupings (if any)\n"
    "   - Any units, footnotes, or special notations used\n"
    "2. **[TABLE DATA]**: The complete table in Markdown format\n"
    "   - Include ALL rows and columns without exception\n"
    "   - Preserve exact numbers, variables, symbols, and units\n"
    "   - For merged cells, repeat the value in each affected cell and note the merge\n"
    "   - For multi-level headers, flatten them clearly\n\n"

    "## Formulas & Special Content\n"
    "- Mathematical formulas: render in LaTeX inline notation (e.g. $E = mc^2$)\n"
    "- Lists and bullet points: preserve hierarchy and indentation using Markdown\n"
    "- Images/charts/diagrams: mark with [FIGURE: <brief description of what it shows>]\n\n"

    "## Output Format\n"
    "Output raw extracted content only. Do not add commentary or any text not in the original document."
)

# Appended to EXTRACTION_PROMPT only when processing the first page of a paper
_METADATA_SUFFIX = (
    "\n\n## Additional Task (First Page Only)\n"
    "After ALL extracted page content, append exactly:\n"
    "---METADATA---\n"
    "Then on the next line output a single-line JSON object with these keys (use null for missing):\n"
    '{"title":"...","authors":[...],"publish_date":"...","venue":"...","description":"..."}\n'
    "description: 4-5 sentences covering (1) what problem the paper addresses, "
    "(2) what approach or system it proposes, (3) its key result or contribution.\n"
    "Output the JSON on one line only. Do not wrap in code fences."
)

ANSWER_WITH_IMAGES_PROMPT = (
    "You are an expert research assistant specialized in analyzing academic and technical documents.\n\n"

    "## Your Task\n"
    "Answer the user's question using ONLY the information visible in the provided page image(s). "
    "Do not use prior knowledge to fill in gaps — if the pages lack sufficient information, say so explicitly.\n\n"

    "## Answer Guidelines\n"
    "- Be precise and thorough: extract exact numbers, statistics, variable names, and technical terms as they appear\n"
    "- For questions involving tables or figures, describe the relevant data specifically rather than generally\n"
    "- If multiple pages contribute to the answer, synthesize them coherently\n"
    "- If the question has multiple sub-parts, address each one separately\n"
    "- Do not speculate or infer beyond what is explicitly stated in the pages\n\n"

    "## Citation Format\n"
    "Always cite your sources inline using the format (Page X). Example:\n"
    "  'The model achieves 94.2% accuracy on the test set (Page 5), "
    "which the authors attribute to the attention mechanism described in the methodology (Page 3).'\n\n"

    "## When Information Is Insufficient\n"
    "If the provided pages do not contain enough information:\n"
    "- State clearly what IS available and what is missing\n"
    "- Indicate whether the answer might be found elsewhere in the document (e.g. 'This may be defined in an earlier section not provided')\n"
    "- Never fabricate or guess data\n\n"

    "## Output Format\n"
    "- Use clear, structured prose\n"
    "- For complex answers, use headers or bullet points to organize information\n"
    "- Keep your answer focused and avoid restating the question"
)

METADATA_ANSWER_PROMPT = """\
You are a research assistant. Answer the user's question using ONLY the bibliographic metadata provided below.
Be concise and direct. If the metadata does not contain the answer, say so explicitly — do not guess.

## Paper Metadata
{metadata_context}

## Question
{question}"""

PLANNER_PROMPT = """\
You are a research assistant planning how to answer a question about academic papers.

## Available Actions
- read_metadata: retrieve stored bibliographic info for a paper (title, authors, publish_date, venue, description)
- retrieve: run semantic search over a paper's content for a specific sub-query

## Papers Available
{papers_context}

## Rules
- For questions about authors, publish_date, venue, or paper description → use read_metadata
- For questions about paper content, methods, results, experiments → use retrieve with a focused sub-query
- For comparison questions → use retrieve for each relevant paper separately with its paper_id
- paper_id: null in retrieve = search across all papers (use when unsure which paper)
- Combine actions freely (e.g. read_metadata + retrieve together)
- Maximum 4 actions total
- paper_id must exactly match an ID from the list above, or null

## Output (JSON array only, no explanation)
[{{"action": "read_metadata", "paper_id": "uuid-or-null"}}, {{"action": "retrieve", "paper_id": "uuid-or-null", "query": "focused sub-query"}}]

## User Question
{question}"""

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

def _extract_json(raw: str) -> dict:
    """
    Robustly extract a JSON object from a model response.
    Tries in order:
      1. Direct parse
      2. Strip markdown code fences then parse
      3. Regex: find first {...} block in the string
    Raises json.JSONDecodeError if all attempts fail.
    """
    text = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Strip code fences
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

    # 3. Regex: grab first { ... } block (handles extra prose before/after JSON)
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group())

    raise json.JSONDecodeError("No JSON object found", text, 0)


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


def _parse_batch_response(
    raw: str,
    pages: List[Tuple[int, str]],
    extract_metadata: bool,
) -> List[Dict[str, Any]]:
    """
    Parse the VLM response for a batch of pages (currently always size 1).
    Strips ---METADATA--- block from the end when extract_metadata=True.
    Returns [{"page_num": int, "text": str, "metadata": dict}].
    """
    # Strip metadata block (appended at end of response for first page)
    global_metadata: Dict[str, Any] = {}
    if extract_metadata and "---METADATA---" in raw:
        parts = raw.split("---METADATA---", 1)
        raw = parts[0].strip()
        try:
            global_metadata = _extract_json(parts[1].strip())
        except Exception as e:
            logger.warning("Failed to parse metadata JSON: %s", e)

    results = []
    for i, (page_num, _) in enumerate(pages):
        results.append({
            "page_num": page_num,
            "text": raw if i == 0 else "",   # single-page: full response goes to first (only) page
            "metadata": global_metadata if (i == 0 and extract_metadata) else {},
        })

    return results


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_page_content(
    pages: List[Tuple[int, str]],
    extract_metadata: bool = False,
) -> List[Dict[str, Any]]:
    """
    Batch vision call to extract content from one or more page images in a single API call.

    pages: list of (0-based page_num, image_path) pairs.
    extract_metadata: set True when the batch contains page 0 (first page of a paper).
                      The model will append ---METADATA--- JSON at the end.

    Returns a list of {"page_num": int, "text": str, "metadata": dict} dicts,
    one per input page, in the same order.
    metadata is populated only for the first page entry when extract_metadata=True.
    """
    client = _get_client()
    prompt = EXTRACTION_PROMPT + (_METADATA_SUFFIX if extract_metadata else "")

    content: List[Dict] = [{"type": "text", "text": prompt}]
    for _, path in pages:
        b64 = _encode_image(path)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"},
        })

    # Scale max_tokens with batch size (up to 8192)
    max_tokens = min(4096 * len(pages), 8192)

    response = await client.chat.completions.create(
        model=settings.OPENROUTER_VISION_MODEL,
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
    )
    raw = (response.choices[0].message.content or "").strip()
    return _parse_batch_response(raw, pages, extract_metadata)


async def generate_answer(
    question: str,
    image_paths: List[str],
    results: List[Dict[str, Any]],
) -> str:
    """
    Generate an answer by passing page images directly to a VLM.
    image_paths: ordered list of page image files (deduplicated, sorted by page).
    results: used only for debug logging.
    """
    client = _get_client()

    logger.debug("=== generate_answer ===")
    for i, r in enumerate(results, 1):
        rtype = r.get("type", "?")
        page = r.get("page_num", "?")
        title = r.get("paper_title", "Unknown")
        score = round(r.get("score", 0), 4)
        logger.debug(f"  [{i}] {rtype:5s} | page {page} | score {score} | {title}")
    logger.debug(f"  images sent: {[p.replace('\\', '/').split('/')[-1] for p in image_paths]}")
    logger.debug("===================================")

    user_content: List[Dict] = []
    for path in image_paths:
        b64 = _encode_image(path)
        user_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64}"},
            "detail": "high"
        })
    user_content.append({"type": "text", "text": f"Question: {question}"})

    response = await client.chat.completions.create(
        model=settings.OPENROUTER_ANSWER_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_WITH_IMAGES_PROMPT},
            {"role": "user", "content": user_content},
        ],
        max_tokens=1024,
    )
    return (response.choices[0].message.content or "").strip()


async def generate_metadata_answer(question: str, papers: List[Dict[str, Any]]) -> str:
    """
    Answer a metadata question using only the structured metadata already
    extracted from the papers. Uses the router model (text-only, no images).
    Falls back to a plain formatted dump if the LLM call fails.
    """
    client = _get_client()

    # Build a readable metadata context block for each paper
    sections = []
    for paper in papers:
        meta = paper.get("metadata") or {}
        title = meta.get("title") or paper.get("title", "Unknown")
        lines = [f"Title: {title}"]
        authors = meta.get("authors")
        if authors:
            lines.append(f"Authors: {', '.join(authors)}")
        if meta.get("year"):
            lines.append(f"Year: {meta['year']}")
        if meta.get("venue"):
            lines.append(f"Venue: {meta['venue']}")
        if meta.get("abstract"):
            lines.append(f"Abstract: {meta['abstract']}")
        if meta.get("keywords"):
            lines.append(f"Keywords: {', '.join(meta['keywords'])}")
        if meta.get("description"):
            lines.append(f"Description: {meta['description']}")
        sections.append("\n".join(lines))

    metadata_context = "\n\n---\n\n".join(sections) if sections else "(no metadata available)"
    prompt = METADATA_ANSWER_PROMPT.format(
        metadata_context=metadata_context,
        question=question,
    )

    try:
        response = await client.chat.completions.create(
            model=settings.OPENROUTER_ROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
        )
        return (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("Metadata answer generation failed: %s", e)
        return metadata_context  # fallback: return raw metadata text


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
        prompt = PLANNER_PROMPT.format(papers_context=papers_context, question=question)
        response = await client.chat.completions.create(
            model=settings.OPENROUTER_ROUTER_MODEL,
            messages=[{"role": "user", "content": prompt}],
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