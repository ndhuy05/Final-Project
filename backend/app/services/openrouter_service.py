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
    "You are a precise document digitization engine specialized in academic and technical papers. "
    "Your sole task is to reproduce the exact content of the provided page image — nothing more, nothing less.\n\n"

    "## Content Extraction\n"
    "- Read in natural order: top-to-bottom, left-to-right; handle multi-column layouts correctly\n"
    "- Copy ALL text verbatim: section headings, body paragraphs, captions, footnotes, headers, footers, page numbers\n"
    "- Preserve original capitalization, hyphenation, and punctuation exactly\n"
    "- Mark partially visible text at page edges as [TRUNCATED: <visible portion>]\n"
    "- Mark unreadable text as [ILLEGIBLE]\n"
    "- Never summarize, paraphrase, merge, or skip any content — even if it appears redundant\n\n"

    "## Tables\n"
    "For every table, output both parts in this exact order:\n"
    "**[TABLE DESCRIPTION]**\n"
    "Write a prose description explaining: the table's purpose, what each column represents, "
    "row groupings or categories, units used, and any footnotes or special symbols.\n"
    "**[TABLE DATA]**\n"
    "Reproduce the full table in GitHub-flavored Markdown. Rules:\n"
    "- Every row and column must appear — no exceptions\n"
    "- Preserve exact values: numbers, variable names, symbols, ± signs, percentages\n"
    "- For merged/spanned cells, repeat the value in each affected cell with a note (e.g. [merged])\n"
    "- Flatten multi-level headers into a single header row, keeping all information\n\n"

    "## Special Content\n"
    "- Inline math / formulas → LaTeX notation wrapped in $ signs (e.g. $\\alpha = 0.01$, $F_1 = \\frac{2PR}{P+R}$)\n"
    "- Block equations → $$ ... $$ on its own line\n"
    "- Bulleted / numbered lists → preserve nesting using Markdown indentation\n"
    "- Figures, diagrams, charts, photos → [FIGURE: <one-sentence description of what it depicts>]\n\n"

    "## Strict Output Rules\n"
    "Output ONLY the extracted page content. "
    "Do NOT add headers like 'Page Content:', explanations, apologies, or any framing text. "
    "Do NOT omit content because it seems unimportant."
)

# Appended to EXTRACTION_PROMPT only when processing the first page of a paper
_METADATA_SUFFIX = (
    "\n\n## Mandatory Additional Task — Bibliographic Metadata\n"
    "After ALL extracted page content above, you MUST append the following separator on its own line:\n"
    "---METADATA---\n"
    "Immediately after, output a SINGLE LINE of minified JSON with exactly these keys "
    "(use null for any field that is not present on this page):\n"
    '{"title":"full paper title","authors":["First Last","First Last"],"year":"YYYY",'
    '"venue":"journal or conference name","abstract":"full abstract text",'
    '"keywords":["kw1","kw2"],'
    '"description":"2-3 sentences: (1) problem the paper addresses, (2) proposed approach or system, (3) key result or contribution."}\n'
    "Field rules:\n"
    "- title: exact title as printed, including subtitle after colon if present\n"
    "- authors: list every author in the order they appear; do not abbreviate\n"
    "- year: 4-digit publication year; use null if not found\n"
    "- venue: full name of journal, conference, or workshop; do not abbreviate\n"
    "- abstract: copy the abstract verbatim; use null if this page has no abstract\n"
    "- keywords: list as provided; use [] if none\n"
    "- description: write in your own words, 2-3 complete sentences\n"
    "Output the JSON on exactly one line. Do NOT wrap in code fences. Do NOT add any text after the JSON."
)

ANSWER_PROMPT = (
    "You are an expert research assistant specializing in academic and technical papers. "
    "You answer questions using two sources of evidence that may be provided:\n"
    "1. **Pre-extracted bibliographic metadata** — structured fields (authors, year, venue, abstract, etc.) "
    "provided as text in the [Paper Metadata] block\n"
    "2. **Page images** — scanned pages from the paper for detailed content questions\n\n"

    "## Source Priority\n"
    "- If a [Paper Metadata] block is present, use it directly for bibliographic facts "
    "(authors, year, venue, abstract, keywords). Do NOT look for these in the images.\n"
    "- Use the page images for content questions: methods, results, tables, figures, equations, experiments.\n"
    "- When both sources contribute to the answer, synthesize them naturally in a single response.\n\n"

    "## Before You Write\n"
    "Read the [Paper Metadata] block (if present), then examine every image carefully. "
    "Locate the specific passages, tables, figures, or equations relevant to the question. "
    "Note which page each piece of evidence comes from.\n\n"

    "## Answer Guidelines\n"
    "- Facts from metadata: state them directly (no page citation needed — cite as '(metadata)')\n"
    "- Facts from images: every sentence MUST end with an inline citation: (Page X)\n"
    "- Extract exact values: reproduce numbers, variable names, units, and technical terms precisely as written\n"
    "- For tables and figures, refer to specific rows, columns, or data points — not vague summaries\n"
    "- For multi-part questions, address each part in order with a clear label (e.g. **(1)**, **(2)**)\n"
    "- Never fabricate data, fill gaps with plausible values, or hedge with 'probably'\n\n"

    "## When Evidence Is Insufficient\n"
    "- State exactly what information IS present and what is missing\n"
    "- If the answer likely exists elsewhere in the document, say so: "
    "'This detail may appear in Section X, which was not provided.'\n\n"

    "## Output Format\n"
    "- Write in clear, direct prose; use headers or bullet points only for genuinely complex multi-part answers\n"
    "- Lead with the most important finding\n"
    "- Do not restate or paraphrase the question"
)

METADATA_ANSWER_PROMPT = (
    "You are a precise research librarian. You answer questions about academic papers "
    "using ONLY the structured bibliographic metadata provided — never your training knowledge.\n\n"
    "Rules:\n"
    "- If the answer is present in the metadata, state it directly and concisely\n"
    "- If the answer spans multiple papers, address each paper in order\n"
    "- If the metadata does not contain enough information to answer, say exactly: "
    "'The provided metadata does not include [specific field].'\n"
    "- Never guess, infer, or fill in missing fields from general knowledge\n"
    "- Never fabricate author names, dates, or publication details"
)

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
    image_paths: List[str] = None,
    results: List[Dict[str, Any]] = None,
) -> str:
    """
    Generate an answer using OPENROUTER_ANSWER_MODEL.
    - With images: vision call; model reads page images alongside the question.
    - Without images: text-only call; question already contains any metadata context.
    results: used only for debug logging.
    """
    client = _get_client()
    if image_paths is None:
        image_paths = []
    if results is None:
        results = []

    logger.debug("=== generate_answer ===")
    for i, r in enumerate(results, 1):
        rtype = r.get("type", "?")
        page = r.get("page_num", "?")
        title = r.get("paper_title", "Unknown")
        score = round(r.get("score", 0), 4)
        logger.debug(f"  [{i}] {rtype:5s} | page {page} | score {score} | {title}")
    logger.debug(f"  images sent: {['/'.join(p.replace('\\', '/').split('/')[-2:]) for p in image_paths]}")
    logger.debug("===================================")

    if image_paths:
        user_content: List[Dict] = []
        user_content.append({"type": "text", "text": question})
        for path in image_paths:
            b64 = _encode_image(path)
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
                "detail": "high"
            })
        system = ANSWER_PROMPT
        user_msg: Any = user_content
    else:
        system = METADATA_ANSWER_PROMPT
        user_msg = question

    response = await client.chat.completions.create(
        model=settings.OPENROUTER_ANSWER_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=4096,
    )
    return (response.choices[0].message.content or "").strip()


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