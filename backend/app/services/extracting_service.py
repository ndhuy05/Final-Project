"""
OpenRouter extraction service: vision-based page extraction and metadata extraction.
"""
import base64
import json
import logging
import re
from typing import List, Dict, Any, Tuple

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

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

_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    return _client


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
