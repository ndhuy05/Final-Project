"""
OpenRouter service: vision-based page extraction and answer generation.
"""
import base64
import logging
from typing import List, Dict, Any

from openai import AsyncOpenAI

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = (
    "You are an expert document extraction assistant. Your task is to extract ALL content from the provided page image with maximum accuracy and completeness.\n\n"
    
    "## General Rules\n"
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
    "Output raw extracted content only. Do not add commentary, explanations, or any text that does not appear in the original document."
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def extract_page_content(image_paths: List[str]) -> str:
    """
    Single vision call to extract page content as plain text.
    Tables are described in prose rather than rendered as Markdown.
    Returns the extracted text string.
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
    return (response.choices[0].message.content or "").strip()


async def generate_answer_with_images(
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

    logger.debug("=== generate_answer_with_images ===")
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