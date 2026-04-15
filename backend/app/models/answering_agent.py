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

from app.config import settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

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
