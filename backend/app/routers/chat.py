"""
Chat router: embed query with fastembed, retrieve relevant chunks
from Qdrant, collect page images (N-1, N, N+1), generate answer via VLM.
"""
import logging
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List

from app.config import settings
from app.services import qdrant_service, memory_store, embedding_service, openrouter_service, reranker_service

logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    top_k: int = 50


class Citation(BaseModel):
    id: int
    title: str
    page: int
    excerpt: str
    score: float


class ChatResponse(BaseModel):
    content: str
    citations: List[Citation]


@router.post("/notebooks/{notebook_id}/chat", response_model=ChatResponse)
async def chat(notebook_id: str, request: ChatRequest):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    papers = memory_store.get_papers(notebook_id)
    if not papers:
        return ChatResponse(
            content="No papers have been uploaded to this notebook yet. Please upload a PDF first.",
            citations=[],
        )

    # Embed the query
    query_vector = embedding_service.embed_text(request.question)

    # Search Qdrant for the most relevant chunks
    try:
        results = qdrant_service.search(
            notebook_id=notebook_id,
            query_vector=query_vector,
            top_k=request.top_k,
        )
    except Exception:
        return ChatResponse(
            content="No indexed content found. Please upload a PDF to this notebook first.",
            citations=[],
        )

    if not results:
        return ChatResponse(
            content="I couldn't find relevant content for your question in the uploaded papers.",
            citations=[],
        )

    # Rerank results with cross-encoder before dedup
    try:
        results = reranker_service.rerank(request.question, results, top_k=5)
    except Exception as e:
        logger.warning("Reranking failed, using Qdrant scores: %s", e)

    # Filter out empty/broken points and deduplicate by (paper_id, page_num, type)
    seen: dict = {}
    for r in results:
        has_content = bool(r.get("page_text") or r.get("content") or r.get("markdown_table"))
        if not has_content:
            continue
        key = (r.get("paper_id"), r.get("page_num"), r.get("type"))
        if key not in seen or r.get("score", 0) > seen[key].get("score", 0):
            seen[key] = r
    results = list(seen.values())

    # Use only the top (highest score) result to determine the 3 pages to send
    top = results[0]
    paper_id = top.get("paper_id")
    page = top.get("page_num")  # 1-based

    image_paths = []
    for p in [page - 1, page, page + 1]:
        if p < 1:
            continue
        img_path = os.path.join(settings.IMAGE_DIR, paper_id, f"page_{p}.png")
        if os.path.exists(img_path):
            image_paths.append(img_path)

    # Build citations (type-aware)
    citations = []
    for i, result in enumerate(results):
        page_num = result.get("page_num", 0)
        title = result.get("paper_title", "Unknown")
        if result.get("type") == "table":
            table_idx = result.get("table_index", 1)
            excerpt = f"Table {table_idx} on page {page_num} of {title}"
        else:
            excerpt = f"Page {page_num} of {title}"
        citations.append(Citation(
            id=i + 1,
            title=title,
            page=page_num,
            excerpt=excerpt,
            score=round(result.get("score", 0), 4),
        ))

    # Generate answer: VLM reads page images directly
    try:
        if image_paths:
            content = await openrouter_service.generate_answer_with_images(
                request.question, image_paths, results
            )
        else:
            content = "No page images found for the matched content. Please re-upload the paper."
    except Exception as e:
        listing = "\n\n".join(f"**[{c.id}] {c.excerpt}**" for c in citations)
        content = (
            f"Based on the most relevant content found:\n\n{listing}\n\n"
            f"*(Note: AI answer generation unavailable — {str(e)[:100]})*"
        )

    return ChatResponse(content=content, citations=citations)
