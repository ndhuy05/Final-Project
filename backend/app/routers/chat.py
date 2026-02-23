"""
Chat router: embed user query with ColQwen2, retrieve relevant pages from Qdrant,
generate a direct answer with Qwen2.5-1.5B-Instruct, return answer with citations.
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

from app.services import colpali_service, qdrant_service, memory_store, llm_service

router = APIRouter()


class ChatRequest(BaseModel):
    question: str
    top_k: int = 3


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

    # Embed the query with ColQwen2
    query_vector = colpali_service.embed_query(request.question)

    # Search Qdrant for most relevant pages
    try:
        results = qdrant_service.search(
            notebook_id=notebook_id,
            query_multi_vector=query_vector,
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

    # Build citations (no text excerpt — VLM reads images directly)
    citations = []
    for i, result in enumerate(results):
        citations.append(Citation(
            id=i + 1,
            title=result.get("paper_title", "Unknown"),
            page=result.get("page_num", 0),
            excerpt=f"Page {result.get('page_num', '?')} of {result.get('paper_title', 'Unknown')}",
            score=round(result.get("score", 0), 4),
        ))

    # Generate answer with LLM using retrieved pages as context
    try:
        content = llm_service.generate_answer(request.question, results)
    except Exception as e:
        # Fallback: return raw excerpts if LLM fails
        excerpts = "\n\n".join(
            f"**[{c.id}] {c.title} — Page {c.page}**\n{c.excerpt}"
            for c in citations
        )
        content = (
            f"Based on the most relevant pages found:\n\n{excerpts}\n\n"
            f"*(Note: AI answer generation unavailable — {str(e)[:100]})*"
        )

    return ChatResponse(content=content, citations=citations)
