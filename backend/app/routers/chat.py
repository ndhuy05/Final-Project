"""
Chat router: agentic Q&A pipeline.
A planner LLM decides which actions to run (read_metadata / retrieve),
actions execute in parallel, then a VLM generates the final answer.
All turns are persisted to the notebook's single ChatSession.
"""
import asyncio
import json
import logging
import os
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from typing import List, Tuple
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.notebook import Notebook
from app.models.chat import ChatSession
from app.models.user import User
from app.services import (
    qdrant_service, memory_store, embedding_service,
    reranker_service,
)
from app.models import PlannerAgent, AnsweringAgent
from app.services.auth_service import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

_planner = PlannerAgent()
_answering = AnsweringAgent()


def _require_notebook(notebook_id: str, current_user: User, db: Session) -> Notebook:
    """Return the Notebook owned by current_user or raise HTTP 404."""
    notebook = (
        db.query(Notebook)
        .filter(
            Notebook.id == notebook_id,
            Notebook.user_id == current_user.id,
            Notebook.deleted_at.is_(None),
        )
        .first()
    )
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found.")
    return notebook


def _get_or_create_session(notebook: Notebook, db: Session) -> ChatSession:
    """Return the notebook's single ChatSession, creating it if it doesn't exist yet."""
    session = notebook.get_chat_session()
    if session is None:
        session = ChatSession(notebook_id=notebook.id)
        db.add(session)
        db.flush()  # assign session.id without committing the transaction
    return session


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
    query_type: str = "retrieve"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_images(paper_id: str, page: int) -> List[str]:
    """Return existing image paths for pages N-1, N, N+1."""
    paths = []
    for p in [page - 1, page, page + 1]:
        if p < 1:
            continue
        img_path = os.path.join(settings.IMAGE_DIR, paper_id, f"page_{p}.png")
        if os.path.exists(img_path):
            paths.append(img_path)
    return paths


def _make_citations(results: List[dict]) -> List[Citation]:
    citations = []
    for i, r in enumerate(results):
        page_num = r.get("page_num", 0)
        title = r.get("paper_title", "Unknown")
        citations.append(Citation(
            id=i + 1,
            title=title,
            page=page_num,
            excerpt=f"Page {page_num} of {title}",
            score=round(r.get("score", 0), 4),
        ))
    return citations


async def _qdrant_rerank(
    question: str, notebook_id: str, top_k: int, paper_id: str = None
) -> List[dict]:
    """Embed → Qdrant search (optionally scoped to paper_id) → rerank → dedup."""
    query_vector = await embedding_service.embed_text(question)
    results = qdrant_service.search(
        notebook_id=notebook_id,
        query_vector=query_vector,
        top_k=top_k,
        paper_id=paper_id,
    )
    if not results:
        return []
    try:
        results = reranker_service.rerank(question, results, top_k=5)
    except Exception as e:
        logger.warning("Reranking failed: %s", e)

    seen: dict = {}
    for r in results:
        if not (r.get("page_text") or r.get("content")):
            continue
        key = (r.get("paper_id"), r.get("page_num"), r.get("type"))
        if key not in seen or r.get("score", 0) > seen[key].get("score", 0):
            seen[key] = r
    return list(seen.values())


# ---------------------------------------------------------------------------
# Action executor
# ---------------------------------------------------------------------------

async def _execute_actions(
    actions: List[dict],
    question: str,
    notebook_id: str,
    top_k: int,
    papers: List[dict],
) -> Tuple[List[dict], List[dict]]:
    """
    Execute all planner actions in parallel.
    Returns (metadata_papers, retrieved_results).
    """
    async def _run_one(action: dict):
        act = action.get("action")
        paper_id = action.get("paper_id") or None

        if act == "read_metadata":
            if paper_id:
                targets = [p for p in papers if p["id"] == paper_id]
            else:
                targets = papers
            return ("metadata", targets)

        elif act == "retrieve":
            query = action.get("query") or question
            results = await _qdrant_rerank(query, notebook_id, top_k, paper_id=paper_id)
            return ("retrieve", results)

        return None

    raw = await asyncio.gather(*[_run_one(a) for a in actions], return_exceptions=True)

    metadata_papers: List[dict] = []
    retrieved: List[dict] = []
    for r in raw:
        if isinstance(r, Exception):
            logger.warning("Action failed: %s", r)
            continue
        if r is None:
            continue
        kind, data = r
        if kind == "metadata":
            for p in data:
                if p not in metadata_papers:
                    metadata_papers.append(p)
        elif kind == "retrieve":
            retrieved.extend(data)

    return metadata_papers, retrieved


# ---------------------------------------------------------------------------
# Endpoint
# ---------------------------------------------------------------------------

class MessageOut(BaseModel):
    role: str
    content: str
    citations: List[Citation]


class HistoryResponse(BaseModel):
    messages: List[MessageOut]


@router.get("/notebooks/{notebook_id}/chat/history", response_model=HistoryResponse)
async def get_chat_history(
    notebook_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return the full ordered chat history for a notebook's session."""
    notebook = _require_notebook(notebook_id, current_user, db)
    session = notebook.get_chat_session()
    if session is None:
        return HistoryResponse(messages=[])
    messages: List[MessageOut] = []
    for msg in session.get_history():
        citations: List[Citation] = []
        if msg.citations_json:
            try:
                raw = json.loads(msg.citations_json)
                citations = [Citation(**c) for c in raw]
            except Exception:
                logger.warning("Failed to parse citations_json for message %s", msg.id)
        messages.append(MessageOut(role=msg.role, content=msg.content, citations=citations))
    return HistoryResponse(messages=messages)


@router.post("/notebooks/{notebook_id}/chat", response_model=ChatResponse)
async def chat(
    notebook_id: str,
    request: ChatRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    notebook = _require_notebook(notebook_id, current_user, db)
    papers = memory_store.get_papers(notebook_id)
    if not papers:
        return ChatResponse(
            content="No papers have been uploaded to this notebook yet. Please upload a PDF first.",
            citations=[],
            query_type="none",
        )

    # Plan actions
    actions = await _planner.plan_actions(request.question, papers)
    action_types = list({a.get("action", "retrieve") for a in actions})
    logger.debug("Actions planned: %s", actions)

    try:
        metadata_papers, retrieved = await _execute_actions(
            actions, request.question, notebook_id, request.top_k, papers
        )

        # Dedup retrieved results by (paper_id, page_num)
        seen_pages: dict = {}
        for r in retrieved:
            key = (r.get("paper_id"), r.get("page_num"))
            if key not in seen_pages or r.get("score", 0) > seen_pages[key].get("score", 0):
                seen_pages[key] = r
        top_results = list(seen_pages.values())

        # Collect images: best result per retrieve-target paper (N-1, N, N+1 each)
        image_paths: List[str] = []
        if top_results:
            retrieve_paper_ids = [
                a.get("paper_id") for a in actions
                if a.get("action") == "retrieve" and a.get("paper_id")
            ]
            if len(retrieve_paper_ids) > 1:
                # Multi-paper comparison: pick best result from each targeted paper
                best_per_paper: dict = {}
                for r in top_results:
                    pid = r.get("paper_id")
                    if pid not in best_per_paper or r.get("score", 0) > best_per_paper[pid].get("score", 0):
                        best_per_paper[pid] = r
                for r in best_per_paper.values():
                    for p in _collect_images(r.get("paper_id"), r.get("page_num")):
                        if p not in image_paths:
                            image_paths.append(p)
            else:
                best = max(top_results, key=lambda r: r.get("score", 0))
                image_paths = _collect_images(best.get("paper_id"), best.get("page_num"))

        citations = _make_citations(top_results)

        # Build question with metadata context if read_metadata was used
        question_with_context = request.question
        if metadata_papers:
            meta_lines = []
            for p in metadata_papers:
                meta = p.get("metadata") or {}
                title = meta.get("title") or p.get("title", "Unknown")
                parts = [f"Paper: {title}"]
                if meta.get("authors"):
                    parts.append(f"Authors: {', '.join(meta['authors'])}")
                if meta.get("year"):
                    parts.append(f"Year: {meta['year']}")
                if meta.get("venue"):
                    parts.append(f"Venue: {meta['venue']}")
                if meta.get("abstract"):
                    parts.append(f"Abstract: {meta['abstract']}")
                if meta.get("keywords"):
                    parts.append(f"Keywords: {', '.join(meta['keywords'])}")
                if meta.get("description"):
                    parts.append(f"Description: {meta['description']}")
                meta_lines.append("\n".join(parts))
            metadata_block = "\n\n---\n\n".join(meta_lines)
            question_with_context = (
                f"[Paper Metadata]\n"
                f"The following bibliographic information was pre-extracted from the paper(s). "
                f"Use it directly to answer metadata-related parts of the question.\n\n"
                f"{metadata_block}\n\n"
                f"[Question]\n{request.question}"
            )

        # Generate answer (single path for all cases)
        content = await _answering.generate_answer(
            question_with_context, image_paths, top_results
        ) if (image_paths or metadata_papers) else "I couldn't find relevant content for your question in the uploaded papers."

    except Exception as e:
        logger.exception("Chat handler failed")
        content = f"An error occurred while generating the answer: {str(e)[:120]}"
        citations = []
        action_types = ["error"]

    # --- Persist both turns to the notebook's chat session ---
    try:
        chat_session = _get_or_create_session(notebook, db)
        chat_session.send_message(role="user", content=request.question)
        citations_json = json.dumps([c.model_dump() for c in citations]) if citations else None
        chat_session.send_message(role="assistant", content=content, citations_json=citations_json)
        db.commit()
    except Exception:
        logger.exception("Failed to persist chat messages for notebook %s", notebook_id)
        db.rollback()

    return ChatResponse(
        content=content,
        citations=citations,
        query_type=",".join(action_types),
    )

