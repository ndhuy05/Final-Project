"""
Papers router: upload PDF, extract content with OpenRouter vision model,
chunk with LangChain, embed with fastembed, store in Qdrant.
Extracts structured metadata + description from the first batch (contains page 0).
"""
import logging
import os
import uuid
import asyncio
from typing import List, Tuple
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services import memory_store, pdf_service, qdrant_service, embedding_service, openrouter_service

logger = logging.getLogger(__name__)
router = APIRouter()

BATCH_SIZE = 1  # pages per VLM call (increase only if vision model reliably follows [PAGE N] delimiters)

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,
    separators=["\n\n", "\n", ". ", " ", ""],
    strip_whitespace=True,
)


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks using LangChain RecursiveCharacterTextSplitter."""
    chunks = _text_splitter.split_text(text)
    return [c for c in chunks if len(c) > 20]


async def _process_batch(
    pages: List[Tuple[int, str]],  # [(0-based page_num, image_path), ...]
    paper: dict,
    notebook_id: str,
) -> List[dict]:
    """
    Extract and index a batch of pages in one VLM call.
    extract_metadata=True when page 0 is in the batch (first batch only).
    Returns list of {"chunks": int, "metadata": dict} per page.
    """
    has_page0 = any(pn == 0 for pn, _ in pages)
    page_results = await openrouter_service.extract_page_content(
        pages,
        extract_metadata=has_page0,
    )

    output = []
    for r in page_results:
        page_num = r["page_num"]
        text = r["text"]
        metadata = r["metadata"]
        page_label = page_num + 1  # 1-based for display

        chunk_count = 0
        if text:
            chunks = _chunk_text(text)
            for chunk_idx, chunk in enumerate(chunks):
                vector = embedding_service.embed_text(chunk)
                point_id = abs(hash(f"{paper['id']}_{page_num}_text_{chunk_idx}")) % (2**53)
                qdrant_service.upsert_table(
                    notebook_id=notebook_id,
                    point_id=point_id,
                    vector=vector,
                    payload={
                        "type": "text",
                        "paper_id": paper["id"],
                        "paper_title": paper["title"],
                        "page_num": page_label,
                        "content": chunk,
                        "page_text": text,
                    },
                )
                chunk_count += 1

        output.append({"chunks": chunk_count, "metadata": metadata})

    return output


@router.post("/notebooks/{notebook_id}/papers/upload")
async def upload_paper(notebook_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Reject duplicate: same original filename already in this notebook
    existing = memory_store.get_papers(notebook_id)
    for p in existing:
        original_name = p.get("filename", "").split("_", 1)[-1]  # strip UUID prefix
        if original_name == file.filename:
            raise HTTPException(
                status_code=409,
                detail=f"A paper named '{file.filename}' already exists in this notebook.",
            )

    # Save uploaded PDF
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    safe_name = f"{uuid.uuid4()}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    try:
        # Convert PDF pages to images
        images = pdf_service.pdf_to_images(file_path)

        # Save paper metadata
        paper = memory_store.add_paper(
            notebook_id=notebook_id,
            title=file.filename.replace(".pdf", ""),
            filename=safe_name,
            page_count=len(images),
        )

        # Save page images to disk
        image_dir = os.path.join(settings.IMAGE_DIR, paper["id"])
        os.makedirs(image_dir, exist_ok=True)
        image_paths = []
        for page_num, img in enumerate(images):
            img_path = os.path.join(image_dir, f"page_{page_num + 1}.png")
            img.save(img_path, format="PNG")
            image_paths.append(img_path)

        # Ensure Qdrant collection exists
        qdrant_service.ensure_collection(notebook_id)

        # Group pages into batches of BATCH_SIZE, fire one VLM call per batch
        batches: List[List[Tuple[int, str]]] = [
            [(i, image_paths[i]) for i in range(start, min(start + BATCH_SIZE, len(image_paths)))]
            for start in range(0, len(image_paths), BATCH_SIZE)
        ]
        batch_tasks = [_process_batch(batch, paper, notebook_id) for batch in batches]
        batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)

        chunk_count = 0
        metadata: dict = {}
        for batch_result in batch_results:
            if isinstance(batch_result, Exception):
                logger.error("Batch processing error: %s", batch_result)
                continue
            for r in batch_result:
                chunk_count += r.get("chunks", 0)
                if r.get("metadata"):  # take metadata from first page that has it
                    metadata = r["metadata"]

        # Store metadata extracted inline from page 1
        if metadata:
            memory_store.update_paper_metadata(notebook_id, paper["id"], metadata)
            logger.debug("Metadata stored for %s: %s", paper["title"], list(metadata.keys()))

        return JSONResponse(content={"paper": paper, "chunks_indexed": chunk_count})

    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notebooks/{notebook_id}/papers")
async def list_papers(notebook_id: str):
    papers = memory_store.get_papers(notebook_id)
    return {"papers": papers}


@router.get("/notebooks/{notebook_id}/chunks")
async def list_chunks(notebook_id: str, limit: int = 20, offset: int = 0, type: str = None):
    """
    Debug endpoint: browse indexed chunks/tables stored in Qdrant.
    ?type=text  — show only text chunks
    ?type=table — show only table chunks
    ?limit=20&offset=0 — pagination
    """
    client = qdrant_service.get_client()
    name = qdrant_service.collection_name(notebook_id)
    try:
        scroll_result, _ = client.scroll(
            collection_name=name,
            limit=limit,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
    except Exception as e:
        return {"error": str(e), "chunks": []}

    chunks = []
    for point in scroll_result:
        p = point.payload or {}
        if type and p.get("type") != type:
            continue
        entry = {
            "id": point.id,
            "type": p.get("type"),
            "paper_title": p.get("paper_title"),
            "page_num": p.get("page_num"),
        }
        if p.get("type") == "table":
            entry["table_index"] = p.get("table_index")
            entry["summary"] = p.get("summary")
            entry["markdown_table"] = p.get("markdown_table")
        else:
            entry["content"] = p.get("content")          # small search chunk
            entry["pair_text_len"] = len(p.get("pair_text", ""))  # full context length
        chunks.append(entry)

    info = client.get_collection(name)
    return {
        "total_points": info.points_count,
        "returned": len(chunks),
        "offset": offset,
        "chunks": chunks,
    }


@router.delete("/notebooks/{notebook_id}/papers/{paper_id}")
async def delete_paper(notebook_id: str, paper_id: str):
    paper = memory_store.get_paper(notebook_id, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    qdrant_service.delete_paper_points(notebook_id, paper_id)

    file_path = os.path.join(settings.UPLOAD_DIR, paper["filename"])
    if os.path.exists(file_path):
        os.remove(file_path)

    image_dir = os.path.join(settings.IMAGE_DIR, paper_id)
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    memory_store.delete_paper(notebook_id, paper_id)
    return {"success": True}
