"""
Papers router: upload PDF, extract tables with OpenRouter vision model,
embed table summaries with fastembed, store in Qdrant.
Original Markdown tables are stored in the Qdrant payload for later retrieval.
"""
import os
import uuid
import shutil
import asyncio
from typing import List
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import aiofiles
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.services import memory_store, pdf_service, qdrant_service, embedding_service, openrouter_service

router = APIRouter()

_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=75,        # 15% of 500
    separators=["\n\n", "\n", ". ", " ", ""],
    strip_whitespace=True,
)


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks using LangChain RecursiveCharacterTextSplitter."""
    chunks = _text_splitter.split_text(text)
    return [c for c in chunks if len(c) > 20]


async def _process_page(image_path: str, page_num: int, paper: dict, notebook_id: str) -> int:
    """
    Extract page content as plain text (tables described in prose) via one vision call.
    page_num: 0-based page index.
    Returns chunk_count.
    """
    text = await openrouter_service.extract_page_content([image_path])
    page_label = page_num + 1  # 1-based for display/lookup

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

    return chunk_count


@router.post("/notebooks/{notebook_id}/papers/upload")
async def upload_paper(notebook_id: str, file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

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

        # Process each page independently — one vision call per page
        tasks = [
            _process_page(image_paths[i], i, paper, notebook_id)
            for i in range(len(image_paths))
        ]
        results = await asyncio.gather(*tasks)
        chunk_count = sum(results)

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
