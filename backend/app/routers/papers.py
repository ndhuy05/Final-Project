"""
Papers router: upload PDF, embed with ColQwen2, store in Qdrant.
Page images are saved to disk for later use by the VLM.
"""
import os
import uuid
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import aiofiles

from app.config import settings
from app.services import memory_store, pdf_service, colpali_service, qdrant_service

router = APIRouter()


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

        # Embed all page images with ColQwen2
        embeddings = colpali_service.embed_images(images)

        # Ensure Qdrant collection exists for this notebook
        qdrant_service.ensure_collection(notebook_id)

        # Save paper metadata
        paper = memory_store.add_paper(
            notebook_id=notebook_id,
            title=file.filename.replace(".pdf", ""),
            filename=safe_name,
            page_count=len(images),
        )

        # Save page images to disk for VLM retrieval
        image_dir = os.path.join(settings.IMAGE_DIR, paper["id"])
        os.makedirs(image_dir, exist_ok=True)
        for page_num, img in enumerate(images):
            img_path = os.path.join(image_dir, f"page_{page_num + 1}.png")
            img.save(img_path, format="PNG")

        # Upsert each page into Qdrant (no text — VLM reads images directly)
        for page_num, multi_vec in enumerate(embeddings):
            point_id = abs(hash(f"{paper['id']}_{page_num}")) % (2**53)
            img_path = os.path.join(image_dir, f"page_{page_num + 1}.png")
            qdrant_service.upsert_page(
                notebook_id=notebook_id,
                point_id=point_id,
                multi_vector=multi_vec,
                payload={
                    "paper_id": paper["id"],
                    "paper_title": paper["title"],
                    "page_num": page_num + 1,
                    "image_path": img_path,
                },
            )

        return JSONResponse(content={"paper": paper})

    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/notebooks/{notebook_id}/papers")
async def list_papers(notebook_id: str):
    papers = memory_store.get_papers(notebook_id)
    return {"papers": papers}


@router.delete("/notebooks/{notebook_id}/papers/{paper_id}")
async def delete_paper(notebook_id: str, paper_id: str):
    paper = memory_store.get_paper(notebook_id, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    # Remove vectors from Qdrant
    qdrant_service.delete_paper_points(notebook_id, paper_id)

    # Remove uploaded PDF
    file_path = os.path.join(settings.UPLOAD_DIR, paper["filename"])
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove saved page images
    import shutil
    image_dir = os.path.join(settings.IMAGE_DIR, paper_id)
    if os.path.exists(image_dir):
        shutil.rmtree(image_dir)

    memory_store.delete_paper(notebook_id, paper_id)
    return {"success": True}
