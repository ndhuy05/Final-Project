"""
Poster router: start, poll, cancel, and download Paper2Poster generation jobs.
"""
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.user import User
from app.services import memory_store
from app.services.auth_service import get_current_user
from app.agents import PosterAgent
from app.core.config import settings

router = APIRouter()

_poster_agent = PosterAgent()


@router.post("/notebooks/{notebook_id}/papers/{paper_id}/generate/poster")
async def start_poster_generation(
    notebook_id: str,
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Start the PosterAgent pipeline for a paper.
    Returns job_id for status polling.
    Responds with HTTP 409 if another poster job is already running.
    """
    paper = memory_store.get_paper(notebook_id, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    if PosterAgent.is_busy():
        raise HTTPException(
            status_code=409,
            detail="A poster generation job is already running. Cancel it before starting a new one.",
        )

    # Resolve the PDF path from the paper's stored filename
    pdf_path = os.path.abspath(
        os.path.join(settings.UPLOAD_DIR, paper["filename"])
    )
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    job_id = _poster_agent.generate_poster(
        notebook_id=notebook_id,
        paper_id=paper_id,
        paper_title=paper.get("title", ""),
        pdf_path=pdf_path,
    )

    if job_id is None:
        # Race condition: became busy between the is_busy() check and run()
        raise HTTPException(
            status_code=409,
            detail="A poster generation job is already running. Cancel it before starting a new one.",
        )

    return {"job_id": job_id}


@router.get("/generate/poster/{job_id}/status")
async def get_poster_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Poll generation progress."""
    job = _poster_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "status":   job["status"],
        "progress": job["progress"],
        "step":     job["step"],
        "error":    job.get("error"),
    }


@router.post("/generate/poster/{job_id}/cancel")
async def cancel_poster_generation(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Request cancellation of a running job."""
    job = _poster_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    cancelled = _poster_agent.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Job is not running.")
    return {"cancelled": True}


@router.get("/generate/poster/{job_id}/download")
async def download_poster_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Download the generated poster as a .pptx file. Only available when status=done."""
    job = _poster_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Poster generation not complete yet.")
    pptx_path = job.get("output_path")
    if not pptx_path or not os.path.exists(pptx_path):
        raise HTTPException(status_code=404, detail="PPTX file not found.")
    return FileResponse(
        path=pptx_path,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        filename=os.path.basename(pptx_path),
    )
