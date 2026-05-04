"""
Web router: start, poll, cancel, and download Paper2Web generation jobs.
"""
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.services import memory_store
from app.services.auth_service import get_current_user
from app.models import WebAgent
from app.config import settings

router = APIRouter()

_web_agent = WebAgent()


@router.post("/notebooks/{notebook_id}/papers/{paper_id}/generate/web")
async def start_web_generation(
    notebook_id: str,
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Start the WebAgent pipeline for a paper.
    Returns job_id for status polling.
    Responds with HTTP 409 if another web job is already running.
    """
    paper = memory_store.get_paper(notebook_id, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    if WebAgent.is_busy():
        raise HTTPException(
            status_code=409,
            detail="A web generation job is already running. Cancel it before starting a new one.",
        )

    pdf_path = os.path.abspath(
        os.path.join(settings.UPLOAD_DIR, paper["filename"])
    )
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="PDF file not found on disk.")

    job_id = _web_agent.run(
        notebook_id=notebook_id,
        paper_id=paper_id,
        paper_title=paper.get("title", ""),
        pdf_path=pdf_path,
    )

    if job_id is None:
        raise HTTPException(
            status_code=409,
            detail="A web generation job is already running. Cancel it before starting a new one.",
        )

    return {"job_id": job_id}


@router.get("/generate/web/{job_id}/status")
async def get_web_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Poll generation progress."""
    job = _web_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "status":   job["status"],
        "progress": job["progress"],
        "step":     job["step"],
        "error":    job.get("error"),
    }


@router.post("/generate/web/{job_id}/cancel")
async def cancel_web_generation(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Request cancellation of a running job."""
    job = _web_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    cancelled = _web_agent.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Job is not running.")
    return {"cancelled": True}


@router.get("/generate/web/{job_id}/download")
async def download_web_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Download the generated website as a .zip file. Only available when status=done."""
    job = _web_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Web generation not complete yet.")
    zip_path = job.get("output_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP file not found.")
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )
