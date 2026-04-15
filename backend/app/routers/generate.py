"""
Generate router: start and track Paper2Code generation jobs.
"""
import os
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.services import memory_store
from app.services.auth_service import get_current_user
from app.models import CodeAgent

router = APIRouter()

_code_agent = CodeAgent()


@router.post("/notebooks/{notebook_id}/papers/{paper_id}/generate/code")
async def start_code_generation(
    notebook_id: str,
    paper_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Start Paper2Code pipeline for a paper. Returns job_id for status polling."""
    paper = memory_store.get_paper(notebook_id, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    job_id = _code_agent.run(
        notebook_id=notebook_id,
        paper_id=paper_id,
        paper_title=paper.get("title", ""),
        page_count=paper.get("page_count", 0),
    )
    return {"job_id": job_id}


@router.post("/generate/code/{job_id}/cancel")
async def cancel_code_generation(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Request cancellation of a running job."""
    job = _code_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    cancelled = _code_agent.cancel(job_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Job is not running.")
    return {"cancelled": True}


@router.get("/generate/code/{job_id}/status")
async def get_job_status(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Poll generation progress."""
    job = _code_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "status": job["status"],
        "progress": job["progress"],
        "step": job["step"],
        "error": job.get("error"),
    }


@router.get("/generate/code/{job_id}/download")
async def download_result(
    job_id: str,
    current_user: User = Depends(get_current_user),
):
    """Download the generated repository as a ZIP. Only available when status=done."""
    job = _code_agent.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Generation not complete yet.")
    zip_path = job.get("output_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP file not found.")
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )
