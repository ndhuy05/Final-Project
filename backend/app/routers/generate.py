"""
Generate router: start and track Paper2Code generation jobs.
"""
import os
from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from app.services import memory_store, paper2code_service

router = APIRouter()


@router.post("/notebooks/{notebook_id}/papers/{paper_id}/generate/code")
async def start_code_generation(notebook_id: str, paper_id: str):
    """Start Paper2Code pipeline for a paper. Returns job_id for status polling."""
    paper = memory_store.get_paper(notebook_id, paper_id)
    if not paper:
        raise HTTPException(status_code=404, detail="Paper not found.")

    job_id = paper2code_service.start_job(
        notebook_id=notebook_id,
        paper_id=paper_id,
        paper_title=paper.get("title", ""),
        page_count=paper.get("page_count", 0),
    )
    return {"job_id": job_id}


@router.post("/generate/code/{job_id}/cancel")
async def cancel_code_generation(job_id: str):
    """Request cancellation of a running job."""
    job = paper2code_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    cancelled = paper2code_service.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Job is not running.")
    return {"cancelled": True}



@router.get("/generate/code/{job_id}/status")
async def get_job_status(job_id: str):
    """Poll generation progress."""
    job = paper2code_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "status": job["status"],
        "progress": job["progress"],
        "step": job["step"],
        "error": job["error"],
    }


@router.get("/generate/code/{job_id}/download")
async def download_result(job_id: str):
    """Download the generated repository as a ZIP. Only available when status=done."""
    job = paper2code_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Generation not complete yet.")
    zip_path = job.get("zip_path")
    if not zip_path or not os.path.exists(zip_path):
        raise HTTPException(status_code=404, detail="ZIP file not found.")
    return FileResponse(
        path=zip_path,
        media_type="application/zip",
        filename=os.path.basename(zip_path),
    )
