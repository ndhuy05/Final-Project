"""
Notebooks router: create, list, rename, and delete user-owned notebooks.
All endpoints require a valid JWT (via get_current_user).
"""
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.notebook import Notebook
from app.models.user import User
from app.services import memory_store
from app.services.auth_service import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Request schemas ---

class NotebookCreate(BaseModel):
    name: str


class NotebookRename(BaseModel):
    name: str


# --- Internal helpers ---

def _notebook_to_dict(notebook: Notebook) -> dict:
    """Serialize a Notebook ORM row to the shape expected by the frontend."""
    papers = memory_store.get_papers(notebook.id)
    return {
        "id": notebook.id,
        "name": notebook.title,
        "createdAt": (
            notebook.created_at.strftime("%Y-%m-%d") if notebook.created_at else None
        ),
        "papers": papers,
        "messages": [],
        "notes": [],
    }


# --- Endpoints ---

@router.get("/notebooks")
async def list_notebooks(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Return all non-deleted notebooks owned by the current user."""
    notebooks = (
        db.query(Notebook)
        .filter(
            Notebook.user_id == current_user.id,
            Notebook.deleted_at.is_(None),
        )
        .order_by(Notebook.created_at.asc())
        .all()
    )
    return {"notebooks": [_notebook_to_dict(n) for n in notebooks]}


@router.post("/notebooks", status_code=201)
async def create_notebook(
    request: NotebookCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new notebook owned by the current user."""
    notebook = Notebook(
        id=str(uuid.uuid4()),
        title=request.name,
        user_id=current_user.id,
    )
    db.add(notebook)
    db.commit()
    db.refresh(notebook)
    return _notebook_to_dict(notebook)


@router.patch("/notebooks/{notebook_id}")
async def rename_notebook(
    notebook_id: str,
    request: NotebookRename,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename a notebook owned by the current user."""
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
    notebook.rename(request.name)
    db.commit()
    return {"success": True}


@router.delete("/notebooks/{notebook_id}")
async def delete_notebook(
    notebook_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete a notebook owned by the current user."""
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
    notebook.delete()
    db.commit()
    return {"success": True}
