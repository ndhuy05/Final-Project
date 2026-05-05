"""
Notebooks router: create, list, rename, and delete user-owned notebooks.
All endpoints require a valid JWT (via get_current_user).
"""
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.notebook import Notebook
from app.models.user import User
from app.services import memory_store
from app.services.auth_service import get_current_user
from app.services.notebook_service import (
    list_notebooks_for_user,
    create_notebook,
    rename_notebook,
    delete_notebook as _delete_notebook,
)

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
    notebooks = list_notebooks_for_user(db, current_user.id)
    return {"notebooks": [_notebook_to_dict(n) for n in notebooks]}


@router.post("/notebooks", status_code=201)
async def create_notebook_endpoint(
    request: NotebookCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Create a new notebook owned by the current user."""
    notebook = create_notebook(db, name=request.name, user_id=current_user.id)
    return _notebook_to_dict(notebook)


@router.patch("/notebooks/{notebook_id}")
async def rename_notebook_endpoint(
    notebook_id: str,
    request: NotebookRename,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Rename a notebook owned by the current user."""
    rename_notebook(db, notebook_id, current_user.id, request.name)
    return {"success": True}


@router.delete("/notebooks/{notebook_id}")
async def delete_notebook(
    notebook_id: str,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Soft-delete a notebook owned by the current user."""
    _delete_notebook(db, notebook_id, current_user.id)
    return {"success": True}
