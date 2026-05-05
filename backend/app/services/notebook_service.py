"""
Notebook service: CRUD operations for Notebook records.
All SQL queries that touch the notebooks table live here.
"""
import uuid
import logging
from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.notebook import Notebook

logger = logging.getLogger(__name__)


def get_notebook_for_user(db: Session, notebook_id: str, user_id: str) -> Notebook:
    """Return notebook owned by user or raise HTTP 404."""
    notebook = (
        db.query(Notebook)
        .filter(
            Notebook.id == notebook_id,
            Notebook.user_id == user_id,
            Notebook.deleted_at.is_(None),
        )
        .first()
    )
    if not notebook:
        raise HTTPException(status_code=404, detail="Notebook not found.")
    return notebook


def list_notebooks_for_user(db: Session, user_id: str) -> list[Notebook]:
    """Return all non-deleted notebooks owned by user, ordered by creation date."""
    return (
        db.query(Notebook)
        .filter(
            Notebook.user_id == user_id,
            Notebook.deleted_at.is_(None),
        )
        .order_by(Notebook.created_at.asc())
        .all()
    )


def create_notebook(db: Session, name: str, user_id: str) -> Notebook:
    """Create a new notebook owned by user. Returns the refreshed ORM object."""
    notebook = Notebook(
        id=str(uuid.uuid4()),
        title=name,
        user_id=user_id,
    )
    db.add(notebook)
    db.commit()
    db.refresh(notebook)
    return notebook


def rename_notebook(db: Session, notebook_id: str, user_id: str, name: str) -> None:
    """Rename a notebook. Raises HTTP 404 if not found or not owned by user."""
    notebook = get_notebook_for_user(db, notebook_id, user_id)
    notebook.rename(name)
    db.commit()


def delete_notebook(db: Session, notebook_id: str, user_id: str) -> None:
    """Soft-delete a notebook. Raises HTTP 404 if not found or not owned by user."""
    notebook = get_notebook_for_user(db, notebook_id, user_id)
    notebook.delete()
    db.commit()
