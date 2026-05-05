"""
Notebook and paper metadata store backed by SQLite via SQLAlchemy.

Public API is identical to the old JSON-file implementation so all callers
(routers, agents) require zero changes:

    add_paper(notebook_id, title, filename, page_count) -> dict
    get_papers(notebook_id) -> list[dict]
    get_paper(notebook_id, paper_id) -> dict | None
    update_paper_metadata(notebook_id, paper_id, metadata) -> None
    delete_paper(notebook_id, paper_id) -> bool
    get_or_create_notebook(notebook_id) -> dict

Each function opens its own short-lived Session so it is safe to call from
both sync and async FastAPI code.
"""
import json as _json
import logging
from typing import Any

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.models.notebook import Notebook
from app.models.paper import Paper

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _paper_to_dict(paper: Paper) -> dict[str, Any]:
    """Convert a Paper ORM row to the dict format expected by all callers."""
    d: dict[str, Any] = {
        "id": paper.id,
        "notebook_id": paper.notebook_id,
        "title": paper.title,
        "filename": paper.storage_path,
        "page_count": paper.page_count,
    }
    if paper.metadata_json:
        d["metadata"] = paper.metadata_json
    return d


def _ensure_notebook(db: Session, notebook_id: str) -> Notebook:
    """Return existing Notebook or create one on demand (no commit — caller owns it)."""
    notebook = db.get(Notebook, notebook_id)
    if notebook is None:
        notebook = Notebook(id=notebook_id, title=notebook_id, user_id=None)
        db.add(notebook)
    return notebook


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_or_create_notebook(notebook_id: str) -> dict[str, Any]:
    with SessionLocal() as db:
        notebook = _ensure_notebook(db, notebook_id)
        db.commit()
        return {"id": notebook.id, "title": notebook.title}


def add_paper(notebook_id: str, title: str, filename: str, page_count: int) -> dict[str, Any]:
    with SessionLocal() as db:
        _ensure_notebook(db, notebook_id)
        # storage_path is the UUID-prefixed filename; original_filename strips the prefix
        original_filename = filename.split("_", 1)[-1] if "_" in filename else filename
        paper = Paper(
            notebook_id=notebook_id,
            title=title,
            original_filename=original_filename,
            storage_path=filename,
            page_count=page_count,
        )
        db.add(paper)
        db.commit()
        db.refresh(paper)
        return _paper_to_dict(paper)


def get_papers(notebook_id: str) -> list[dict[str, Any]]:
    with SessionLocal() as db:
        papers = (
            db.query(Paper)
            .filter(Paper.notebook_id == notebook_id, Paper.deleted_at.is_(None))
            .all()
        )
        return [_paper_to_dict(p) for p in papers]


def get_paper(notebook_id: str, paper_id: str) -> dict[str, Any] | None:
    with SessionLocal() as db:
        paper = (
            db.query(Paper)
            .filter(
                Paper.id == paper_id,
                Paper.notebook_id == notebook_id,
                Paper.deleted_at.is_(None),
            )
            .first()
        )
        return _paper_to_dict(paper) if paper else None


def update_paper_metadata(notebook_id: str, paper_id: str, metadata: dict) -> None:
    with SessionLocal() as db:
        paper = (
            db.query(Paper)
            .filter(Paper.id == paper_id, Paper.notebook_id == notebook_id)
            .first()
        )
        if paper is None:
            logger.warning("update_paper_metadata: paper %s not found", paper_id)
            return
        paper.metadata_json = metadata
        # Also populate indexed scalar columns for future querying
        paper.year = metadata.get("year")
        paper.venue = metadata.get("venue")
        paper.abstract = metadata.get("abstract")
        paper.description = metadata.get("description")
        authors = metadata.get("authors")
        if authors:
            paper.authors = _json.dumps(authors, ensure_ascii=False)
        db.commit()


def delete_paper(notebook_id: str, paper_id: str) -> bool:
    with SessionLocal() as db:
        paper = (
            db.query(Paper)
            .filter(Paper.id == paper_id, Paper.notebook_id == notebook_id)
            .first()
        )
        if paper is None:
            return False
        db.delete(paper)
        db.commit()
        return True
