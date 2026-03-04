"""
In-memory store for notebook and paper metadata, persisted to a JSON file
so data survives server restarts.
"""
import json
import logging
import os
from typing import Dict, List, Optional
import uuid

logger = logging.getLogger(__name__)

_STORE_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "memory_store.json")

# Structure: { notebook_id: { "papers": [ {id, title, filename, page_count, metadata?} ] } }
_store: Dict[str, Dict] = {}


def _load():
    global _store
    path = os.path.abspath(_STORE_PATH)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _store = json.load(f)
        except Exception as e:
            logger.warning("Could not load memory_store.json: %s", e)
            _store = {}


def _save():
    path = os.path.abspath(_STORE_PATH)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(_store, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("Could not save memory_store.json: %s", e)


# Load on module import
_load()


def get_or_create_notebook(notebook_id: str) -> Dict:
    if notebook_id not in _store:
        _store[notebook_id] = {"papers": []}
    return _store[notebook_id]


def add_paper(notebook_id: str, title: str, filename: str, page_count: int) -> Dict:
    notebook = get_or_create_notebook(notebook_id)
    paper = {
        "id": str(uuid.uuid4()),
        "notebook_id": notebook_id,
        "title": title,
        "filename": filename,
        "page_count": page_count,
    }
    notebook["papers"].append(paper)
    _save()
    return paper


def get_papers(notebook_id: str) -> List[Dict]:
    return get_or_create_notebook(notebook_id).get("papers", [])


def get_paper(notebook_id: str, paper_id: str) -> Optional[Dict]:
    for paper in get_papers(notebook_id):
        if paper["id"] == paper_id:
            return paper
    return None


def update_paper_metadata(notebook_id: str, paper_id: str, metadata: dict) -> None:
    """Merge extracted metadata dict into the paper record."""
    paper = get_paper(notebook_id, paper_id)
    if paper is not None:
        paper["metadata"] = metadata
        _save()


def delete_paper(notebook_id: str, paper_id: str) -> bool:
    notebook = get_or_create_notebook(notebook_id)
    before = len(notebook["papers"])
    notebook["papers"] = [p for p in notebook["papers"] if p["id"] != paper_id]
    _save()
    return len(notebook["papers"]) < before
