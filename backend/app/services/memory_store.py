"""
In-memory store for notebook and paper metadata.
No database required — data resets on server restart.
"""
from typing import Dict, List, Optional
import uuid

# Structure: { notebook_id: { "papers": [ {id, title, filename, page_count} ] } }
_store: Dict[str, Dict] = {}


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
    return paper


def get_papers(notebook_id: str) -> List[Dict]:
    return get_or_create_notebook(notebook_id).get("papers", [])


def get_paper(notebook_id: str, paper_id: str) -> Optional[Dict]:
    for paper in get_papers(notebook_id):
        if paper["id"] == paper_id:
            return paper
    return None


def delete_paper(notebook_id: str, paper_id: str) -> bool:
    notebook = get_or_create_notebook(notebook_id)
    before = len(notebook["papers"])
    notebook["papers"] = [p for p in notebook["papers"] if p["id"] != paper_id]
    return len(notebook["papers"]) < before
