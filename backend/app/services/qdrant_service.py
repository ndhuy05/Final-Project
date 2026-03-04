"""
Qdrant service: local on-disk client, dense single-vector collection.
Vectors are 384-dim text embeddings from fastembed (BAAI/bge-small-en-v1.5).
No Docker required — uses QdrantClient(path=...).
"""
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config import settings

_client: QdrantClient = None

VECTOR_DIM = 384  # BAAI/bge-small-en-v1.5 output dimension


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        os.makedirs(settings.QDRANT_LOCAL_PATH, exist_ok=True)
        _client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
    return _client


def collection_name(notebook_id: str) -> str:
    return f"notebook_{notebook_id}"


def ensure_collection(notebook_id: str):
    """Create dense vector collection for a notebook if it doesn't exist.
    If an existing collection has an incompatible schema (e.g. old ColQwen2
    multi-vector format), it is deleted and recreated automatically.
    """
    client = get_client()
    name = collection_name(notebook_id)
    existing = [c.name for c in client.get_collections().collections]
    if name in existing:
        # Validate the collection has the expected dense vector config
        info = client.get_collection(name)
        config = info.config.params.vectors
        # Dense config is a VectorParams object (not a dict of named vectors)
        compatible = (
            hasattr(config, "size")
            and config.size == VECTOR_DIM
        )
        if not compatible:
            client.delete_collection(name)
            existing = []  # force recreation below

    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )


def upsert_table(
    notebook_id: str,
    point_id: int,
    vector: List[float],
    payload: Dict[str, Any],
):
    """Upsert a single table's embedding into the collection."""
    client = get_client()
    client.upsert(
        collection_name=collection_name(notebook_id),
        points=[PointStruct(id=point_id, vector=vector, payload=payload)],
    )


def search(
    notebook_id: str,
    query_vector: List[float],
    top_k: int = 5,
    paper_id: str = None,
) -> List[Dict[str, Any]]:
    """Find top-k most relevant chunks for a query vector.
    If paper_id is provided, restricts search to that paper only.
    """
    client = get_client()
    search_filter = None
    if paper_id:
        search_filter = Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        )
    results = client.query_points(
        collection_name=collection_name(notebook_id),
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
    )
    return [{**point.payload, "score": point.score} for point in results.points]


def get_page_text(notebook_id: str, paper_id: str, page_num: int) -> str:
    """
    Return the full extracted text of a specific page (1-based page_num).
    Finds the first text chunk for this page and returns its page_text payload.
    Returns empty string if not found.
    """
    client = get_client()
    results, _ = client.scroll(
        collection_name=collection_name(notebook_id),
        scroll_filter=Filter(must=[
            FieldCondition(key="paper_id", match=MatchValue(value=paper_id)),
            FieldCondition(key="page_num", match=MatchValue(value=page_num)),
            FieldCondition(key="type", match=MatchValue(value="text")),
        ]),
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    if results:
        return results[0].payload.get("page_text", "")
    return ""


def delete_paper_points(notebook_id: str, paper_id: str):
    """Delete all Qdrant points belonging to a paper."""
    client = get_client()
    client.delete(
        collection_name=collection_name(notebook_id),
        points_selector=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
    )
