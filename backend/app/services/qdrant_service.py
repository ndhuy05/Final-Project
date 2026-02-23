"""
Qdrant service: local on-disk client, multi-vector collection for ColQwen2.
No Docker required — uses QdrantClient(path=...).
"""
import os
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams,
    Distance,
    PointStruct,
    MultiVectorConfig,
    MultiVectorComparator,
    Filter,
    FieldCondition,
    MatchValue,
)
from app.config import settings

_client: QdrantClient = None

VECTOR_NAME = "colqwen2"
VECTOR_DIM = 128  # ColQwen2 patch embedding dimension


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        os.makedirs(settings.QDRANT_LOCAL_PATH, exist_ok=True)
        _client = QdrantClient(path=settings.QDRANT_LOCAL_PATH)
    return _client


def collection_name(notebook_id: str) -> str:
    return f"notebook_{notebook_id}"


def ensure_collection(notebook_id: str):
    """Create multi-vector collection for a notebook if it doesn't exist."""
    client = get_client()
    name = collection_name(notebook_id)
    existing = [c.name for c in client.get_collections().collections]
    if name not in existing:
        client.create_collection(
            collection_name=name,
            vectors_config={
                VECTOR_NAME: VectorParams(
                    size=VECTOR_DIM,
                    distance=Distance.COSINE,
                    multivector_config=MultiVectorConfig(
                        comparator=MultiVectorComparator.MAX_SIM
                    ),
                )
            },
        )


def upsert_page(
    notebook_id: str,
    point_id: int,
    multi_vector: List[List[float]],
    payload: Dict[str, Any],
):
    """Upsert a single page's multi-vector into the collection."""
    client = get_client()
    client.upsert(
        collection_name=collection_name(notebook_id),
        points=[
            PointStruct(
                id=point_id,
                vector={VECTOR_NAME: multi_vector},
                payload=payload,
            )
        ],
    )


def search(
    notebook_id: str,
    query_multi_vector: List[List[float]],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    MaxSim search: find top-k most relevant pages for a query.
    Returns list of payload dicts with score.
    """
    client = get_client()
    results = client.query_points(
        collection_name=collection_name(notebook_id),
        query=query_multi_vector,
        using=VECTOR_NAME,
        limit=top_k,
    )
    return [
        {**point.payload, "score": point.score}
        for point in results.points
    ]


def delete_paper_points(notebook_id: str, paper_id: str):
    """Delete all Qdrant points belonging to a paper."""
    client = get_client()
    client.delete(
        collection_name=collection_name(notebook_id),
        points_selector=Filter(
            must=[FieldCondition(key="paper_id", match=MatchValue(value=paper_id))]
        ),
    )
