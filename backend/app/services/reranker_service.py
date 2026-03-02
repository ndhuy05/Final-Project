"""
Reranker service using BAAI/bge-reranker-base (ONNX via fastembed).
Loaded lazily on first use; uses CUDA if available, CPU otherwise.
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        logger.info("Loading BAAI/bge-reranker-base via fastembed")
        _reranker = TextCrossEncoder(
            model_name="BAAI/bge-reranker-base",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    return _reranker


def rerank(
    query: str,
    results: List[Dict[str, Any]],
    top_k: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Rerank *results* by relevance to *query* using a cross-encoder.

    Each result dict must have a 'content' or 'page_text' key with passage text.
    The 'score' field on each result is replaced with the reranker score.
    Returns results sorted by score descending (optionally truncated to top_k).
    """
    if not results:
        return results

    reranker = _get_reranker()
    passages = [r.get("content") or r.get("page_text") or "" for r in results]

    scores = list(reranker.rerank(query, passages))

    for r, s in zip(results, scores):
        r["score"] = float(s)

    results = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
    if top_k is not None:
        results = results[:top_k]

    logger.debug(
        "Reranker top scores: %s",
        [round(r.get("score", 0), 4) for r in results[:5]],
    )
    return results
