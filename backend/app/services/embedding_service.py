"""
Embedding service: GPU-accelerated text embeddings via fastembed-gpu.
Model: BAAI/bge-small-en-v1.5 (384-dim, ~22MB).
Uses CUDAExecutionProvider via ONNX Runtime; falls back to CPU if unavailable.
"""
import os
import sys
from typing import List
from fastembed import TextEmbedding

_model: TextEmbedding | None = None


def _add_cuda_dll_path():
    """Add PyTorch's CUDA DLL directory to the search path (Windows only)."""
    if sys.platform != "win32":
        return
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass


def _get_model() -> TextEmbedding:
    global _model
    if _model is None:
        _add_cuda_dll_path()
        _model = TextEmbedding(
            model_name="BAAI/bge-small-en-v1.5",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
    return _model


def embed_text(text: str) -> List[float]:
    """Embed a single string. Returns a flat 384-dim vector."""
    model = _get_model()
    return next(model.embed([text])).tolist()


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a list of strings. Returns one vector per text."""
    model = _get_model()
    return [v.tolist() for v in model.embed(texts)]
