"""
ColQwen2 service: load vidore/colqwen2-v1.0 and provide
page embedding + query embedding as a singleton.
"""
import torch
from PIL import Image
from typing import List
from colpali_engine.models import ColQwen2, ColQwen2Processor
from app.config import settings

_model: ColQwen2 = None
_processor: ColQwen2Processor = None


def _load():
    global _model, _processor
    if _model is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model = ColQwen2.from_pretrained(
            settings.COLPALI_MODEL_NAME,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        ).to(device).eval()
        _processor = ColQwen2Processor.from_pretrained(settings.COLPALI_MODEL_NAME)


def offload_to_cpu():
    """Move ColQwen2 to CPU and free VRAM for another model (e.g. VLM)."""
    global _model
    if _model is not None and next(_model.parameters()).device.type == "cuda":
        _model.to("cpu")
        torch.cuda.empty_cache()


def reload_to_gpu():
    """Move ColQwen2 back to GPU after VLM is done."""
    global _model
    if _model is not None and torch.cuda.is_available():
        _model.to("cuda")


def embed_images(images: List[Image.Image]) -> List[List[List[float]]]:
    """
    Embed a list of page images.
    Returns a list of multi-vectors (one per image),
    where each multi-vector is a list of patch embeddings.
    """
    _load()
    device = next(_model.parameters()).device
    batch_size = 4  # process in small batches to manage memory

    all_embeddings = []
    for i in range(0, len(images), batch_size):
        batch = images[i : i + batch_size]
        inputs = _processor.process_images(batch).to(device)
        with torch.no_grad():
            embeddings = _model(**inputs)  # shape: (B, num_patches, dim)
        all_embeddings.extend(embeddings.cpu().float().tolist())

    return all_embeddings


def embed_query(text: str) -> List[List[float]]:
    """
    Embed a text query.
    Returns a multi-vector (list of token embeddings) for MaxSim search.
    """
    _load()
    device = next(_model.parameters()).device
    inputs = _processor.process_queries([text]).to(device)
    with torch.no_grad():
        embedding = _model(**inputs)  # shape: (1, num_tokens, dim)
    return embedding[0].cpu().float().tolist()
