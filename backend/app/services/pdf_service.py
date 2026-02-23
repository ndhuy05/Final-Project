"""
PDF processing service: convert PDF pages to PIL images.
Uses PyMuPDF (fitz). Text extraction removed — the VLM reads page images directly.
"""
from typing import List
from PIL import Image
import fitz  # PyMuPDF


def pdf_to_images(pdf_path: str, dpi: int = 150) -> List[Image.Image]:
    """
    Convert each page of a PDF into a PIL Image (RGB).
    Returns list indexed by page number (0-based).
    """
    doc = fitz.open(pdf_path)
    images = []
    mat = fitz.Matrix(dpi / 72, dpi / 72)

    for page in doc:
        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)

    doc.close()
    return images
