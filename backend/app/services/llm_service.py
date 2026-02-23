"""
VLM service: load Qwen2.5-VL-3B-Instruct locally and generate answers
by reading actual page images (vision-language RAG).
Runs on CPU in float16 (~6GB RAM). ColQwen2 stays on GPU uninterrupted.
"""
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from typing import List, Dict, Any
from app.config import settings

_model = None
_processor = None


def _load():
    global _model, _processor
    if _model is None:
        _model = Qwen2VLForConditionalGeneration.from_pretrained(
            settings.LLM_MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="cpu",
        ).eval()
        _processor = AutoProcessor.from_pretrained(settings.LLM_MODEL_NAME)


def generate_answer(question: str, pages: List[Dict[str, Any]]) -> str:
    """
    Generate an answer by passing page images + question to Qwen2.5-VL.
    pages: list of dicts with keys image_path, paper_title, page_num, score
    """
    _load()

    # Build multi-image message content
    content = []
    for i, page in enumerate(pages):
        image_path = page.get("image_path", "")
        title = page.get("paper_title", "Unknown")
        page_num = page.get("page_num", "?")
        if image_path:
            content.append({"type": "text", "text": f"[{i+1}] {title} — Page {page_num}:"})
            content.append({"type": "image", "image": image_path})

    content.append({
        "type": "text",
        "text": (
            f"\nQuestion: {question}\n\n"
            "Answer the question directly and concisely based only on the document pages shown above. "
            "Quote relevant text where possible. "
            "If the answer is not visible in the pages, say so clearly."
        )
    })

    messages = [{"role": "user", "content": content}]

    text = _processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = _processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )  # stays on CPU

    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
    return _processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()
