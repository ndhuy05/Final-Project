from dotenv import load_dotenv
from utils.src.utils import get_json_from_response
from utils.src.model_utils import parse_pdf
import json
import os
import random

from camel.models import ModelFactory
from camel.agents import ChatAgent
from tenacity import retry, stop_after_attempt
from docling_core.types.doc import ImageRefMode, PictureItem, TableItem

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

from pathlib import Path

import PIL

try:
    from marker.models import create_model_dict
    _MARKER_AVAILABLE = True
except Exception:  # noqa: BLE001 — marker is an optional fallback for when Docling fails
    create_model_dict = None  # type: ignore[assignment]
    _MARKER_AVAILABLE = False

from utils.wei_utils import *

from utils.pptx_utils import *
from utils.critic_utils import *
import torch
from jinja2 import Template
import re

load_dotenv()
IMAGE_RESOLUTION_SCALE = 2.0  # lowered from 5.0 — figure crops only, no need for 5× resolution

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = IMAGE_RESOLUTION_SCALE
pipeline_options.generate_page_images = False  # full-page rasters not needed; only figure/table crops are used
pipeline_options.generate_picture_images = True
pipeline_options.do_ocr = False          # PDF has embedded text; OCR wastes ~2 min/page

doc_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
    }
)

@retry(stop=stop_after_attempt(5))
def parse_raw(args, actor_config, version=1):
    raw_source = args.poster_path
    markdown_clean_pattern = re.compile(r"<!--[\s\S]*?-->")

    # --- Determine text_content BEFORE running Docling ---
    # Prefer pre-extracted VLM text from Qdrant (written by the parent service)
    # because it is higher quality than Docling's markdown export.
    preextracted_path = getattr(args, "preextracted_text_path", None)
    text_content = None
    if preextracted_path:
        try:
            with open(preextracted_path, "r", encoding="utf-8") as _f:
                page_texts: dict = json.load(_f)
            # Sort pages numerically and concatenate
            sorted_pages = sorted(page_texts.items(), key=lambda kv: int(kv[0]))
            text_content = "\n\n".join(text for _, text in sorted_pages if text.strip())
            print(f"[parse_raw] Using pre-extracted text: {len(sorted_pages)} pages, "
                  f"{len(text_content)} chars total.")
        except Exception as exc:
            print(f"[parse_raw] WARNING: failed to load pre-extracted text ({exc}); "
                  "falling back to Docling markdown.")
            preextracted_path = None
            text_content = None

    # Check whether image/table extraction outputs from a previous run are cached.
    # These paths mirror what gen_image_and_table() writes.
    _img_json = f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}_images.json'
    _tbl_json = f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}_tables.json'
    _images_cached = os.path.exists(_img_json) and os.path.exists(_tbl_json)

    # Only run the heavy Docling ML pipeline when it is actually needed:
    #   - text_content is not yet available (no pre-extracted text), OR
    #   - image/table crops have not been extracted yet for this paper.
    if text_content is not None and _images_cached:
        print("[parse_raw] Skipping Docling: pre-extracted text and cached "
              "images/tables are both available.")
        raw_result = None
    else:
        raw_result = doc_converter.convert(raw_source)

    if text_content is None:
        raw_markdown = raw_result.document.export_to_markdown()
        text_content = markdown_clean_pattern.sub("", raw_markdown)

        if len(text_content) < 500:
            if not _MARKER_AVAILABLE or create_model_dict is None:
                raise RuntimeError(
                    "Docling produced < 500 chars and the marker fallback is not installed. "
                    "Install marker-pdf or check the PDF."
                )
            print('\nParsing with docling failed, using marker instead\n')
            parser_model = create_model_dict(device='cuda', dtype=torch.float16)
            text_content, rendered = parse_pdf(raw_source, model_lst=parser_model, save_file=False)

    if version == 1:
        template = Template(open("utils/prompts/gen_poster_raw_content.txt").read())
    elif version == 2:
        template = Template(open("utils/prompts/gen_poster_raw_content_v2.txt").read())

    if args.model_name_t.startswith('vllm_qwen'):
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
            url=actor_config['url'],
        )
    else:
        actor_model = ModelFactory.create(
            model_platform=actor_config['model_platform'],
            model_type=actor_config['model_type'],
            model_config_dict=actor_config['model_config'],
        )

    actor_sys_msg = 'You are the author of the paper, and you will create a poster for the paper.'

    actor_agent = ChatAgent(
        system_message=actor_sys_msg,
        model=actor_model,
        message_window_size=10,
        token_limit=actor_config.get('token_limit', None)
    )

    while True:
        prompt = template.render(
            markdown_document=text_content, 
        )
        actor_agent.reset()
        response = actor_agent.step(prompt)
        input_token, output_token = account_token(response)

        content_json = get_json_from_response(response.msgs[0].content)

        if len(content_json) > 0:
            break
        print('Error: Empty response, retrying...')
        if args.model_name_t.startswith('vllm_qwen'):
            text_content = text_content[:80000]

    if 'sections' not in content_json:
        print('Ouch! The response is missing the "sections" key, the LLM is not following the format :(')
        print('Trying again...')
        raise ValueError("Response is invalid: LLM response has no 'sections' key")

    if len(content_json['sections']) > 9:
        # First 2 sections + randomly select 5 sections + last 2 sections
        selected_sections = content_json['sections'][:2] + random.sample(content_json['sections'][2:-2], 5) + content_json['sections'][-2:]
        content_json['sections'] = selected_sections

    has_title = False

    for section in content_json['sections']:
        if type(section) != dict or not 'title' in section or not 'content' in section:
            print(f"Ouch! The response is invalid, the LLM is not following the format :(")
            print('Trying again...')
            raise ValueError("Response is invalid: LLM is not following the format")
        if 'title' in section['title'].lower():
            has_title = True

    if not has_title:
        print('Ouch! The response is invalid, the LLM is not following the format :(')
        raise ValueError("Response is invalid: no section with 'title' in its title")

    os.makedirs('contents', exist_ok=True)
    json.dump(content_json, open(f'contents/({args.model_name_t}_{args.model_name_v})_{args.poster_name}_raw_content.json', 'w'), indent=4)
    return input_token, output_token, raw_result


def gen_image_and_table(args, conv_res):
    input_token, output_token = 0, 0

    images_json_path = f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}_images.json'
    tables_json_path = f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}_tables.json'

    # conv_res is None when parse_raw() skipped Docling because both pre-extracted
    # text and cached image/table JSONs were already present.
    if conv_res is None:
        print("[gen_image_and_table] Loading cached images/tables from previous run.")
        with open(images_json_path, 'r', encoding='utf-8') as _f:
            images = json.load(_f)
        with open(tables_json_path, 'r', encoding='utf-8') as _f:
            tables = json.load(_f)
        return input_token, output_token, images, tables

    raw_source = args.poster_path

    output_dir = Path(f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}')

    output_dir.mkdir(parents=True, exist_ok=True)
    doc_filename = args.poster_name

    # Save images of figures and tables
    table_counter = 0
    picture_counter = 0
    saved_files: set[str] = set()
    for element, _level in conv_res.document.iterate_items():
        if isinstance(element, TableItem):
            table_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-table-{table_counter}.png"
            )
            img = element.get_image(conv_res.document)
            if img is not None:
                try:
                    with element_image_filename.open("wb") as fp:
                        img.save(fp, "PNG")
                    # Normalise to forward slashes so the membership check below
                    # works on Windows regardless of os.sep.
                    saved_files.add(element_image_filename.as_posix())
                except Exception:
                    pass

        if isinstance(element, PictureItem):
            picture_counter += 1
            element_image_filename = (
                output_dir / f"{doc_filename}-picture-{picture_counter}.png"
            )
            img = element.get_image(conv_res.document)
            if img is not None:
                try:
                    with element_image_filename.open("wb") as fp:
                        img.save(fp, "PNG")
                    saved_files.add(element_image_filename.as_posix())
                except Exception:
                    pass

    # Save markdown with embedded pictures
    md_filename = output_dir / f"{doc_filename}-with-images.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.EMBEDDED)

    # Save markdown with externally referenced pictures
    md_filename = output_dir / f"{doc_filename}-with-image-refs.md"
    conv_res.document.save_as_markdown(md_filename, image_mode=ImageRefMode.REFERENCED)

    # Save HTML with externally referenced pictures
    html_filename = output_dir / f"{doc_filename}-with-image-refs.html"
    conv_res.document.save_as_html(html_filename, image_mode=ImageRefMode.REFERENCED)

    tables = {}

    table_index = 1
    for table in conv_res.document.tables:
        caption = table.caption_text(conv_res.document)
        # Include tables even without a caption; fall back to a generic label so
        # the LLM filter can still reason about them.
        if not caption:
            caption = f"Table {table_index}"
        table_img_path = f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}/{args.poster_name}-table-{table_index}.png'
        if table_img_path in saved_files:
            try:
                table_img = PIL.Image.open(table_img_path)
                tables[str(table_index)] = {
                    'caption': caption,
                    'table_path': table_img_path,
                    'width': table_img.width,
                    'height': table_img.height,
                    'figure_size': table_img.width * table_img.height,
                    'figure_aspect': table_img.width / table_img.height,
                }
            except Exception:
                pass

        table_index += 1

    images = {}
    image_index = 1
    for image in conv_res.document.pictures:
        caption = image.caption_text(conv_res.document)
        # Include images even without a caption; fall back to a generic label.
        if not caption:
            caption = f"Figure {image_index}"
        image_img_path = f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}/{args.poster_name}-picture-{image_index}.png'
        if image_img_path in saved_files:
            try:
                image_img = PIL.Image.open(image_img_path)
                images[str(image_index)] = {
                    'caption': caption,
                    'image_path': image_img_path,
                    'width': image_img.width,
                    'height': image_img.height,
                    'figure_size': image_img.width * image_img.height,
                    'figure_aspect': image_img.width / image_img.height,
                }
            except Exception:
                pass
        image_index += 1

    json.dump(images, open(f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}_images.json', 'w'), indent=4)
    json.dump(tables, open(f'({args.model_name_t}_{args.model_name_v})_images_and_tables/{args.poster_name}_tables.json', 'w'), indent=4)

    return input_token, output_token, images, tables
