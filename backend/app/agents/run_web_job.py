"""
Standalone runner for the Paper2Web pipeline.

Called as a subprocess by web_agent.py with CWD set to the agents/ directory.
Progress is emitted to stdout as JSON lines so the parent process can poll and
update the in-memory job dict without any shared state beyond stdout.

Pipeline stages:
  1. parse_raw              - PDF → raw_content JSON
  2. gen_image_and_table    - extract figure/table images from docling result
  3. filter_image_table     - LLM filters unnecessary figures
  4. gen_outline_layout     - website outline JSON (1 LLM call)
  5. extract_important_info - key facts JSON (1 LLM call)
  6. generate_website_end_to_end - LLM generates index.html
  7. save_website_files     - write v0/ dir
  8. WebsiteIterativeOptimizerV3 (max_try=0) - copy v0 → v1_v3/
  9. ZIP v1_v3/ → {output_dir}/{website_name}.zip

Usage (run from agents/):
    python run_web_job.py \\
        --pdf_path /abs/path/to/paper.pdf \\
        --website_name my_paper \\
        --output_dir /abs/path/to/output \\
        --model_t openrouter_qwen3_30b_a3b \\
        --model_g openrouter_qwen3_coder \\
        --model_v openrouter_qwen2_5_VL_72B \\
        --model_c openrouter_qwen3_coder

Exits 0 on success, 1 on any unhandled error.
"""

import argparse
import glob
import json
import logging
import os
import re
import shutil
import sys
import warnings
import zipfile
from types import SimpleNamespace

warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*use_fast.*", category=UserWarning)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_web_job")

logging.getLogger("root").addFilter(
    lambda r: "Invalid or missing `max_tokens`" not in r.getMessage()
)
logging.getLogger().addFilter(
    lambda r: "Invalid or missing `max_tokens`" not in r.getMessage()
)


def emit(obj: dict) -> None:
    """Write a JSON progress message to stdout (line-buffered)."""
    print(json.dumps(obj), flush=True)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="Paper2Web pipeline runner")
    parser.add_argument("--pdf_path",     required=True,  help="Absolute path to paper.pdf")
    parser.add_argument("--website_name", required=True,  help="Short slug for intermediate filenames")
    parser.add_argument("--output_dir",   required=True,  help="Directory where the ZIP is written")
    parser.add_argument("--model_t",        default="openrouter_qwen3_30b_a3b", help="Text model alias (parse/filter/outline/extract)")
    parser.add_argument("--model_g",        default="openrouter_qwen3_coder",   help="Generator model alias (HTML generation)")
    parser.add_argument("--model_v",        default="openrouter_qwen2_5_VL_72B", help="Vision model alias (iterative optimizer)")
    parser.add_argument("--model_c",        default="openrouter_qwen3_coder",   help="Coder model alias (iterative optimizer)")
    parser.add_argument("--paper_text_file", default=None, help="Path to pre-extracted paper text file; skips docling OCR and figure extraction")
    cli = parser.parse_args()

    os.makedirs(cli.output_dir, exist_ok=True)

    # Directories expected by pipeline modules (relative to CWD = agents/)
    for d in ("contents", "log", "website_outlines", "simple_gen_logs"):
        os.makedirs(d, exist_ok=True)

    # --- sys.path setup: agents/ is the single root for all modules ---
    # camel/ must NOT be added: agents/camel/types/ would shadow stdlib types,
    # causing a circular import. camel.* imports work via agents/ already.
    agents_dir = os.path.dirname(os.path.abspath(__file__))
    if agents_dir not in sys.path:
        sys.path.insert(0, agents_dir)

    def _slug(name: str) -> str:
        """Replace path-unsafe characters with hyphens."""
        return re.sub(r'[/\\:*?"<>|]', '-', name)

    # --- Import pipeline modules ---
    emit({"progress": 0.02, "step": "Importing pipeline modules\u2026"})
    try:
        from web_tools.parse_raw import parse_raw, gen_image_and_table
        from web_tools.simple_gen_outline_layout_website import (
            filter_image_table, gen_outline_layout_website_simple,
        )
        from web_tools.extract_importinfo import extract_important_info
        from web_tools.simple_end_to_end_generator_v1 import (
            generate_website_end_to_end, save_website_files,
        )
        from web_tools.web_link_v3 import WebsiteIterativeOptimizerV3
        from utils.wei_utils import get_agent_config
    except Exception as exc:
        emit({"error": f"Import failed: {exc}"})
        raise

    # Build agent config for text/parse stages (use original alias before slugging)
    agent_config_t = get_agent_config(cli.model_t)

    # args for stages 1–5 (parse / filter / outline / extract)
    # model_name_t is slugged for safe use in file/directory names
    args = SimpleNamespace(
        website_path=cli.pdf_path,
        website_name=cli.website_name,
        model_name_t=_slug(cli.model_t),
        model_name_v=_slug(cli.model_v),
    )

    # args_g for stages 6–7 (generate / save)
    # model_name_t must be the original alias so create_generator_agent() resolves it
    args_g = SimpleNamespace(
        website_path=cli.pdf_path,
        website_name=cli.website_name,
        model_name_t=cli.model_g,
        model_name_v=cli.model_v,
    )

    # --- Stage 1: Parse PDF text ---
    emit({"progress": 0.05, "step": "Parsing paper text\u2026"})
    if cli.paper_text_file:
        with open(cli.paper_text_file, "r", encoding="utf-8") as _f:
            _pre_text = _f.read()
        _in, _out, raw_result = parse_raw(args, agent_config_t, version=2, pre_extracted_text=_pre_text)
        # Skip figure/table extraction — downstream stages handle empty dicts gracefully
        images, tables = {}, {}
    else:
        _in, _out, raw_result = parse_raw(args, agent_config_t, version=2)

        # --- Stage 2: Extract figures/tables ---
        emit({"progress": 0.12, "step": "Extracting figures and tables\u2026"})
        _, _, images, tables = gen_image_and_table(args, raw_result)

    # --- Stage 3: Filter figures ---
    emit({"progress": 0.20, "step": "Filtering figures\u2026"})
    if images or tables:
        filter_image_table(args, agent_config_t)
    else:
        # No captioned figures — write empty filtered JSONs so downstream stages don't crash
        _img_dir = f'{args.model_name_t}_images_and_tables'
        os.makedirs(_img_dir, exist_ok=True)
        with open(f'{_img_dir}/{args.website_name}_images_filtered.json', 'w') as _f:
            json.dump({}, _f)
        with open(f'{_img_dir}/{args.website_name}_tables_filtered.json', 'w') as _f:
            json.dump({}, _f)

    # --- Fallback: if filter removed all images, restore unfiltered so the outline
    #     and HTML generator receive real figure paths instead of using placeholders ---
    _img_dir = f'{args.model_name_t}_images_and_tables'
    _filt_path = f'{_img_dir}/{args.website_name}_images_filtered.json'
    _orig_path = f'{_img_dir}/{args.website_name}_images.json'
    if os.path.exists(_filt_path) and os.path.exists(_orig_path):
        with open(_filt_path) as _f:
            _filt_data = json.load(_f)
        if not _filt_data:
            with open(_orig_path) as _f:
                _orig_data = json.load(_f)
            if _orig_data:
                shutil.copy2(_orig_path, _filt_path)
                logger.warning(
                    "filter_image_table removed all %d images; restored unfiltered as fallback.",
                    len(_orig_data),
                )

    # --- Stage 4: Generate website outline ---
    emit({"progress": 0.30, "step": "Planning website outline\u2026"})
    gen_outline_layout_website_simple(args, agent_config_t)

    # --- Stage 5: Extract important information ---
    emit({"progress": 0.40, "step": "Extracting key information\u2026"})
    _, _, important_info = extract_important_info(args, agent_config_t)
    if important_info is None:
        emit({"error": "extract_important_info returned None"})
        sys.exit(1)

    # --- Load intermediate data for the generation stage ---
    emit({"progress": 0.50, "step": "Loading generated data\u2026"})

    content_files = (
        glob.glob(f'contents/*_{args.website_name}_raw_content.json') +
        glob.glob(f'*_{args.website_name}_raw_content.json')
    )
    if not content_files:
        emit({"error": "raw_content.json not found after parse_raw stage"})
        sys.exit(1)
    with open(content_files[0], 'r', encoding='utf-8') as _f:
        research_content = json.load(_f)

    outline_files = glob.glob(f'website_outlines/*_{args.website_name}_website_outline.json')
    if not outline_files:
        emit({"error": "website_outline.json not found after gen_outline stage"})
        sys.exit(1)
    with open(outline_files[0], 'r', encoding='utf-8') as _f:
        visual_assets_raw = json.load(_f)

    # Construct visual_assets (mirrors simple_end_to_end_generator_v1.py:main())
    visual_assets = {
        "meta": {
            "title": research_content.get("meta", {}).get("website_title", ""),
            "authors": research_content.get("meta", {}).get("authors", ""),
            "affiliations": research_content.get("meta", {}).get("affiliations", ""),
            "project_name": cli.website_name,
        },
        "images": [],
        "tables": [],
    }
    for page in visual_assets_raw.get("pages", []):
        images_field = page.get("images")
        if images_field is not None:
            id_list = images_field if isinstance(images_field, list) else [images_field]
            for img_id in id_list:
                img_info = visual_assets_raw.get("arranged_images", {}).get(str(img_id))
                if img_info:
                    visual_assets["images"].append({
                        "id": str(img_id),
                        "src": img_info.get("image_path", ""),
                        "alt": img_info.get("caption", ""),
                        "web_width": img_info.get("width", 800),
                        "web_height": img_info.get("height", 600),
                    })
        tables_field = page.get("tables")
        if tables_field is not None:
            id_list = tables_field if isinstance(tables_field, list) else [tables_field]
            for tbl_id in id_list:
                tbl_info = visual_assets_raw.get("arranged_tables", {}).get(str(tbl_id))
                if tbl_info:
                    visual_assets["tables"].append({
                        "id": str(tbl_id),
                        "src": tbl_info.get("table_path", ""),
                        "alt": tbl_info.get("caption", ""),
                    })

    # --- Stage 6: Generate website HTML ---
    emit({"progress": 0.55, "step": "Generating website HTML\u2026"})
    _in, _out, website_code = generate_website_end_to_end(
        args_g, research_content, visual_assets, important_info
    )
    if website_code is None:
        emit({"error": "generate_website_end_to_end returned None — LLM failed to produce valid HTML"})
        sys.exit(1)

    # --- Stage 7: Save v0 website ---
    emit({"progress": 0.65, "step": "Saving v0 website\u2026"})
    # save_website_files returns a relative dir name like "generated_website_{name}_simple"
    v0_src_name = save_website_files(args_g, website_code)
    v0_dir = os.path.join(cli.output_dir, "v0")
    if os.path.exists(v0_dir):
        shutil.rmtree(v0_dir)
    shutil.copytree(v0_src_name, v0_dir)

    # --- Stage 8: Iterative optimization (max_try=0 → direct copy v0 → v1_v3/) ---
    emit({"progress": 0.70, "step": "Optimizing website (v1)\u2026"})
    optimizer = WebsiteIterativeOptimizerV3(
        v0_dir=os.path.abspath(v0_dir),
        vision_model=cli.model_v,
        code_model=cli.model_c,
        max_try=0,
    )
    v1_v3_dir = optimizer.run_iterative_optimization()  # str path to {output_dir}/v1_v3/

    # --- Stage 9: Package ZIP ---
    emit({"progress": 0.95, "step": "Packaging ZIP\u2026"})
    zip_path = os.path.join(cli.output_dir, cli.website_name) + '.zip'

    # Docling's save_as_markdown(REFERENCED) creates deeply-nested `_artifacts/`
    # directories with hash-named images alongside `*-with-image-refs.md` /
    # `*-with-image-refs.html` exports.  These paths exceed 260 chars inside the
    # archive, which causes the Windows built-in zip extractor to refuse to open
    # the file.  We build the zip manually and skip those files entirely — they
    # are not referenced by index.html.
    _EXCLUDE_CONTAINS = ('with-image-refs', 'with-images', '_artifacts')
    _EXCLUDE_EXT = {'.md'}

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as _zf:
        for _root, _dirs, _files in os.walk(v1_v3_dir):
            # Prune artifact directories from traversal so os.walk won't descend
            _dirs[:] = [d for d in _dirs if not any(p in d for p in _EXCLUDE_CONTAINS)]
            for _fname in _files:
                _full = os.path.join(_root, _fname)
                _rel  = os.path.relpath(_full, v1_v3_dir)
                if any(p in _rel for p in _EXCLUDE_CONTAINS):
                    continue
                if os.path.splitext(_fname)[1].lower() in _EXCLUDE_EXT:
                    continue
                _zf.write(_full, _rel)

    emit({"progress": 1.0, "step": "Done", "done": True, "zip_path": zip_path})


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        emit({"error": str(exc), "traceback": traceback.format_exc()})
        sys.exit(1)
