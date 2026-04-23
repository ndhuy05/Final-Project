"""
Standalone runner for the PosterAgent pipeline (new fast flow).

Called as a subprocess by paper2poster_service.py with CWD set to the
poster_pipeline/ directory.  Progress is emitted to stdout as JSON lines so
the parent process can poll and update the in-memory job dict without any
shared state or IPC beyond stdout.

New pipeline stages (~18-30 LLM calls, ~15-30 min):
  1. parse_raw              - Docling PDF → raw_content JSON
  2. gen_image_and_table    - extract figure/table images
  3. filter_image_table     - LLM filters unnecessary figures
  4. gen_outline_layout_v2  - 1 LLM call (figure placement JSON)
  5. main_train             - sklearn fit on XML poster dataset (0 LLM)
  6. main_inference         - recursive binary tree-split layout (0 LLM)
  7. load_poster_yaml_config - optional YAML config with fallback defaults
  8. gen_bullet_point_content - N parallel LLM calls (replaces old stages 5-8)
  9. apply_all_styles       - color/style injection (0 LLM)
  10. generate_poster_code  - code assembly (0 LLM)
  11. run_code(poster_code) - execute generated PPTX code
  12. Copy poster.pptx to output_dir

Usage (run from poster_pipeline/):
    python run_poster_job.py \\
        --pdf_path /abs/path/to/paper.pdf \\
        --poster_name my_paper \\
        --output_dir /abs/path/to/output \\
        --model_t openrouter_qwen3 \\
        --model_v openrouter_qwen3 \\
        --tmp_dir /abs/path/to/tmp

The script exits 0 on success and 1 on any unhandled error.
"""

import argparse
import json
import logging
import os
import re
import shutil
import sys
import warnings
from types import SimpleNamespace

# Suppress noisy deprecation warnings from transformers / huggingface
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
warnings.filterwarnings("ignore", message=".*use_fast.*", category=UserWarning)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("run_poster_job")

units_per_inch = 25


def emit(obj: dict) -> None:
    """Write a JSON progress message to stdout (line-buffered)."""
    print(json.dumps(obj), flush=True)


def main() -> None:  # noqa: C901
    parser = argparse.ArgumentParser(description="PosterAgent pipeline runner (new fast flow)")
    parser.add_argument("--pdf_path",    required=True,  help="Absolute path to paper.pdf")
    parser.add_argument("--poster_name", required=True,  help="Short slug used in intermediate filenames")
    parser.add_argument("--output_dir",  required=True,  help="Directory where poster.pptx is written")
    parser.add_argument("--model_t",     default="openrouter_qwen3", help="Text-model alias for get_agent_config()")
    parser.add_argument("--model_v",     default="openrouter_qwen3", help="Vision-model alias for get_agent_config()")
    parser.add_argument("--tmp_dir",     default=None,   help="Abs path for intermediate PPTX/JPG scratch files")
    parser.add_argument("--index",       type=int, default=0)
    parser.add_argument("--max_workers", type=int, default=4)
    parser.add_argument(
        "--preextracted_text_path",
        default=None,
        help="Path to a JSON file mapping page_num (str) -> page_text, "
             "produced by the parent service from Qdrant. When supplied, "
             "parse_raw() uses this instead of running VLM text extraction.",
    )
    cli = parser.parse_args()

    # Default tmp_dir to an absolute path inside output_dir
    if cli.tmp_dir is None:
        cli.tmp_dir = os.path.join(cli.output_dir, "tmp")

    # --- Ensure required output dirs exist ---
    os.makedirs(cli.output_dir, exist_ok=True)
    os.makedirs(cli.tmp_dir,    exist_ok=True)
    os.makedirs("contents",     exist_ok=True)
    os.makedirs("outlines",     exist_ok=True)
    os.makedirs("tree_splits",  exist_ok=True)
    os.makedirs("checkpoints",  exist_ok=True)
    os.makedirs("log",          exist_ok=True)
    os.makedirs("tmp",          exist_ok=True)

    # --- Add local camel/ to sys.path before importing CAMEL modules ---
    repo_root = os.path.dirname(os.path.abspath(__file__))
    camel_dir = os.path.join(repo_root, "camel")
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    if camel_dir not in sys.path:
        sys.path.append(camel_dir)

    # --- Import pipeline modules ---
    emit({"progress": 0.02, "step": "Importing pipeline modules\u2026"})
    try:
        from PosterAgent.parse_raw import parse_raw, gen_image_and_table
        from PosterAgent.gen_outline_layout import filter_image_table, gen_outline_layout_v2
        from PosterAgent.tree_split_layout import (
            main_train, main_inference, get_arrangments_in_inches,
            split_textbox, to_inches
        )
        from PosterAgent.gen_pptx_code import generate_poster_code
        from PosterAgent.gen_poster_content import gen_bullet_point_content
        from utils.wei_utils import (
            get_agent_config, utils_functions, run_code,
            scale_to_target_area, char_capacity
        )
        from utils.config_utils import (
            load_poster_yaml_config, extract_font_sizes, extract_colors,
            extract_vertical_alignment, extract_section_title_symbol,
            normalize_config_values
        )
        from utils.style_utils import apply_all_styles
        from utils.theme_utils import get_default_theme, create_theme_with_alignment, resolve_colors
    except Exception as exc:
        emit({"error": f"Import failed: {exc}"})
        raise

    # Sanitize model names for use in file paths: OpenRouter IDs contain '/'
    # (e.g. "openai/gpt-4o-mini") which would be interpreted as directory
    # separators and cause FileNotFoundError when building intermediate paths.
    def _slug(name: str) -> str:
        """Replace any path-unsafe characters with hyphens."""
        return re.sub(r'[/\\:*?"<>|]', '-', name)

    # Build agent configs BEFORE sanitising so the full model ID is used.
    agent_config_t = get_agent_config(cli.model_t)
    agent_config_v = get_agent_config(cli.model_v)

    # --- Build the args namespace expected by the pipeline stages ---
    args = SimpleNamespace(
        poster_path=cli.pdf_path,
        model_name_t=_slug(cli.model_t),
        model_name_v=_slug(cli.model_v),
        poster_name=cli.poster_name,
        tmp_dir=cli.tmp_dir,
        index=cli.index,
        max_workers=cli.max_workers,
        max_retry=3,
        preextracted_text_path=cli.preextracted_text_path,
        ablation_no_tree_layout=False,
        ablation_no_commenter=False,
        ablation_no_example=False,
        no_blank_detection=False,
        estimate_chars=False,
        poster_width_inches=None,
        poster_height_inches=None,
    )

    # --- Determine poster dimensions ---
    poster_width  = 48 * units_per_inch
    poster_height = 36 * units_per_inch
    poster_width, poster_height = scale_to_target_area(poster_width, poster_height)
    poster_width_inches  = to_inches(poster_width,  units_per_inch)
    poster_height_inches = to_inches(poster_height, units_per_inch)

    # Clamp to max 56 inches on any side
    if poster_width_inches > 56 or poster_height_inches > 56:
        if poster_width_inches >= poster_height_inches:
            scale_factor = 56 / poster_width_inches
        else:
            scale_factor = 56 / poster_height_inches
        poster_width_inches  *= scale_factor
        poster_height_inches *= scale_factor
        poster_width  = poster_width_inches  * units_per_inch
        poster_height = poster_height_inches * units_per_inch

    emit({"progress": 0.03, "step": f"Poster size: {poster_width_inches:.1f} x {poster_height_inches:.1f} inches"})

    # --- Stage 1: Parse PDF with Docling ---
    emit({"progress": 0.05, "step": "Parsing PDF with Docling\u2026"})
    _input_tok, _output_tok, raw_result = parse_raw(args, agent_config_t, version=2)
    _, _, images, tables = gen_image_and_table(args, raw_result)

    # --- Stage 2: Filter unnecessary figures/tables ---
    emit({"progress": 0.15, "step": "Filtering figures\u2026"})
    _input_tok, _output_tok = filter_image_table(args, agent_config_t)

    # --- Stage 3: Generate outline (1 LLM call) ---
    emit({"progress": 0.22, "step": "Generating outline layout (v2)\u2026"})
    _input_tok, _output_tok, panels, figures = gen_outline_layout_v2(args, agent_config_t)

    # --- Stage 4: Tree-split layout (0 LLM calls) ---
    emit({"progress": 0.30, "step": "Training layout models\u2026"})
    panel_model_params, figure_model_params = main_train()

    emit({"progress": 0.33, "step": "Running tree-split layout inference\u2026"})
    panel_arrangement, figure_arrangement, text_arrangement = main_inference(
        panels,
        panel_model_params,
        figure_model_params,
        poster_width,
        poster_height,
        shrink_margin=3
    )

    # Split the title textbox into title + author rows
    text_arrangement_title = text_arrangement[0]
    text_arrangement = text_arrangement[1:]
    text_arrangement_title_top, text_arrangement_title_bottom = split_textbox(
        text_arrangement_title, 0.8
    )
    text_arrangement = [text_arrangement_title_top, text_arrangement_title_bottom] + text_arrangement

    # Attach figure paths to figure_arrangement entries
    for i in range(len(figure_arrangement)):
        panel_id = figure_arrangement[i]['panel_id']
        panel_section_name = panels[panel_id]['section_name']
        figure_info = figures.get(panel_section_name, {})
        figure_path = None
        if 'image' in figure_info:
            figure_id = figure_info['image']
            img_entry = images.get(figure_id) or images.get(str(figure_id))
            if img_entry:
                figure_path = img_entry['image_path']
        elif 'table' in figure_info:
            figure_id = figure_info['table']
            tbl_entry = tables.get(figure_id) or tables.get(str(figure_id))
            if tbl_entry:
                figure_path = tbl_entry['table_path']
        if figure_path:
            figure_arrangement[i]['figure_path'] = figure_path

    # Compute char capacities for text arrangements
    for text_arrangement_item in text_arrangement:
        num_chars = char_capacity(
            bbox=(
                text_arrangement_item['x'],
                text_arrangement_item['y'],
                text_arrangement_item['height'],
                text_arrangement_item['width']
            )
        )
        text_arrangement_item['num_chars'] = num_chars

    # Convert to inches and save tree-split JSON
    width_inch, height_inch, panel_arrangement_inches, figure_arrangement_inches, text_arrangement_inches = \
        get_arrangments_in_inches(
            poster_width, poster_height,
            panel_arrangement, figure_arrangement, text_arrangement,
            units_per_inch
        )

    tree_split_results = {
        'poster_width': poster_width,
        'poster_height': poster_height,
        'poster_width_inches': width_inch,
        'poster_height_inches': height_inch,
        'panels': panels,
        'panel_arrangement': panel_arrangement,
        'figure_arrangement': figure_arrangement,
        'text_arrangement': text_arrangement,
        'panel_arrangement_inches': panel_arrangement_inches,
        'figure_arrangement_inches': figure_arrangement_inches,
        'text_arrangement_inches': text_arrangement_inches,
    }
    tree_split_path = (
        f'tree_splits/({args.model_name_t}_{args.model_name_v})'
        f'_{args.poster_name}_tree_split_{args.index}.json'
    )
    with open(tree_split_path, 'w') as f:
        json.dump(tree_split_results, f, indent=4)

    # --- Stage 5: Load YAML config (with graceful fallback if none present) ---
    emit({"progress": 0.36, "step": "Loading poster configuration\u2026"})
    yaml_cfg = load_poster_yaml_config(args.poster_path)

    bullet_fs, title_fs, poster_title_fs, poster_author_fs = extract_font_sizes(yaml_cfg)
    title_text_color, title_fill_color, main_text_color, main_text_fill_color = extract_colors(yaml_cfg)
    section_title_vertical_align = extract_vertical_alignment(yaml_cfg)
    section_title_symbol = extract_section_title_symbol(yaml_cfg)

    (
        bullet_fs, title_fs, poster_title_fs, poster_author_fs,
        title_text_color, title_fill_color, main_text_color, main_text_fill_color
    ) = normalize_config_values(
        bullet_fs, title_fs, poster_title_fs, poster_author_fs,
        title_text_color, title_fill_color, main_text_color, main_text_fill_color
    )

    # Store config values in args so pipeline stages can access them
    args.bullet_font_size            = bullet_fs
    args.section_title_font_size     = title_fs
    args.poster_title_font_size      = poster_title_fs
    args.poster_author_font_size     = poster_author_fs
    args.title_text_color            = title_text_color
    args.title_fill_color            = title_fill_color
    args.main_text_color             = main_text_color
    args.main_text_fill_color        = main_text_fill_color
    args.section_title_vertical_align = section_title_vertical_align

    # --- Stage 6: Generate bullet point content (N parallel LLM calls) ---
    emit({"progress": 0.40, "step": "Generating poster content (parallel)\u2026"})
    _t_in, _t_out, _v_in, _v_out = gen_bullet_point_content(
        args, agent_config_t, agent_config_v, tmp_dir=args.tmp_dir
    )

    # Load the saved bullet_point_content
    bullet_content_path = (
        f'contents/({args.model_name_t}_{args.model_name_v})'
        f'_{args.poster_name}_bullet_point_content_{args.index}.json'
    )
    with open(bullet_content_path, 'r') as _bcf:
        bullet_content = json.load(_bcf)

    # --- Stage 7: Apply styles (0 LLM) ---
    emit({"progress": 0.88, "step": "Applying styles\u2026"})
    final_title_text_color, final_title_fill_color, final_main_text_color, final_main_text_fill_color = \
        resolve_colors(
            getattr(args, 'title_text_color', None),
            getattr(args, 'title_fill_color', None),
            getattr(args, 'main_text_color', None),
            getattr(args, 'main_text_fill_color', None)
        )

    bullet_content = apply_all_styles(
        bullet_content,
        title_text_color=final_title_text_color,
        title_fill_color=final_title_fill_color,
        main_text_color=final_main_text_color,
        main_text_fill_color=final_main_text_fill_color,
        section_title_symbol=section_title_symbol,
        main_text_font_size=bullet_fs
    )

    # --- Stage 8: Generate PPTX code (0 LLM) ---
    emit({"progress": 0.92, "step": "Generating PowerPoint code\u2026"})
    base_theme = get_default_theme()
    theme_with_alignment = create_theme_with_alignment(
        base_theme,
        getattr(args, 'section_title_vertical_align', None)
    )

    poster_code = generate_poster_code(
        panel_arrangement_inches,
        text_arrangement_inches,
        figure_arrangement_inches,
        presentation_object_name='poster_presentation',
        slide_object_name='poster_slide',
        utils_functions=utils_functions,
        slide_width=width_inch,
        slide_height=height_inch,
        img_path=None,
        save_path=f'{args.tmp_dir}/poster.pptx',
        visible=False,
        content=bullet_content,
        theme=theme_with_alignment,
        tmp_dir=args.tmp_dir,
    )

    # --- Stage 9: Run generated code to produce poster.pptx ---
    emit({"progress": 0.95, "step": "Running poster code\u2026"})
    output, err = run_code(poster_code)
    if err is not None:
        emit({"error": f"run_code failed: {err}"})
        sys.exit(1)

    # --- Stage 10: Copy final poster to output directory ---
    emit({"progress": 0.97, "step": "Saving poster\u2026"})
    src_pptx = os.path.join(args.tmp_dir, "poster.pptx")
    if not os.path.exists(src_pptx):
        emit({"error": "poster.pptx was not produced by the pipeline."})
        sys.exit(1)

    output_pptx = os.path.join(cli.output_dir, f"{cli.poster_name}.pptx")
    shutil.copy2(src_pptx, output_pptx)

    emit({"progress": 1.0, "step": "Done", "done": True, "pptx_path": output_pptx})


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        import traceback
        emit({"error": str(exc), "traceback": traceback.format_exc()})
        sys.exit(1)
