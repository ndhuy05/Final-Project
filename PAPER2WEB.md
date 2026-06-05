# Paper2Web — Giải thích chi tiết pipeline

## Tổng quan

Paper2Web là pipeline tự động chuyển đổi một bài báo khoa học (PDF) thành một trang web học thuật hiện đại (single-page HTML). Pipeline kết hợp 4 mô hình LLM với vai trò khác nhau: LLM văn bản để phân tích nội dung, LLM generator chuyên sinh HTML, LLM thị giác (vision) để đánh giá giao diện, và LLM code để tối ưu từng thành phần. Kết quả là một file ZIP chứa trang web hoàn chỉnh có thể triển khai ngay.

---

## Kiến trúc tổng thể

```
HTTP POST /api/v1/notebooks/{notebook_id}/papers/{paper_id}/generate/web
    │
    └── WebAgent (daemon thread)
            │
            ├── Lấy text từ Qdrant → paper_text.txt (fast path, bỏ qua Docling OCR)
            │
            └── subprocess: run_web_job.py  (chạy trong agents/)
                    │
                    ├── Stage 1   — Phân tích văn bản PDF → raw_content.json
                    ├── Stage 1b  — Trích xuất hình ảnh/bảng (Docling) → PNG + images.json
                    ├── Stage 2   — Lọc hình ảnh không liên quan → images_filtered.json
                    ├── Stage 3   — Lên cấu trúc trang web → website_outline.json
                    ├── Stage 4   — Trích xuất link quan trọng → important_info.json
                    ├── Stage 5   — Xây dựng visual_assets (ghép dữ liệu)
                    ├── Stage 6   — Sinh HTML (single-file website) → index.html
                    ├── Stage 7   — Lưu v0 website
                    ├── Stage 8   — Tối ưu hoá vòng lặp (hiện tại max_try=0 → copy trực tiếp)
                    └── Stage 9   — Đóng gói ZIP → {website_name}.zip
```

---

## Cấu hình mô hình

**File:** `backend/app/core/config.py` + `backend/.env`

Pipeline dùng **4 mô hình LLM riêng biệt** cho 4 vai trò khác nhau:

| Setting | Default | Vai trò |
|---------|---------|---------|
| `PAPER2WEB_TEXT_MODEL` | `qwen/qwen3.6-flash` | Phân tích text: parse PDF, lọc hình, lên outline, trích xuất link |
| `PAPER2WEB_GENERATOR_MODEL` | `qwen/qwen3-coder-next` | Sinh HTML website (single-file) |
| `PAPER2WEB_VISION_MODEL` | `qwen/qwen3-vl-32b-instruct` | Đánh giá giao diện bằng screenshot (vòng lặp tối ưu) |
| `PAPER2WEB_CODE_MODEL` | `qwen/qwen3-coder-next` | Tinh chỉnh HTML từng component (vòng lặp tối ưu) |

Tất cả đều đi qua OpenRouter. Các alias được resolve bởi `utils/wei_utils.py::get_agent_config()`.

---

## Entry Point — HTTP API

**File:** `backend/app/routers/web.py`

Có 4 endpoint:

| Method | Path | Mục đích |
|--------|------|----------|
| `POST` | `/notebooks/{notebook_id}/papers/{paper_id}/generate/web` | Bắt đầu pipeline, trả `job_id` ngay |
| `GET` | `/generate/web/{job_id}/status` | Kiểm tra tiến trình (`status`, `progress` 0–1, `step`) |
| `POST` | `/generate/web/{job_id}/cancel` | Yêu cầu huỷ |
| `GET` | `/generate/web/{job_id}/download` | Tải ZIP khi `status == "done"` |

**Luồng khi nhận POST:**
1. Tra cứu paper qua `memory_store.get_paper(notebook_id, paper_id)`.
2. Kiểm tra `WebAgent.is_busy()` → HTTP 409 nếu đang chạy (chỉ 1 job cùng lúc, không có hàng đợi).
3. Lấy đường dẫn PDF từ `settings.UPLOAD_DIR + paper["filename"]` → 404 nếu không có file.
4. Gọi `_web_agent.generate_web(notebook_id, paper_id, paper_title, pdf_path)` → trả `job_id` ngay.
5. Pipeline nặng chạy trong daemon thread; client poll `/status`.

---

## `WebAgent` class

**File:** `backend/app/agents/web_agent.py`

Kế thừa từ `GenerationAgent` (abstract base class dùng chung với `PosterAgent`, `CodeAgent`).

**Cơ chế đồng thời:**
- `_lock: threading.Lock` + `_running_job_id` — chỉ 1 job chạy tại một thời điểm.
- `generate_web()` acquire lock, kiểm tra `_running_job_id`, set nó, release lock, rồi spawn thread.
- `finally` block trong `_run_pipeline()` luôn clear `_running_job_id`.

**Fast path — lấy text từ Qdrant:**

Trước khi spawn subprocess, `generate_web()` gọi:
```python
qdrant_service.get_all_page_texts(notebook_id, paper_id)
```
Nếu có kết quả → ghi toàn bộ text (sắp xếp theo số trang, nối `\n\n`) vào `{OUTPUT_DIR}/{job_id}/paper_text.txt`, truyền `--paper_text_file` cho subprocess.

Lợi ích: bỏ qua bước Docling OCR tốn thời gian (tiết kiệm 2–5 phút). Docling vẫn chạy riêng để trích hình ảnh.

Nếu Qdrant không có text → subprocess tự chạy Docling đầy đủ.

**Subprocess launch:**
- CWD: `settings.PAPER2WEB_DIR` = `backend/app/agents/`
- stdout: piped, đọc từng dòng JSON
- stderr: drained bởi background thread, log là WARNING
- env: kế thừa + `OPENROUTER_API_KEY` + `PYTHONUTF8=1`

**Relay tiến trình:**
- Mỗi dòng JSON từ stdout được parse.
- `{"progress": float, "step": str}` → update `_jobs[job_id]`.
- `{"error": str}` → kill subprocess, set `status=error`.
- `{"done": true, "zip_path": str}` → set `output_path=zip_path`, set `status=done`.

---

## Subprocess Runner (`run_web_job.py`)

**File:** `backend/app/agents/run_web_job.py`

Orchestrator của tất cả stage. Emit JSON-lines ra stdout, flush ngay sau mỗi dòng.

**Lưu ý kỹ thuật quan trọng — 2 namespace `args`:**
- `args.model_name_t` = **slug** của model name (thay `/`, `\` bằng `-`) → dùng làm **prefix thư mục** (vd. `qwen-qwen3-6-flash_images_and_tables/`).
- `args_g.model_name_t` = **alias gốc** (vd. `qwen/qwen3-coder-next`) → dùng để **resolve model** qua `get_agent_config()`.

Trộn 2 cái này sẽ gây lỗi vì `get_agent_config()` không nhận dạng được slug.

**Thư mục tạo lúc khởi động (relative to `agents/`):**
`contents/`, `log/`, `website_outlines/`, `simple_gen_logs/`

**Các mốc tiến trình:**

| Progress | Step |
|----------|------|
| 0.02 | Importing pipeline modules |
| 0.05 | Parsing paper text |
| 0.12 | Extracting figures and tables |
| 0.20 | Filtering figures |
| 0.30 | Planning website outline |
| 0.40 | Extracting key information |
| 0.50 | Loading generated data |
| 0.55 | Generating website HTML |
| 0.65 | Saving v0 website |
| 0.70 | Optimizing website (v1) |
| 0.95 | Packaging ZIP |
| 1.0 | Done |

---

## Các stage chi tiết

### Stage 1 — Phân tích văn bản PDF

**File:** `web_tools/parse_raw.py`, hàm `parse_raw()`

**Tiến trình:** 0.05 → "Parsing paper text…"

**Input:**
- `args.website_path`: đường dẫn PDF
- `pre_extracted_text` (optional): nếu có → bỏ qua Docling hoàn toàn

**Xử lý:**

*Nếu không có `pre_extracted_text`:*
- Chạy Docling với `PdfPipelineOptions(images_scale=5.0, generate_page_images=True, generate_picture_images=True)`.
- Convert PDF → Markdown, strip HTML comments bằng regex.

*Nếu có `pre_extracted_text`:*
- `text_content = pre_extracted_text`, `raw_result = None`. Docling bỏ qua hoàn toàn.

**LLM call:**
- Mô hình: `PAPER2WEB_TEXT_MODEL` (mặc định `qwen/qwen3.6-flash`)
- Template: `utils/prompts/gen_website_raw_content_v2_enhanced.txt` (Jinja2)
- System: `"You are the author of the paper, and you will create a website for the paper."`
- Input: toàn bộ text paper qua `{{ markdown_document }}`
- Output (JSON):
  ```json
  {
    "meta": {
      "website_title": "Tên bài báo",
      "authors": "Tác giả 1, Tác giả 2",
      "affiliations": "Trường đại học X"
    },
    "sections": [
      {"title": "Introduction", "content": "250-600 ký tự tóm tắt..."},
      {"title": "Method", "content": "..."},
      {"title": "Results", "content": "..."}
    ]
  }
  ```
- Retry tối đa 5 lần (`@retry(stop_after_attempt(5))`) nếu JSON không hợp lệ.
- Nếu >9 sections → subsample: 2 đầu + 5 giữa ngẫu nhiên + 2 cuối.

**Files ghi ra:**
- `log/{website_name}_llm_response_N.txt` — prompt + raw response
- `contents/{slugged_model}_{website_name}_raw_content.json`

**Returns:** `(input_tokens, output_tokens, raw_result)` — `raw_result` là docling ConversionResult (hoặc `None`).

---

### Stage 1b — Trích xuất hình ảnh và bảng

**File:** `web_tools/parse_raw.py`, hàm `docling_convert()` + `gen_image_and_table()`

**Tiến trình:** 0.12 → "Extracting figures and tables…". **0 LLM call.**

**`docling_convert(pdf_path)`:**
- Chạy Docling (cùng config: scale 5x, page images, picture images).
- Dùng khi `raw_result = None` (fast path) để lấy ConversionResult chỉ phục vụ trích hình.

**`gen_image_and_table(args, conv_res)`:**

1. Tạo thư mục `{slugged_model}_images_and_tables/{website_name}/`.
2. Lưu ảnh từng trang: `{website_name}-{page_no}.png`.
3. Duyệt `conv_res.document.iterate_items()`:
   - `TableItem` → `{website_name}-table-N.png`
   - `PictureItem` → `{website_name}-picture-N.png`
4. Xuất Docling markdown/HTML (chỉ để debug, bị loại ra khỏi ZIP sau).
5. Xây dựng `images` dict: chỉ giữ hình **có caption** (`caption_text()` không rỗng). Ghi `{caption, image_path, width, height, figure_size, figure_aspect}`.
6. Xây dựng `tables` dict: tương tự.

**Files ghi ra:**
- `{slugged_model}_images_and_tables/{website_name}/` — tất cả PNG
- `{slugged_model}_images_and_tables/{website_name}_images.json`
- `{slugged_model}_images_and_tables/{website_name}_tables.json`

**Fallback sau bước này:** Nếu bước lọc (Stage 2) loại bỏ hết hình, runner tự động khôi phục `_images.json` gốc (unfiltered) vào `_images_filtered.json` để các stage sau không bị thiếu hình hoàn toàn.

---

### Stage 2 — Lọc hình ảnh không liên quan

**File:** `web_tools/simple_gen_outline_layout_website.py`, hàm `filter_image_table()`

**Tiến trình:** 0.20 → "Filtering figures…"

**Input:** `_images.json`, `_tables.json`, `raw_content.json` (tìm bằng glob model-agnostic).

**LLM call:**
- Mô hình: `PAPER2WEB_TEXT_MODEL`
- Template: `utils/prompt_templates/website_image_table_filter_agent.yaml`
- System: "Acts as an assistant reviewing paper content and filtering irrelevant images/tables for a project website. Keep all relevant visual elements, no artificial quantity limits."
- Input:
  - `{{ json_content }}` — toàn bộ raw_content JSON
  - `{{ image_information }}` — JSON string tất cả hình với caption + path
  - `{{ table_information }}` — JSON string tất cả bảng
- Output:
  ```json
  {
    "image_information": {"1": {...}, "3": {...}},
    "table_information": {"2": {...}}
  }
  ```

**Files ghi ra:**
- `simple_gen_logs/{website_name}_filter_image_table_{timestamp}.txt` — log prompt + response
- `{images_dir}/{website_name}_images_filtered.json`
- `{images_dir}/{website_name}_tables_filtered.json`

---

### Stage 3 — Lên cấu trúc trang web

**File:** `web_tools/simple_gen_outline_layout_website.py`, hàm `gen_outline_layout_website_simple()`

**Tiến trình:** 0.30 → "Planning website outline…"

**Input:** filtered images/tables JSON, `raw_content.json`.

**Xử lý:**
1. Trích caption-only từ filtered images/tables (bỏ metadata path/kích thước) → giảm kích thước prompt.
2. Gọi LLM để quyết định section nào dùng hình nào.

**LLM call:**
- Mô hình: `PAPER2WEB_TEXT_MODEL`
- Template: `utils/prompt_templates/website_planner_agent.yaml`
- System: "Expert assistant planning website structure. Maps each paper section to zero or more images/tables that best fit it for website layout."
- Input:
  - `{{ json_content }}` — raw_content JSON đầy đủ
  - `{{ image_information }}` — chỉ captions
  - `{{ table_information }}` — chỉ captions
- Output (JSON):
  ```json
  {
    "Introduction": {"image": 1, "reason": "Hình 1 minh hoạ kiến trúc hệ thống"},
    "Results": {"table": 2, "reason": "Bảng 2 so sánh kết quả thực nghiệm"},
    "Method": {"images": [3, 4], "reason": "Hình 3 và 4 mô tả pipeline"}
  }
  ```
  Mỗi hình/bảng chỉ được gán cho một section (không reuse).

**Post-processing:**
- Xây dựng `arranged_images`, `arranged_tables` (resolve ID về full metadata, dedup).
- Xây dựng `website_pages` list: 1 entry/section với `{page_id, section_name, content, text_len, images/tables}`.

**Files ghi ra:**
- `simple_gen_logs/{website_name}_gen_outline_layout_{timestamp}.txt`
- `website_outlines/{slugged_model}_{website_name}_website_outline.json`
  ```json
  {
    "meta": {"website_title": "...", "authors": "...", "affiliations": "..."},
    "pages": [
      {"page_id": 0, "section_name": "Introduction", "content": "...", "image": 1},
      ...
    ],
    "figure_arrangement": {...},
    "arranged_images": {"1": {caption, image_path, width, height, ...}},
    "arranged_tables": {"2": {...}}
  }
  ```

---

### Stage 4 — Trích xuất link quan trọng

**File:** `web_tools/extract_importinfo.py`, hàm `extract_important_info()`

**Tiến trình:** 0.40 → "Extracting key information…"

**Input:** `raw_content.json` (tìm bằng glob).

**LLM call:**
- Mô hình: `PAPER2WEB_TEXT_MODEL`
- Template: `utils/prompt_templates/extract_important_info_agent.yaml`
- System: "Expert research analyst extracting links and URLs from academic papers."
- Input: `{{ input_data }}` — paper title, authors, affiliations, số section
- Output (JSON):
  ```json
  {
    "important_info": [
      {"url": "https://github.com/author/repo", "describe": "Code implementation on GitHub"},
      {"url": "https://arxiv.org/abs/2401.12345", "describe": "arXiv preprint"},
      {"url": "https://huggingface.co/datasets/xxx", "describe": "Dataset on HuggingFace"},
      {"url": "https://paperswithcode.com/sota/...", "describe": "Benchmark results"}
    ]
  }
  ```
- **Trích xuất:** repo code, link arXiv, dataset, benchmark platform, trang tác giả, tool/framework.
- **Loại trừ:** project page của chính bài báo, homepage của platform chung.
- Mô tả tối đa 25 từ.
- Return `(None, None, None)` nếu không tìm thấy `raw_content.json` → runner gọi `sys.exit(1)`.

**Files ghi ra:**
- `contents/{slugged_model}_{website_name}_important_info.json`

---

### Stage 5 — Xây dựng `visual_assets` (ghép dữ liệu)

**Trong `run_web_job.py`** — không có LLM call, chỉ là xử lý dữ liệu.

**Tiến trình:** 0.50 → "Loading generated data…"

Load các file đã sinh và xây dựng cấu trúc `visual_assets` để truyền cho Stage 6:

```python
visual_assets = {
    "meta": {
        "title": raw_content["meta"]["website_title"],
        "authors": raw_content["meta"]["authors"],
        "affiliations": raw_content["meta"]["affiliations"],
        "project_name": website_name,
    },
    "images": [
        {
            "src": "qwen-qwen3-6-flash_images_and_tables/paper/paper-picture-1.png",
            "alt": "Figure 1 caption text",
            "web_width": 800,    # width chuẩn hoá cho web
            "web_height": 600,
            "section": "Introduction",
        },
        ...
    ],
    "tables": [...]
}
```

`visual_assets.images` được xây dựng bằng cách duyệt từng page trong `website_pages`:
- Lấy `images/image/tables/table` field.
- Lookup trong `arranged_images`/`arranged_tables` để lấy full metadata.
- Flatten thành list (mỗi entry có `src`, `alt`, `web_width`, `web_height`, `section`).

---

### Stage 6 — Sinh HTML website

**File:** `web_tools/simple_end_to_end_generator_v1.py`, hàm `generate_website_end_to_end()`

**Tiến trình:** 0.55 → "Generating website HTML…"

Đây là **stage trung tâm** của pipeline — sinh ra toàn bộ single-page HTML website.

**Xử lý trước khi gọi LLM:**
1. Load template `utils/prompt_templates/simple_end_to_end_website_generator_v0.yaml`.
2. Gọi `load_random_html_template()`: **chọn ngẫu nhiên** 1 file `.html` từ `utils/template/` làm design reference. Mỗi lần chạy có thể ra layout khác nhau.
3. Tạo generator agent: `create_generator_agent(args_g.model_name_t)` → `get_agent_config()` → `OpenRouterModel` trực tiếp (không qua `ModelFactory`).

**LLM call:**
- Mô hình: `PAPER2WEB_GENERATOR_MODEL` (mặc định `qwen/qwen3-coder-next`)
- System: `"You are an expert web developer and UI/UX designer specializing in creating beautiful, modern, and interactive academic project websites."`
- Template: prompt dài ~300 dòng, inject:
  - `{{ research_content | tojson }}` — nội dung đầy đủ các section
  - `{{ visual_assets | tojson }}` — tất cả hình/bảng với `src`, `alt`, kích thước
  - `{{ important_info | tojson }}` — tất cả link quan trọng
  - `{{ html_template }}` — file HTML mẫu chọn ngẫu nhiên
  - Các giá trị cụ thể: `{{ visual_assets.meta.title }}`, `{{ visual_assets.meta.authors }}`, …

**Yêu cầu bắt buộc trong prompt (tóm tắt):**

| Yêu cầu | Chi tiết |
|---------|----------|
| Single HTML file | CSS trong `<style>` ở `<head>`, JS trước `</body>` |
| Hero section | Title, authors, affiliations |
| Responsive | CSS Grid/Flexbox, không dùng `width`/`height` HTML attribute cho ảnh |
| Navigation | Smooth scroll + scroll-spy |
| Image gallery | Lightbox interactive |
| Dark/light theme | Toggle button |
| Resources section | Tất cả URL từ `important_info` |
| BibTeX block | Với nút copy |
| Nội dung | Tóm tắt 100–200 từ/section, không copy verbatim |
| Animation | Tham chiếu 10 hiệu ứng CSS cung cấp sẵn |
| DOM | Toàn bộ DOM access trong `DOMContentLoaded` |

- Output (JSON code block):
  ```json
  {"index.html": "<html lang='en'>...</html>"}
  ```
- Retry tối đa 5 lần: `generator_agent.reset()` + `generator_agent.step(prompt)`.
- Validate: `"index.html"` phải tồn tại, nội dung ≥ 1000 ký tự, chứa `<html` và `</html>`.

**Files ghi ra:**
- `log/{website_name}_simple_end_to_end_llm_response.txt` — response + thống kê token

**Returns:** `(input_tokens, output_tokens, {"index.html": "<full html>"})`

---

### Stage 7 — Lưu v0 Website

**File:** `web_tools/simple_end_to_end_generator_v1.py`, hàm `save_website_files()`

**Tiến trình:** 0.65 → "Saving v0 website…". **0 LLM call.**

**Xử lý:**
1. Tạo thư mục `generated_website_{website_name}_simple/`.
2. Ghi `index.html`.
3. Tìm `*_images_and_tables/` directory bằng glob.
4. Sao chép **chỉ những gì cần thiết cho paper này** (tránh cross-paper contamination):
   - Subdirectory: `{images_dir}/{website_name}/` (các PNG của paper này)
   - 4 JSON sidecar: `{website_name}_images.json`, `{website_name}_tables.json`, `{website_name}_images_filtered.json`, `{website_name}_tables_filtered.json`

**Runner tiếp theo:**
- Copy `generated_website_{website_name}_simple/` → `{output_dir}/v0/` bằng `shutil.copytree()`.
- Truyền `v0_dir` (path tuyệt đối) cho Stage 8.

---

### Stage 8 — Tối ưu hoá vòng lặp

**File:** `web_tools/web_link_v3.py`, class `WebsiteIterativeOptimizerV3`

**Tiến trình:** 0.70 → "Optimizing website (v1)…"

**Hiện tại: `max_try=0` (hardcoded trong runner) → không có LLM call, copy trực tiếp.**

**Với `max_try=0` (chế độ sản xuất hiện tại):**
- Gọi `generate_final_version_v3_direct_copy()`:
  1. Copy `index.html` (và `style.css`, `script.js` nếu có) từ `v0/` → `v1_v3/`.
  2. Copy thư mục hình ảnh bằng `ensure_image_directories()`.
  3. Gọi `fix_image_paths_before_screenshot()`: rewrite relative `src` paths để Playwright có thể resolve.
  4. Chụp screenshot `v1_v3/index.html` bằng Playwright (1920×1080) → `v1_v3/final_screenshot_v3.png`.

**Với `max_try > 0` (chế độ tối ưu đầy đủ — hiện tại chưa bật):**

Đây là kiến trúc hoàn chỉnh nhưng chưa kích hoạt trong production:

1. **Chụp screenshot v0** bằng Playwright.
2. **Slice thành components** (`slice_webpage_into_components()`): tìm các CSS selector `"header, nav, section"`, cắt screenshot thành PNG crop cho từng component → lưu vào `v0/slices/`.
3. **Phân loại:** tách `headers[]`, `navs[]`, `sections[]` theo HTML tag.
4. **Vòng lặp tối ưu** (merge count tăng dần: 1, 2, 4, 8, ...):
   - Merge sections theo nhóm `count` phần bằng `merge_html_sections_clean()` (BeautifulSoup).
   - Render HTML mới, chụp screenshot, slice lại.
   - Với **mỗi component** (header, nav, section):
     - **Vision LLM call** (`model_v`): phân tích screenshot crop của component.
       - Template: `head_html_optimization.yaml` / `navigator.html_optimization.yaml` / `content_block_code_optimization.yaml`
       - Input: screenshot JPEG của component
       - Output: nhận xét về UX/design issues
     - **Code LLM call** (`model_c`): rewrite HTML của component dựa trên nhận xét.
       - Output: HTML mới cho component đó
   - Thay thế hoàn toàn headers/navs/sections bằng phiên bản đã tối ưu.
   - Log HTML evolution vào `html_evolution_log_v3.txt`.
5. Chụp screenshot cuối, lưu `final_screenshot_v3.png`.

**Files ghi ra:**
- `{output_dir}/v1_v3/index.html` — website cuối
- `{output_dir}/v1_v3/final_screenshot_v3.png` — Playwright screenshot 1920×1080
- `{output_dir}/optimization_log_v3.txt`
- `{output_dir}/html_evolution_log_v3.txt`

**Returns:** Đường dẫn tuyệt đối đến `{output_dir}/v1_v3/`.

**Tại sao cần Playwright kể cả khi `max_try=0`?**
Playwright được import ở module level trong `wei_utils.py`. Ngoài ra, `final_screenshot_v3.png` vẫn được tạo trong chế độ direct copy. Do đó `playwright install chromium` là bắt buộc dù không có iterative optimization.

---

### Stage 9 — Đóng gói ZIP

**Trong `run_web_job.py`** — inline code, không gọi hàm riêng.

**Tiến trình:** 0.95 → "Packaging ZIP…"

**Xử lý:**
```python
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk(v1_v3_dir):
        # Prune thư mục có tên chứa các pattern này:
        dirs[:] = [d for d in dirs if not any(
            p in d for p in ['with-image-refs', 'with-images', '_artifacts']
        )]
        for file in files:
            # Bỏ qua file .md và file chứa các pattern trên
            if file.endswith('.md') or any(p in file for p in [...]):
                continue
            zf.write(filepath, arcname=relative_path)
```

**Tại sao loại trừ `with-image-refs`, `with-images`, `_artifacts`?**
Đây là các thư mục export artifact của Docling (`save_as_markdown(REFERENCED)`, `save_as_html(REFERENCED)`). Chúng tạo ra đường dẫn file cực dài (>260 ký tự) gây lỗi khi giải nén trên Windows Explorer.

**Files ghi ra:**
- `{output_dir}/{website_name}.zip`

**Emit cuối:**
```json
{"progress": 1.0, "step": "Done", "done": true, "zip_path": "/abs/path/to/website_name.zip"}
```

`WebAgent` nhận → set `status="done"`, `output_path=zip_path` → endpoint `/download` trả `FileResponse`.

---

## Tổng hợp LLM calls

| Stage | Hàm | Mô hình | Input chính | Output | Số lần |
|-------|-----|---------|-------------|--------|--------|
| 1 | `parse_raw` | `TEXT_MODEL` | Full paper text | JSON: `{meta, sections}` | 1 |
| 2 | `filter_image_table` | `TEXT_MODEL` | Sections + tất cả hình/bảng | JSON: filtered images/tables | 1 |
| 3 | `gen_outline_layout_website_simple` | `TEXT_MODEL` | Sections + captions | JSON: section→hình/bảng mapping | 1 |
| 4 | `extract_important_info` | `TEXT_MODEL` | Paper meta + sections | JSON: `[{url, describe}]` | 1 |
| 6 | `generate_website_end_to_end` | `GENERATOR_MODEL` | Sections + visual_assets + links + HTML template | JSON: `{"index.html": "..."}` | 1 (≤5 retry) |
| 8 | Vision analysis (nếu `max_try>0`) | `VISION_MODEL` | Screenshot PNG của component | Nhận xét UX | N×component |
| 8 | Code optimization (nếu `max_try>0`) | `CODE_MODEL` | HTML component + nhận xét | HTML mới | N×component |

**Hiện tại (production):** Chỉ 5 LLM call (Stage 1→4 + Stage 6). Stage 8 không có LLM call vì `max_try=0`.

---

## Cấu trúc file output

```
paper2web_outputs/
└── {job_id}/
    ├── paper_text.txt                     ← text từ Qdrant (nếu fast path)
    ├── v0/                                ← website đầu tiên (trước tối ưu)
    │   ├── index.html
    │   └── {slugged_model}_images_and_tables/
    │       ├── {website_name}/*.png
    │       └── {website_name}_images*.json
    ├── v1_v3/                             ← website cuối
    │   ├── index.html
    │   ├── final_screenshot_v3.png
    │   └── {images_dir}/...
    ├── optimization_log_v3.txt
    ├── html_evolution_log_v3.txt
    └── {website_name}.zip                ← file download

agents/
├── contents/
│   ├── {slugged_model}_{website_name}_raw_content.json
│   └── {slugged_model}_{website_name}_important_info.json
├── {slugged_model}_images_and_tables/
│   ├── {website_name}/
│   │   ├── {website_name}-picture-1.png
│   │   ├── {website_name}-table-1.png
│   │   └── ...
│   ├── {website_name}_images.json
│   ├── {website_name}_tables.json
│   ├── {website_name}_images_filtered.json
│   └── {website_name}_tables_filtered.json
├── website_outlines/
│   └── {slugged_model}_{website_name}_website_outline.json
├── log/
│   ├── {website_name}_llm_response_0.txt
│   └── {website_name}_simple_end_to_end_llm_response.txt
└── simple_gen_logs/
    ├── {website_name}_filter_image_table_{ts}.txt
    └── {website_name}_gen_outline_layout_{ts}.txt
```

---

## Sơ đồ dòng dữ liệu

```
PDF (upload) + Qdrant (text/page đã index)
    │
    ├─ paper_text.txt (từ Qdrant, fast path)
    └─ docling_convert(pdf) (chỉ để lấy hình)
    │
    ▼
[Stage 1] raw_content.json            ← 1 LLM call (TEXT): paper text → sections JSON
    │
[Stage 1b] _images.json               ← 0 LLM: Docling trích PNG từ PDF (có caption)
           _tables.json
    │
[Stage 2] _images_filtered.json       ← 1 LLM call (TEXT): lọc hình không liên quan
          _tables_filtered.json
          [fallback: restore unfiltered nếu lọc hết]
    │
[Stage 3] website_outline.json        ← 1 LLM call (TEXT): gán hình vào section
          arranged_images{}
          arranged_tables{}
    │
[Stage 4] important_info.json         ← 1 LLM call (TEXT): trích link/URL từ paper
    │
[Stage 5] visual_assets{}             ← 0 LLM: ghép outline + arranged images thành dict
    │
[Stage 6] {"index.html": "..."}       ← 1 LLM call (GENERATOR): sinh toàn bộ HTML
          [chọn ngẫu nhiên HTML template từ utils/template/]
    │
[Stage 7] generated_website_*/        ← 0 LLM: ghi index.html + copy PNG
          → copy → {output_dir}/v0/
    │
[Stage 8] {output_dir}/v1_v3/         ← 0 LLM (max_try=0): copy v0 + Playwright screenshot
          index.html
          final_screenshot_v3.png
    │
[Stage 9] {website_name}.zip          ← 0 LLM: ZIP v1_v3, loại artifacts Docling
    │
    ▼
GET /download → FileResponse(zip)
```

---

## So sánh với Paper2Poster và Paper2Code

| Khía cạnh | Paper2Web | Paper2Poster | Paper2Code |
|-----------|-----------|-------------|-----------|
| Output | ZIP (HTML website) | PPTX | ZIP (Python code) |
| Số mô hình LLM | 4 vai trò riêng | 2 vai trò (text + vision) | 1 vai trò |
| LLM call (production) | 5 | 50–100+ | 4+2N |
| Subprocess | Có | Có | Không |
| Docling | Chỉ cho hình | Chỉ cho hình | Không |
| Playwright | Có (screenshot) | Không | Không |
| ML truyền thống | Không | Có (sklearn layout) | Không |
| Song song | Không | Có (4 thread Stage 6) | Không |
| Concurrency | 1 job cùng lúc | 1 job cùng lúc | Nhiều job OK |
| Template ngẫu nhiên | Có (HTML reference template) | Không | Không |
| Vòng lặp critic | Kiến trúc có, chưa bật | Có + hoạt động | Không |
| Fast path | Có (Qdrant text → bỏ Docling OCR) | Có (Qdrant text → bỏ Docling OCR) | Không (Qdrant only) |

---

## Các điểm quan trọng cần lưu ý

1. **2 namespace `args` không được trộn lẫn:** `args.model_name_t` là slug (filesystem-safe), `args_g.model_name_t` là alias gốc. `get_agent_config()` chỉ nhận alias gốc — truyền slug sẽ gây KeyError âm thầm.

2. **`max_try=0` là hardcoded:** Toàn bộ `WebsiteIterativeOptimizerV3` với vision + code LLM là kiến trúc hoàn chỉnh nhưng chưa kích hoạt trong sản xuất. Muốn bật phải sửa constant trong `run_web_job.py`.

3. **Playwright bắt buộc kể cả khi không optimize:** `playwright install chromium` phải chạy sau `pip install` vì được import ở module level (`wei_utils.py`) và được dùng để chụp `final_screenshot_v3.png` dù `max_try=0`.

4. **HTML template random:** Mỗi lần chạy Paper2Web trên cùng 1 paper có thể sinh ra layout khác do `load_random_html_template()` chọn ngẫu nhiên. Đây là behavior by design.

5. **Glob model-agnostic:** Các Stage 2, 3, 4, 6 tìm file bằng glob pattern như `contents/*_{website_name}_raw_content.json` thay vì đường dẫn cứng → không bị phá vỡ khi đổi model.

6. **ZIP loại trừ Docling artifacts:** `with-image-refs/`, `with-images/`, `_artifacts/` bị loại trừ do path >260 ký tự gây lỗi trên Windows Explorer.

7. **Text bắt buộc từ Qdrant:** Nếu Qdrant không có text (paper chưa index), `paper_text.txt` không được tạo và subprocess phải tự chạy Docling đầy đủ — mất 2–5 phút và cần transformers đã cài.

8. **`extract_important_info` fail → exit(1):** Stage 4 không có retry. Nếu LLM trả JSON sai format hoặc `raw_content.json` không tìm thấy → runner gọi `sys.exit(1)` ngay → job set `status=error`.

9. **Sao chép hình tránh contamination:** `save_website_files()` chỉ copy subdirectory `{website_name}/` và 4 JSON sidecar của paper đang xử lý. Nếu copy toàn bộ `*_images_and_tables/` sẽ lẫn hình của các paper khác vào ZIP.
