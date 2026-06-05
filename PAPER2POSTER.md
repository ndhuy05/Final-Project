# Paper2Poster — Giải thích chi tiết pipeline

## Tổng quan

Paper2Poster là pipeline chuyển đổi một bài báo khoa học (PDF) thành một poster học thuật định dạng `.pptx` một cách tự động. Pipeline kết hợp nhiều mô hình AI: LLM văn bản để tóm tắt và viết nội dung, LLM thị giác (vision) để kiểm tra bố cục trực quan, và các mô hình máy học truyền thống (sklearn) để bố trí panel tự động.

---

## Kiến trúc tổng thể

```
HTTP POST /api/v1/notebooks/{notebook_id}/papers/{paper_id}/generate/poster
    │
    ├── PosterAgent (daemon thread)
    │       │
    │       └── subprocess: run_poster_job.py  (chạy trong agents/)
    │               │
    │               ├── Stage 1  — Phân tích văn bản PDF
    │               ├── Stage 1b — Trích xuất hình ảnh/bảng (Docling)
    │               ├── Stage 2  — Lọc hình ảnh không liên quan
    │               ├── Stage 3  — Lên bố cục tổng thể (gán hình cho section)
    │               ├── Stage 4  — Tính toán vị trí panel (thuật toán tree-split)
    │               ├── Stage 5  — Đọc cấu hình YAML (font, màu sắc)
    │               ├── Stage 6  — Sinh nội dung bullet point (song song + critic vòng lặp)
    │               ├── Stage 7  — Áp dụng style
    │               ├── Stage 8  — Sinh code Python tạo PPTX
    │               ├── Stage 9  — Chạy code Python (exec)
    │               └── Stage 10 — Sao chép file .pptx ra thư mục output
    │
    └── FileResponse → {job_id}/{poster_name}.pptx
```

---

## Entry Point — HTTP API

**File:** `backend/app/routers/poster.py`

Có 4 endpoint:

| Method | Path | Mục đích |
|--------|------|----------|
| `POST` | `/notebooks/{notebook_id}/papers/{paper_id}/generate/poster` | Bắt đầu sinh poster, trả về `job_id` ngay lập tức |
| `GET` | `/generate/poster/{job_id}/status` | Kiểm tra tiến trình (`status`, `progress` 0–1, `step`) |
| `POST` | `/generate/poster/{job_id}/cancel` | Yêu cầu huỷ job |
| `GET` | `/generate/poster/{job_id}/download` | Tải file `.pptx` khi `status == "done"` |

**Luồng khi nhận POST:**
1. Tra cứu paper trong SQLite qua `memory_store.get_paper(notebook_id, paper_id)`.
2. Kiểm tra `PosterAgent.is_busy()` — nếu đang có job khác chạy thì trả HTTP 409 (chỉ 1 job cùng lúc).
3. Lấy đường dẫn PDF từ `settings.UPLOAD_DIR + paper["filename"]`.
4. Gọi `_poster_agent.generate_poster(...)` → trả `job_id` ngay.
5. Phần nặng chạy trong daemon thread, client poll `/status` để theo dõi.

---

## `PosterAgent` class

**File:** `backend/app/agents/poster_agent.py`

Kế thừa từ `GenerationAgent` (abstract base class dùng chung với `CodeAgent` và `WebAgent`).

**Các field quan trọng:**
- `_lock: threading.Lock` — đảm bảo chỉ 1 job khởi động tại một thời điểm.
- `_running_job_id` — UUID của job đang chạy, xóa đi sau khi xong.
- `_jobs: dict` — lưu trạng thái job trong bộ nhớ (không persisted vào DB).

**Luồng thực thi:**
1. `generate_poster()` tạo job_id mới, spawn daemon thread, trả job_id về HTTP handler ngay.
2. `_run_pipeline()` trong thread:
   - Tạo thư mục `{OUTPUT_DIR}/{job_id}/` và `{OUTPUT_DIR}/{job_id}/tmp/`.
   - **Lấy text từ Qdrant**: gọi `qdrant_service.get_all_page_texts(notebook_id, paper_id)`, nếu có thì ghi vào `preextracted_pages.json`.
   - Build lệnh subprocess chạy `run_poster_job.py`.
   - Spawn subprocess với `cwd = agents/`, truyền `OPENROUTER_API_KEY` qua env.
   - Đọc từng dòng JSON từ stdout của subprocess để cập nhật tiến trình.
   - Nếu subprocess emit `{"done": true, "pptx_path": "..."}` → set `status=done`.
   - Nếu emit `{"error": "..."}` → set `status=error`.
   - Kiểm tra `_is_cancelled()` sau mỗi dòng — nếu bị huỷ thì `proc.kill()`.

---

## Các stage chi tiết

### Stage 1 — Phân tích văn bản PDF

**File:** `poster_pipeline/parse_raw.py`, hàm `parse_raw()`

**Tiến trình:** 0.05 → "Parsing paper text…"

**Input:**
- File `preextracted_pages.json`: `{page_num: page_text}` từ Qdrant (text đã được trích xuất trước khi upload).
- Nếu không có file này → raise `RuntimeError` (text là bắt buộc).

**Xử lý:**
1. Load JSON, sắp xếp theo số trang, nối thành một chuỗi văn bản dài.
2. Kiểm tra cache: nếu `{model}_images_and_tables/{poster_name}_images.json` đã tồn tại thì bỏ qua Docling (dùng cache). Ngược lại chạy `doc_converter.convert(pdf_path)` để trích xuất hình ảnh.
3. Load Jinja2 template từ `utils/prompts/gen_poster_raw_content_v2.txt`.
4. Tạo CAMEL `ChatAgent` với system: `"You are the author of the paper, and you will create a poster for the paper."`.
5. Gọi LLM 1 lần, retry tối đa 5 lần nếu JSON không hợp lệ.
6. Nếu trả về >9 sections → subsample: giữ 2 đầu + 5 giữa ngẫu nhiên + 2 cuối.
7. Lưu vào `contents/{model}_{poster_name}_raw_content.json`.

**LLM call:**
- Mô hình: `model_t` (mặc định `qwen/qwen3.6-flash` qua OpenRouter)
- Input: toàn bộ văn bản paper (qua Jinja template)
- Output (JSON):
  ```json
  {
    "meta": {
      "poster_title": "Tên bài báo",
      "authors": "Tác giả 1, Tác giả 2",
      "affiliations": "Trường đại học X"
    },
    "sections": [
      {"title": "Poster Title & Author", "content": "..."},
      {"title": "Introduction", "content": "..."},
      {"title": "Method", "content": "..."}
    ]
  }
  ```

---

### Stage 1b — Trích xuất hình ảnh và bảng

**File:** `poster_pipeline/parse_raw.py`, hàm `gen_image_and_table()`

**0 LLM call — thuần code.**

**Xử lý:**
- Nếu `conv_res` là `None` (cache hit): load `_images.json` và `_tables.json` hiện có, return.
- Ngược lại, duyệt `conv_res.document.iterate_items()`:
  - `TableItem` → lưu PNG vào `{model}_images_and_tables/{poster_name}/{poster_name}-table-N.png`.
  - `PictureItem` → lưu PNG vào `…-picture-N.png`.
- Xây dựng dict `images` và `tables`: mỗi entry gồm `{caption, image_path, width, height, figure_size, figure_aspect}`.
- Ghi `{model}_images_and_tables/{poster_name}_images.json` và `_tables.json`.

**Tại sao dùng Docling thay vì PyMuPDF?**
Docling render từng trang PDF và dùng ML để detect vùng figure (kể cả vector graphics từ matplotlib, TikZ…). PyMuPDF chỉ trích xuất ảnh nhúng dạng raster (xref), bỏ qua tất cả đồ thị vector — là phần lớn hình ảnh trong bài báo khoa học. Ngoài ra, Docling hiểu caption của hình (e.g. "Figure 3: Architecture") giúp bước lọc LLM phía sau hoạt động tốt hơn.

---

### Stage 2 — Lọc hình ảnh không liên quan

**File:** `poster_pipeline/gen_outline_layout.py`, hàm `filter_image_table()`

**Tiến trình:** 0.15 → "Filtering figures…"

**Input:** `_images.json`, `_tables.json`, `_raw_content.json`.

**Xử lý:**
1. Load danh sách hình/bảng.
2. Tính `min_width`, `min_height`, `max_width`, `max_height` chuẩn hóa theo tỉ lệ.
3. Tạo CAMEL `ChatAgent` với system từ `utils/prompt_templates/image_table_filter_agent.yaml`.
4. Gọi LLM 1 lần: truyền section JSON + metadata hình/bảng.
5. Parse response, ghi `_images_filtered.json` và `_tables_filtered.json` (tối đa 5 entry mỗi loại).

**LLM call:**
- Mô hình: `model_t`
- Input: JSON tất cả section + metadata hình/bảng (caption, kích thước)
- Output (JSON):
  ```json
  {
    "image_information": {"1": {...}, "3": {...}},
    "table_information": {"2": {...}}
  }
  ```

---

### Stage 3 — Lên bố cục tổng thể (gán hình cho section)

**File:** `poster_pipeline/gen_outline_layout.py`, hàm `gen_outline_layout_v2()`

**Tiến trình:** 0.22 → "Generating outline layout (v2)…"

**Xử lý:**
1. Tính `tp` (text proportion) cho mỗi section: `len(content) / tổng_chiều_dài_content`.
2. Trích xuất caption từ hình/bảng đã lọc (giảm kích thước prompt).
3. Gọi LLM 1 lần để LLM quyết định section nào dùng hình nào.
4. Parse `figure_arrangement`: `{section_name: {image: N, reason: "..."}}`.
5. Xây dựng `paper_panels`: danh sách dict mỗi section với `{panel_id, section_name, tp, text_len, gp, figure_size, figure_aspect}`.
   - `gp` (graphic proportion) = `figure_size / tổng_diện_tích_hình`.

**LLM call:**
- Mô hình: `model_t`
- Input: section JSON + captions hình/bảng
- Output (JSON):
  ```json
  {
    "Introduction": {"image": 1, "reason": "Hình 1 minh họa kiến trúc"},
    "Results": {"table": 2, "reason": "Bảng 2 so sánh kết quả"}
  }
  ```

---

### Stage 4 — Tính toán vị trí panel (Tree-Split Layout)

**File:** `poster_pipeline/tree_split_layout.py`

**Tiến trình:** 0.30 → 0.36. **0 LLM call — thuật toán ML truyền thống.**

Đây là bước tính bố cục hoàn toàn tự động bằng sklearn.

#### 4a — Huấn luyện mô hình (`main_train()`)
- Đọc tất cả file XML từ `assets/poster_data/Train/` (dữ liệu layout poster học thuật thực tế).
- **Panel model**: 2 `LinearRegression` trên features `[tp, gp, 1]` để predict:
  - `sp` = diện tích panel / diện tích poster.
  - `rp` = width / height của panel.
- **Figure model**: `LogisticRegression` predict canh hình (trái/giữa/phải) + `LinearRegression` predict chiều rộng hình.

#### 4b — Inference và bố cục (`main_inference()`)
1. **Predict panel attrs**: tính `sp` và `rp` cho mỗi panel từ `tp`, `gp`.
2. **Bố cục panel** (`generate_constrained_layout()`):
   - Title panel cố định ở hàng đầu, chiếm 10% chiều cao, toàn bộ chiều rộng.
   - Các panel còn lại bố cục đệ quy (`panel_layout_generation()`): thuật toán binary tree-split thử tất cả vị trí cắt ngang/dọc trong bounding box, chọn cắt nào tối thiểu `|predicted_rp - actual_rp|`.
3. **Đặt text box và hình** (`place_text_and_figures_exact()`):
   - Panel không có hình: 1 hoặc 2 text box xếp dọc.
   - Panel có hình: predict canh trái/giữa/phải bằng LogisticRegression, predict chiều rộng hình bằng LinearRegression, đảm bảo tỉ lệ khung hình, giới hạn chiều cao hình ≤60% panel.
4. **Title split**: text box đầu tiên trong title panel chia 80/20 (title chính / tên tác giả).
5. **Tính `num_chars`**: ước lượng sức chứa ký tự cho mỗi text box.
6. **Chuyển đổi sang inches**: chia cho `units_per_inch = 25`.
7. Lưu vào `tree_splits/{model}_{poster_name}_tree_split_0.json`.

**Kích thước poster mặc định:** 48 × 36 inch (landscape), scale về target area 900×1200 units.

---

### Stage 5 — Đọc cấu hình YAML

**File:** `poster_pipeline/config_utils.py`, hàm `load_poster_yaml_config()`

**Tiến trình:** 0.36 → "Loading poster configuration…". **0 LLM call.**

Tìm `poster.yaml` theo thứ tự:
1. Cùng thư mục với PDF.
2. `config/poster.yaml` (trong `agents/`).
3. `poster.yaml` (trong `agents/`).

Nếu không tìm thấy → dùng giá trị mặc định.

**Các giá trị đọc từ YAML:**
| Key | Mô tả | Mặc định |
|-----|-------|----------|
| `main_text_font_size` | Cỡ chữ nội dung | do LLM quyết định |
| `section_title_font_size` | Cỡ chữ tiêu đề section | do LLM quyết định |
| `poster_title_font_size` | Cỡ chữ tiêu đề poster | do LLM quyết định |
| `poster_author_font_size` | Cỡ chữ tác giả | do LLM quyết định |
| `title_text_color` | Màu chữ tiêu đề (RGB) | `[255, 255, 255]` |
| `title_fill_color` | Màu nền tiêu đề (RGB) | `[47, 85, 151]` |
| `main_text_color` | Màu chữ nội dung (RGB) | `[0, 0, 0]` |
| `section_title_vertical_align` | Căn dọc tiêu đề section | `None` |
| `section_title_symbol` | Ký tự prefix tiêu đề | `None` |

**Ví dụ `poster.yaml`:**
```yaml
main_text_font_size: 28
section_title_font_size: 36
poster_title_font_size: 48
poster_author_font_size: 32
title_text_color: [255, 255, 255]
title_fill_color: [47, 85, 151]
main_text_color: [0, 0, 0]
section_title_vertical_align: middle
section_title_symbol: "▶ "
```

---

### Stage 6 — Sinh nội dung bullet point

**File:** `poster_pipeline/gen_poster_content.py`, hàm `gen_bullet_point_content()`

**Tiến trình:** 0.40 → 0.88. **Stage tốn LLM call nhiều nhất.**

Các section từ index 1 trở đi chạy **song song** (tối đa 4 thread). Section 0 (title) chạy tuần tự sau.

#### Xử lý từng section (worker `_process_section(i)`)

**Bước 1 — Actor LLM call (sinh bullet points):**
- Mô hình: `model_t`
- Template: `utils/prompt_templates/bullet_point_agent.yaml`
- Input: nội dung thô của section, số text box (1 hoặc 2), font size.
- Output (JSON):
  ```json
  {
    "title": [{"alignment": "left", "bullet": false, "level": 0, "font_size": 36,
               "runs": [{"text": "Introduction", "bold": true}]}],
    "textbox1": [
      {"alignment": "left", "bullet": true, "level": 0, "font_size": 28,
       "runs": [{"text": "• Điểm chính 1"}]},
      {"alignment": "left", "bullet": true, "level": 1, "font_size": 24,
       "runs": [{"text": "  - Chi tiết phụ"}]}
    ],
    "textbox2": [...]
  }
  ```
  (Chỉ có `textbox2` nếu panel có hình, vì hình chia không gian thành 2 vùng text.)

**Bước 2 — Vòng lặp Critic (kiểm tra bằng vision model):**

Đây là vòng lặp độc đáo và quan trọng nhất của pipeline:

Với mỗi text box:
1. **Render thử**: sinh code PPTX tạm cho 1 text box rồi `exec()`, sau đó chuyển PPTX sang JPEG.
2. **Critic LLM call** (vision model):
   - Mô hình: `model_v` (mặc định `qwen/qwen3-vl-32b-instruct`)
   - Input: 3 ảnh:
     - `assets/overflow_example_v2/neg.jpg` — ví dụ xấu (text tràn ra ngoài)
     - `assets/overflow_example_v2/pos.jpg` — ví dụ tốt (text vừa khớp)
     - Ảnh JPEG của text box vừa render
   - Output: `"1"` (tràn text), `"2"` (quá trống), `"3"` (vừa đẹp).
3. Nếu `"1"` (tràn): yêu cầu Actor rút ngắn, lặp lại. Tối đa 10 vòng.
4. Nếu `"2"` (trống): yêu cầu Actor thêm bullet, lặp lại. Tối đa 10 vòng.
5. Nếu `"3"` hoặc hết 10 vòng: chuyển sang text box tiếp.

**Số LLM call trong stage này:**
- `N` actor call (N = số section không phải title)
- Tối đa `10 × N × num_textboxes` critic/actor call (thực tế thường ít hơn)
- `1` call cho title (tuần tự)

**Sau khi song song xong — sinh title:**
- Template: `utils/prompt_templates/poster_title_agent.yaml`
- Input: chuỗi `meta` (title + authors + affiliations)
- Output (JSON):
  ```json
  {
    "title": [{"alignment": "center", "font_size": 48, "runs": [{"text": "Tên Bài Báo", "bold": true}]}],
    "textbox1": [
      {"alignment": "center", "font_size": 32, "runs": [{"text": "Tác Giả 1, Tác Giả 2"}]},
      {"alignment": "center", "font_size": 32, "runs": [{"text": "Đại Học X, Viện Y"}]}
    ]
  }
  ```

Kết quả ghi vào `contents/{model}_{poster_name}_bullet_point_content_0.json`.

---

### Stage 7 — Áp dụng Style

**File:** `poster_pipeline/style_utils.py`, hàm `apply_all_styles()`

**Tiến trình:** 0.88 → "Applying styles…". **0 LLM call.**

Áp dụng theo thứ tự:
1. **Font size**: ghi đè `font_size` của tất cả paragraph trong textbox1/textbox2 thành `main_text_font_size` (từ YAML).
2. **Section title symbol**: thêm prefix (vd. `"▶ "`) vào `title[0]["runs"][0]["text"]` của mỗi section.
3. **Màu tiêu đề**: áp dụng `title_text_color` và `title_fill_color` lên title section và tiêu đề từng section.
4. **Màu nội dung**: áp dụng `main_text_color` và `main_text_fill_color` lên textbox1/textbox2.

---

### Stage 8 — Sinh code Python tạo PPTX

**File:** `poster_pipeline/gen_pptx_code.py`, hàm `generate_poster_code()`

**Tiến trình:** 0.92 → "Generating PowerPoint code…". **0 LLM call.**

Sinh ra một chuỗi Python code bằng cách nối string. Code này khi chạy sẽ tạo file PPTX bằng thư viện `python-pptx` (custom fork).

**Cấu trúc code sinh ra:**
```python
# Khởi tạo
presentation = create_poster(width_inch=48.0, height_inch=36.0)
slide = add_blank_slide(presentation)

# Panel boxes (khung nền)
panel_Introduction = add_textbox(slide, "panel_Introduction", x=0.5, y=4.0, width=15.0, height=12.0, text="")
style_shape_border(panel_Introduction, color=(47, 85, 151), thickness=0, line_style="solid")

# Text boxes
tb_Introduction_title = add_textbox(slide, "tb_Introduction_title", x=0.6, y=4.1, ...)
fill_textframe(tb_Introduction_title, json.load(open('/path/to/Introduction_t0_content.json')))

tb_Introduction_t1 = add_textbox(slide, "tb_Introduction_t1", x=0.6, y=5.0, ...)
fill_textframe(tb_Introduction_t1, json.load(open('/path/to/Introduction_t1_content.json')))

# Hình ảnh
fig_Introduction = add_image(slide, "fig_Introduction", x=0.6, y=10.0, width=7.0, height=5.0,
                              image_path="path/to/picture-1.png")

# Lưu
save_presentation(presentation, file_name="/tmp/.../poster.pptx")
```

**Tọa độ:** tất cả là inches (float), python-pptx chuyển sang EMU nội bộ.

**Thư viện python-pptx:** dùng custom fork từ GitHub (`Force1ess/python-pptx`). **Không dùng phiên bản PyPI** — chúng không tương thích.

---

### Stage 9 — Chạy code Python (exec)

**File:** `utils/wei_utils.py`, hàm `run_code()`

**Tiến trình:** 0.95 → "Running poster code…". **0 LLM call.**

```python
exec(poster_code_string, {"__name__": "__main__"})
```

Code string từ Stage 8 được `exec()` trong process hiện tại. Toàn bộ python-pptx API chạy ở đây: tạo Presentation, thêm slide, điền text, chèn hình, lưu file.

Nếu `exec()` lỗi → emit `{"error": "run_code failed: ..."}`, thoát subprocess với code 1.

---

### Stage 10 — Sao chép file output

**Tiến trình:** 0.97 → 1.0 → "Done".

```
{tmp_dir}/poster.pptx  →  {OUTPUT_DIR}/{job_id}/{poster_name}.pptx
```

Subprocess emit:
```json
{"progress": 1.0, "step": "Done", "done": true, "pptx_path": "/abs/path/to/poster.pptx"}
```

Parent process nhận được → set `status="done"`, `output_path=pptx_path`. Endpoint `/download` trả `FileResponse`.

---

## Tổng hợp LLM calls

| Bước | Stage | Mô hình | Input | Output | Số lần |
|------|-------|---------|-------|--------|--------|
| Phân tích paper → sections | 1 | `model_t` | Full text paper | JSON: `{meta, sections}` | 1 |
| Lọc hình/bảng | 2 | `model_t` | Section JSON + metadata hình | JSON: filtered images/tables | 1 |
| Gán hình cho section | 3 | `model_t` | Section JSON + captions | JSON: `{section: {image: N}}` | 1 |
| Sinh bullet points | 6 | `model_t` | Nội dung thô section + font size | JSON: `{title, textbox1[, textbox2]}` | N (song song) |
| Critic (kiểm tra tràn) | 6 | `model_v` | 3 ảnh: neg + pos + rendered | `"1"`, `"2"`, hoặc `"3"` | ≤10/textbox/section |
| Actor (rút ngắn/thêm) | 6 | `model_t` | Phản hồi critic | JSON mới | ≤10/critic round |
| Sinh nội dung title | 6 | `model_t` | Chuỗi meta | JSON: `{title, textbox1}` | 1 |

**Mô hình mặc định:**
- `model_t` = `qwen/qwen3.6-flash` (qua OpenRouter)
- `model_v` = `qwen/qwen3-vl-32b-instruct` (qua OpenRouter)

---

## Cấu trúc file output

```
paper2poster_outputs/
└── {job_id}/
    ├── preextracted_pages.json          ← text từ Qdrant
    ├── tmp/                             ← files tạm (PPTX trung gian, JPEG critic)
    └── {poster_name}.pptx              ← file cuối (có thể download)

agents/
├── contents/
│   ├── {model}_{poster_name}_raw_content.json
│   └── {model}_{poster_name}_bullet_point_content_0.json
├── {model}_images_and_tables/
│   ├── {poster_name}_images.json
│   ├── {poster_name}_tables.json
│   ├── {poster_name}_images_filtered.json
│   ├── {poster_name}_tables_filtered.json
│   └── {poster_name}/
│       ├── {poster_name}-picture-1.png
│       ├── {poster_name}-table-1.png
│       └── ...
├── outlines/
│   └── {model}_{poster_name}_outline_layout_v2_0.json
└── tree_splits/
    └── {model}_{poster_name}_tree_split_0.json
```

---

## Sơ đồ dòng dữ liệu

```
PDF (upload)
  └─ Qdrant (text/page đã index)
        │
        ▼
[Stage 1]  raw_content.json       ← 1 LLM call: paper text → sections JSON
        │
[Stage 1b] _images.json           ← 0 LLM: Docling trích PNG từ PDF
           _tables.json
        │
[Stage 2]  _images_filtered.json  ← 1 LLM call: lọc hình không liên quan
           _tables_filtered.json
        │
[Stage 3]  figure_arrangement     ← 1 LLM call: gán hình vào section
           paper_panels
        │
[Stage 4]  tree_split_0.json      ← 0 LLM: sklearn predict vị trí (inches)
        │
[Stage 5]  poster.yaml config     ← 0 LLM: font, màu sắc
        │
[Stage 6]  bullet_point_content   ← N+1 actor + ≤10N critic LLM calls
           (trong bộ nhớ, có lưu JSON)
        │
[Stage 7]  styled bullet_content  ← 0 LLM: áp màu, font, symbol
        │
[Stage 8]  poster_code (string)   ← 0 LLM: sinh python-pptx code
        │
[Stage 9]  tmp/poster.pptx        ← 0 LLM: exec() code
        │
[Stage 10] {job_id}/poster.pptx   ← file cuối
```

---

## Các điểm quan trọng cần lưu ý

1. **Chỉ 1 job poster cùng lúc**: `PosterAgent` có class-level lock. Gửi job thứ 2 khi đang chạy sẽ nhận HTTP 409.
2. **Text bắt buộc từ Qdrant**: nếu paper chưa được index vào Qdrant, pipeline sẽ fail ngay ở Stage 1.
3. **Docling chỉ chạy khi không có cache**: nếu `_images.json` đã tồn tại (từ lần chạy trước), Docling bị bỏ qua → rất nhanh ở lần thứ 2.
4. **python-pptx fork là bắt buộc**: stock PyPI `python-pptx` không tương thích. Đừng thay thế.
5. **Vòng lặp critic là vision multimodal**: `model_v` cần là LLM có khả năng xử lý ảnh. Nếu dùng model text-only sẽ lỗi ở Stage 6.
6. **Kích thước poster**: mặc định 48×36 inch, scale về 900×1200 units nội bộ, `units_per_inch = 25`.
7. **Song song Stage 6**: tối đa 4 thread, mỗi thread tạo CAMEL agent riêng (vì CAMEL ChatAgent không thread-safe).
