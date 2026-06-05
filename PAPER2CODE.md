# Paper2Code — Giải thích chi tiết pipeline

## Tổng quan

Paper2Code là pipeline tự động sinh code Python từ một bài báo khoa học. Từ PDF của paper, pipeline đọc toàn bộ nội dung, lên kế hoạch kiến trúc phần mềm, phân tích từng file, rồi sinh code hoàn chỉnh cho từng file — tất cả bằng LLM thông qua một cuộc hội thoại multi-turn có cấu trúc. Kết quả là một thư mục code Python có thể tải về dưới dạng ZIP.

---

## Kiến trúc tổng thể

```
HTTP POST /api/v1/notebooks/{notebook_id}/papers/{paper_id}/generate/code
    │
    └── CodeAgent (daemon thread)
            │
            ├── Pre-stage: Lấy text từ Qdrant (tất cả các trang)
            │
            ├── Stage 1 — plan()         [4 LLM call, multi-turn]
            │     Call 1: Lên kế hoạch tổng thể
            │     Call 2: Thiết kế kiến trúc → danh sách file
            │     Call 3: Phân tích task → task_list + logic_analysis
            │     Call 4: Sinh config.yaml
            │
            ├── Stage 2 — analyze()      [N LLM call, 1 call/file]
            │     Phân tích logic từng file (chưa có code)
            │
            ├── Stage 3 — generate_code() [N LLM call, 1 call/file]
            │     Sinh code thực sự từng file, theo thứ tự phụ thuộc
            │
            ├── Stage 4 — package_zip()
            │     Đóng gói thành {paper_id}_repo.zip
            │
            └── FileResponse → {paper_id}_repo.zip
```

---

## Entry Point — HTTP API

**File:** `backend/app/routers/generate.py`

Có 4 endpoint:

| Method | Path | Mục đích |
|--------|------|----------|
| `POST` | `/notebooks/{notebook_id}/papers/{paper_id}/generate/code` | Bắt đầu pipeline, trả `job_id` ngay |
| `GET` | `/generate/code/{job_id}/status` | Kiểm tra tiến trình (`status`, `progress` 0–1, `step`) |
| `POST` | `/generate/code/{job_id}/cancel` | Yêu cầu huỷ |
| `GET` | `/generate/code/{job_id}/download` | Tải ZIP khi `status == "done"` |

**Luồng khi nhận POST (`start_code_generation`):**
1. Tra cứu paper qua `memory_store.get_paper(notebook_id, paper_id)` → 404 nếu không tìm thấy.
2. Gọi `_code_agent.run(notebook_id, paper_id, paper_title, page_count)` → trả `job_id` ngay.
3. Pipeline chạy trong daemon thread; client poll `/status`.

**Singleton:** `_code_agent = CodeAgent()` là module-level singleton — tất cả request dùng chung 1 instance, job state cách ly theo `job_id`.

---

## `CodeAgent` class

**File:** `backend/app/agents/code_agent.py` (519 dòng)

Kế thừa từ `GenerationAgent` (abstract base class dùng chung với `PosterAgent` và `WebAgent`).

**`GenerationAgent` cung cấp:**
| Method | Vai trò |
|--------|---------|
| `_new_job()` | Tạo UUID job, insert vào `_jobs` dict, trả job_id |
| `_update(job_id, **kwargs)` | Cập nhật trạng thái job (progress, step, status…) |
| `_is_cancelled(job_id)` | Trả `True` nếu job bị huỷ — check trước mỗi LLM call |
| `get_job(job_id)` | Đọc trạng thái job — gọi bởi status endpoint |
| `cancel(job_id)` | Set `cancelled=True`, `status="cancelled"` |

**`_jobs` là class-level dict** — `CodeAgent._jobs`, `PosterAgent._jobs`, `WebAgent._jobs` hoàn toàn độc lập.

**LLM client (`_get_client()`):**
```python
openai.OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=settings.OPENROUTER_API_KEY
)
```
Dùng OpenAI SDK nhưng trỏ vào OpenRouter. Model mặc định: `PAPER2CODE_CODE_MODEL = "minimax/minimax-m2.7"`.

---

## Các stage chi tiết

### Pre-stage — Lấy text từ Qdrant

**Hàm:** `CodeAgent._fetch_paper_text(notebook_id, paper_id, page_count)`

**Tiến trình:** 0.0 → "Fetching paper content…"

**Xử lý:**
- Với mỗi trang 1..`page_count`, gọi `qdrant_service.get_page_text(notebook_id, paper_id, page_num)`.
- `get_page_text` làm Qdrant scroll query filter theo `paper_id`, `page_num`, `type == "text"`, lấy field `page_text` từ payload.
- Ghép tất cả trang thành:
  ```
  === Page 1 ===
  {text trang 1}

  === Page 2 ===
  {text trang 2}
  ...
  ```

**Output:** `paper_content: str` — toàn bộ văn bản paper, phân tách theo trang.

---

### Stage 1 — Plan (Lập kế hoạch)

**Hàm:** `CodeAgent.plan(paper_content, output_dir, job_id, step_counter)`

**4 LLM call multi-turn** — mỗi call thêm cả user message và assistant reply vào `trajectories` list (cuộc hội thoại tích lũy).

#### Call 1 — Kế hoạch tổng thể

**Tiến trình:** "Planning: Overall plan"

- **System prompt:** Vai trò expert researcher/planner; nhấn mạnh reproducibility.
- **User prompt:** Toàn bộ `paper_content` + yêu cầu phác thảo:
  - Phương pháp luận
  - Các experiment
  - Dataset sử dụng
  - Hyperparameter
  - Evaluation metrics
  - **Lưu ý: chưa viết code — chỉ là roadmap**
- **Output:** Text markdown mô tả kế hoạch tổng thể (`responses[0]`).

#### Call 2 — Thiết kế kiến trúc phần mềm

**Tiến trình:** "Planning: Architecture design"

- **Tiếp tục hội thoại** từ Call 1 (append vào `trajectories`).
- **User prompt:** Yêu cầu JSON kiến trúc phần mềm:
  ```
  [CONTENT]
  {
    "Implementation approach": "...",
    "File list": ["main.py", "model.py", "trainer.py", "utils.py"],
    "Data structures and interfaces": "mermaid classDiagram...",
    "Program call flow": "mermaid sequenceDiagram...",
    "Anything UNCLEAR": "..."
  }
  [/CONTENT]
  ```
- **Output:** JSON trong tags `[CONTENT]...[/CONTENT]` với danh sách file và sơ đồ kiến trúc (`responses[1]`).

#### Call 3 — Phân tích task và logic

**Tiến trình:** "Planning: Logic design"

- **Tiếp tục hội thoại**.
- **User prompt:** Yêu cầu JSON chi tiết hơn:
  ```
  [CONTENT]
  {
    "Required packages": ["numpy==1.24.0", "torch==2.0.0", "scikit-learn"],
    "Logic Analysis": [
      ["model.py", "Định nghĩa kiến trúc mạng neural, class Model với forward()"],
      ["trainer.py", "Vòng lặp training, validation, checkpoint"],
      ["main.py", "Entry point, parse args, khởi tạo và chạy trainer"]
    ],
    "Task list": ["model.py", "trainer.py", "main.py"],
    "Shared Knowledge": "Config được load từ config.yaml bằng yaml.safe_load()",
    "Anything UNCLEAR": "..."
  }
  [/CONTENT]
  ```
- **Output:** `responses[2]` — **đây là output quan trọng nhất** của Stage 1:
  - `task_list`: danh sách file cần tạo (theo thứ tự phụ thuộc).
  - `logic_analysis_dict`: map `filename → mô tả logic`.

#### Call 4 — Sinh config.yaml

**Tiến trình:** "Planning: Config generation"

- **Tiếp tục hội thoại**.
- **User prompt:** Trích xuất hyperparameter từ paper (learning rate, batch size, optimizer, epochs, …) và sinh `config.yaml`. **Quy tắc: KHÔNG bịa đặt — chỉ dùng giá trị có trong paper.**
- **Output:** YAML block với training config (`responses[3]`).

**Sau Stage 1:**
- Parse `responses[2]` bằng `_content_to_json()`: tìm JSON trong `[CONTENT]...[/CONTENT]`.
- Trích `todo_file_lst` từ `"Task list"` (danh sách file, bỏ `config.yaml`).
- Trích `logic_analysis_dict` từ `"Logic Analysis"`.
- Parse YAML từ `responses[3]` bằng `_extract_yaml()`.
- Lưu:
  - `{output_dir}/planning_trajectories.json` — toàn bộ 4-turn conversation.
  - `{output_dir}/planning_config.yaml` — config đã parse.

---

### Stage 2 — Analyze (Phân tích từng file)

**Hàm:** `CodeAgent.analyze(paper_content, context_lst, config_yaml, task_list, logic_analysis_dict, output_dir, job_id, step_counter)`

**Tiến trình:** "Analyzing: {filename}" cho mỗi file.

**N LLM call** — 1 call/file, **tuần tự**, bỏ qua `config.yaml`.

Mỗi call:
- **System prompt:** Expert researcher/analyzer/engineer; tuân theo design đã có; chỉ dùng giá trị từ config.yaml.
- **User prompt:**
  - `paper_content` (toàn bộ text paper)
  - `context_lst[0]` = kế hoạch tổng thể (responses[0])
  - `context_lst[1]` = kiến trúc (responses[1])
  - `context_lst[2]` = task design (responses[2])
  - `config_yaml` (nội dung planning_config.yaml)
  - Yêu cầu cụ thể: `"Write the logic analysis in '{filename}', which is intended for '{description}'"`
  - Kết thúc bằng: `"## Logic Analysis: {filename}"`
- **Output:** Văn xuôi phân tích logic cho file đó — mô tả class, function, interface cần có. **Chưa có code.**

**Lưu:** mỗi file → `{output_dir}/{filename_sanitized}_simple_analysis_response.json` (wrap response theo định dạng OpenAI envelope).

**Ví dụ output của analyze:**
```
model.py:
  Class ResNet với forward() nhận tensor (B, C, H, W), trả (B, num_classes).
  Dùng BatchNorm sau mỗi conv layer.
  __init__ nhận config dict: hidden_dim, num_layers, dropout_rate từ config.yaml.
  Không import từ trainer.py để tránh circular import.
```

---

### Stage 3 — Generate Code (Sinh code)

**Hàm:** `CodeAgent.generate_code(paper_content, context_lst, config_yaml, task_list, analyses, output_dir, output_repo_dir, job_id, step_counter)`

**Tiến trình:** "Coding: {filename}" cho mỗi file.

**N LLM call** — 1 call/file, **tuần tự theo thứ tự phụ thuộc**, bỏ qua `config.yaml`.

Duy trì running context của code đã sinh:
- `done_files: list[str]` — bắt đầu là `["config.yaml"]`.
- `done_code: dict[str, str]` — map `filename → code text`.

**Mỗi file:**
- **System prompt:** Expert researcher/engineer; Google-style guidelines; viết code với triple quotes; phải khớp với phương pháp trong paper.
- **User prompt:**
  - `paper_content`
  - 3 context từ Stage 1
  - `config_yaml`
  - **`code_context`**: code thực tế của tất cả file đã sinh (tích lũy) → đây là cơ chế tránh circular import và đảm bảo nhất quán interface.
  - 8 quy tắc code quality:
    1. Code phải đầy đủ, không placeholder, không TODO.
    2. Type hints đầy đủ.
    3. Không circular import.
    4. Dùng giá trị từ `config.yaml`, không hardcode.
    5. Tuân theo interface đã thiết kế.
    6. Error handling phù hợp.
    7. Docstring cho class và function public.
    8. Tuân theo paper's methodology chính xác.
  - Logic analysis từ Stage 2 cho file này.
  - Kết thúc bằng: `"## Code: {filename}"`.
- **Output:** Markdown code block Python. Regex extract: `r'^```(?:\w+)?\s*\n(.*?)(?=^```)```'` (DOTALL+MULTILINE) lấy block đầu tiên. Fallback = raw reply.

**Ghi file:**
- Code → `{output_repo_dir}/{filename}` (tạo thư mục con nếu cần via `os.makedirs`).
- Sau khi xong tất cả: copy `planning_config.yaml` → `{output_repo_dir}/config.yaml`.

**Ví dụ luồng tích lũy `code_context`:**
```
# Sau file model.py:
done_files = ["config.yaml", "model.py"]
code_context = "## config.yaml\n...\n## model.py\n```python\n# model.py\nclass ResNet:\n    ...\n```"

# Khi sinh trainer.py, LLM thấy code thực sự của model.py
# → biết interface ResNet.forward(), constructor args, etc.
# → sinh trainer.py nhất quán với model.py
```

---

### Stage 4 — Package ZIP

**Hàm:** `CodeAgent.package_zip(output_repo_dir, zip_base)`

**Tiến trình:** 0.98 → "Creating ZIP…" → 1.0 → "Done"

```python
shutil.make_archive(zip_base, "zip", output_repo_dir)
```

Tạo `{paper_id}_repo.zip` tại `{PAPER2CODE_OUTPUT_DIR}/{paper_id}_repo.zip`.

Khi xong: `self._update(job_id, status="done", progress=1.0, output_path=zip_path)`.

---

## Tổng hợp LLM calls

| Call | Stage | Prompt input chính | Output format | Số lần |
|------|-------|-------------------|---------------|--------|
| Overall plan | 1, Call 1 | Paper text | Markdown roadmap | 1 |
| Architecture design | 1, Call 2 | + roadmap | JSON `[CONTENT]` file list + diagrams | 1 |
| Task/logic design | 1, Call 3 | + architecture | JSON `[CONTENT]` task_list + logic_analysis | 1 |
| Config generation | 1, Call 4 | + task design | YAML block | 1 |
| Logic analysis | 2 | Paper + plans + config + desc | Văn xuôi phân tích | N (1/file) |
| Code generation | 3 | Paper + plans + config + prior_code + analysis | Python code block | N (1/file) |

**Tổng:** `4 + 2N` LLM call (N = số file, thường 3–8).

**Mô hình:** `PAPER2CODE_CODE_MODEL` (mặc định `minimax/minimax-m2.7`) qua OpenRouter.

**Cơ chế huỷ:** `_is_cancelled()` được check trước mỗi LLM call trong `_call()`. Nếu bị huỷ → raise `InterruptedError` → `_run_pipeline` catch, thoát cleanly.

---

## Cấu trúc file output

```
paper2code_outputs/
├── {paper_id}/                                   ← output_dir (intermediate)
│   ├── planning_trajectories.json                ← 4-turn Stage 1 conversation
│   ├── planning_config.yaml                      ← extracted config từ Call 4
│   ├── model_py_simple_analysis_response.json    ← Stage 2 analysis (per file)
│   ├── trainer_py_simple_analysis_response.json
│   └── main_py_simple_analysis_response.json
│
├── {paper_id}_repo/                              ← output_repo_dir (code thực sự)
│   ├── model.py
│   ├── trainer.py
│   ├── main.py
│   ├── utils.py
│   └── config.yaml
│
└── {paper_id}_repo.zip                           ← file cuối (có thể download)
```

---

## Sơ đồ dòng dữ liệu

```
Qdrant (page texts)
    │
    ▼
paper_content: str
  (page 1...N nối theo thứ tự)
    │
    ▼
[Stage 1] plan()  — 4 LLM call, multi-turn conversation
  Call 1: paper_content → overall_plan (text)
  Call 2: + overall_plan → architecture_json (file list, diagrams)
  Call 3: + architecture → task_list[], logic_analysis_dict{}
  Call 4: + all above → config_yaml (YAML text)
  Lưu: planning_trajectories.json, planning_config.yaml
    │
    ├── task_list = ["model.py", "trainer.py", "main.py"]
    └── logic_analysis_dict = {"model.py": "...", ...}
    │
    ▼
[Stage 2] analyze()  — N LLM call (1/file)
  Per file: paper + plans + config + description → analysis prose
  Lưu: {file}_simple_analysis_response.json × N
    │
    └── analyses = {"model.py": "ResNet class với...", ...}
    │
    ▼
[Stage 3] generate_code()  — N LLM call (1/file, theo thứ tự phụ thuộc)
  Per file: paper + plans + config + prior_code + analysis → python code
  Lưu: {output_repo_dir}/{filename} (real files)
  Lưu: {output_repo_dir}/config.yaml (copy từ planning_config.yaml)
    │
    ▼
[Stage 4] package_zip()
  shutil.make_archive → {paper_id}_repo.zip
    │
    ▼
HTTP GET /download → FileResponse({paper_id}_repo.zip)
```

---

## Cấu hình

**`backend/app/core/config.py` + `backend/.env`:**

| Setting | Default | Mô tả |
|---------|---------|-------|
| `PAPER2CODE_CODE_MODEL` | `minimax/minimax-m2.7` | Model LLM dùng cho tất cả call |
| `PAPER2CODE_OUTPUT_DIR` | `../paper2code_outputs` | Thư mục output (relative to `backend/`) |
| `OPENROUTER_API_KEY` | (bắt buộc set trong .env) | API key OpenRouter |

**Không có file prompt template riêng** — tất cả prompt được hardcode dạng f-string trong `code_agent.py`.

---

## So sánh với Paper2Poster

| Khía cạnh | Paper2Code | Paper2Poster |
|-----------|-----------|-------------|
| Output | Python code (ZIP) | Poster PPTX |
| Subprocess | Không — chạy trong thread | Có — spawn subprocess `run_poster_job.py` |
| Vision model | Không | Có (critic vòng lặp kiểm tra overflow) |
| Số LLM call | `4 + 2N` (thấp, ~10–20) | `3 + N + ≤20N` (cao, ~50–100+) |
| Song song | Không (tuần tự) | Có (Stage 6 dùng 4 thread) |
| ML truyền thống | Không | Có (sklearn tree-split layout) |
| File trung gian | Minimal (trajectories + analyses) | Nhiều (images, outlines, tree_splits…) |
| Concurrency | Nhiều job cùng lúc OK | Chỉ 1 job cùng lúc (lock toàn cục) |
| Cấu hình ngoài | Không | Có (`poster.yaml` — font, màu, symbol) |

---

## Các điểm quan trọng cần lưu ý

1. **Multi-turn conversation trong Stage 1**: 4 call không độc lập mà tích lũy trong `trajectories`. LLM "nhớ" kết quả call trước để câu sau có context đầy đủ.

2. **`code_context` là cơ chế chính để tránh inconsistency**: mỗi file mới được sinh ra với toàn bộ code của các file trước đó làm context. LLM biết đúng class, method signature, import path cần dùng.

3. **Không có feedback loop / critic**: khác với Poster pipeline, Code pipeline không có bước kiểm tra/sửa lại code. Code sinh ra một lần là xong.

4. **Thứ tự `task_list` quan trọng**: LLM ở Stage 1 Call 3 tự quyết định thứ tự file theo phụ thuộc (base class trước, utils trước main). Stage 3 follow đúng thứ tự này.

5. **`config.yaml` không được gửi lên LLM để sinh**: nó được copy trực tiếp từ `planning_config.yaml`. Các file code khác import giá trị từ đây bằng `yaml.safe_load()`.

6. **Không có test, không có validation**: code sinh ra không được chạy hay kiểm tra. Chất lượng phụ thuộc hoàn toàn vào LLM và quality của paper.

7. **Text phải có trong Qdrant**: nếu paper chưa được index, `page_count = 0`, `paper_content` sẽ rỗng, LLM sẽ không có input để làm việc.
