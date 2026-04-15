# AGENTS.md — VibeProject Coding Guidelines

## Project Overview

Full-stack research assistant: **Vue 3 + Pinia frontend** (JavaScript, no TypeScript) and a
**FastAPI Python backend** serving an agentic RAG pipeline with Paper2Code generation.
The two apps live in `frontend/` and `backend/` with no shared monorepo tooling.

---

## Prerequisites

- **Node.js 20+** (frontend)
- **Python 3.11+** (backend)
- **OpenRouter API key** — set in `backend/.env` as `OPENROUTER_API_KEY`

---

## Commands

### Frontend (`frontend/`)

```bash
npm install          # install dependencies
npm run dev          # Vite dev server → http://localhost:5173
npm run build        # production build → dist/
npm run preview      # serve production build locally
```

> No lint or test runner is configured. If you add tests use Vitest:
> ```bash
> npx vitest run src/stores/app.test.js          # single test file
> npx vitest run src/stores/app.test.js -t "name" # single test by name
> ```

### Backend (`backend/`)

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload   # dev server → http://localhost:8000
```

> No test runner is configured. If you add tests use pytest:
> ```bash
> pytest backend/tests/test_foo.py                    # single file
> pytest backend/tests/test_foo.py::test_bar          # single test
> pytest backend/tests/test_foo.py::TestClass::method # single method
> ```

### Environment

Both apps require a `.env` file. Copy the examples before running:
```bash
cp backend/.env.example backend/.env   # then fill in API keys
# frontend/.env already present; edit VITE_API_BASE_URL if needed
```

---

## Key Files

```
frontend/src/stores/app.js              # all Pinia state + every API call
frontend/src/views/Home.vue             # entire application UI
frontend/src/api/client.js              # Axios singleton — all HTTP goes here
backend/app/main.py                     # FastAPI entry point, CORS, router registration
backend/app/config.py                   # pydantic-settings (all env vars)
backend/app/routers/papers.py           # upload / list / delete papers
backend/app/routers/chat.py             # POST /chat — agentic RAG endpoint
backend/app/routers/generate.py         # Paper2Code start / status / cancel / download
backend/app/services/openrouter_service.py  # VLM extraction, planner, answer generation
backend/app/services/embedding_service.py   # async OpenRouter embeddings
backend/app/services/qdrant_service.py      # local Qdrant client + vector search
backend/app/services/reranker_service.py    # cross-encoder reranking (BAAI/bge-reranker-base)
backend/app/services/memory_store.py        # JSON-file paper metadata persistence
backend/app/services/paper2code_service.py  # 3-stage Paper2Code pipeline (threading)
backend/app/services/pdf_service.py         # PDF → PIL page images (PyMuPDF)
```

---

## Architecture Notes

- **Single-store, single-view** frontend: all state in `frontend/src/stores/app.js`; all UI in
  `frontend/src/views/Home.vue`. Do not reach into the Pinia store from outside store files.
- **Service-module** backend: each `backend/app/services/*.py` is a plain module (not a class)
  with lazy-initialized module-level singletons. Keep this pattern when adding new services.
- **No database/ORM**: persistence is `memory_store.json` (JSON file). `models/` and `schemas/`
  directories exist as empty scaffolding — do not add SQLAlchemy/Alembic unless explicitly asked.
- **Background threads for Paper2Code**: blocking OpenAI SDK calls run in `threading.Thread` to
  avoid blocking the async FastAPI event loop. Mirror this pattern for any new blocking pipeline.
- **Agentic RAG flow**: planner LLM → parallel `asyncio.gather` of `read_metadata` / `retrieve`
  actions → cross-encoder reranking → VLM reads page images (N-1, N, N+1) → answer.
- **Duplicate upload** returns HTTP 409; re-uploading the same filename is intentionally blocked.

---

## Environment Variables

All model names and secrets live in `backend/.env` (never hard-code them):

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_VISION_MODEL=google/gemini-flash-1.5    # page extraction + answer VLM
OPENROUTER_ANSWER_MODEL=google/gemini-flash-1.5    # answer generation VLM
OPENROUTER_PLANNER_MODEL=openai/gpt-4o-mini        # planner + metadata answers
OPENROUTER_EMBEDDING_MODEL=openai/text-embedding-3-small  # 4096-dim vectors
OPENROUTER_CODE_MODEL=anthropic/claude-3.5-sonnet  # Paper to Code generation
```

> **Gotcha**: if `OPENROUTER_EMBEDDING_MODEL` is changed, delete `qdrant_storage/` and
> re-upload all papers — vector dimensions must match what is stored in Qdrant.

---

## Frontend Code Style (JavaScript / Vue 3)

### General

- Plain JavaScript — **no TypeScript**. Do not introduce `*.ts` files or `tsconfig.json`.
- No ESLint/Prettier configured. Keep style consistent with existing code (2-space indent, single
  quotes, no semicolons at end of file-level statements).

### Imports

```js
// 1. Vue reactivity APIs
import { ref, computed, watch, nextTick } from 'vue'
// 2. Store / router
import { useAppStore } from '@/stores/app'
// 3. Third-party libraries
import axios from 'axios'
// 4. Icon components (grouped in one import)
import { Folder, Plus, Trash2 } from 'lucide-vue-next'
```

- Use `import.meta.env.VITE_*` for environment variables; never hard-code base URLs.
- All HTTP calls must go through `frontend/src/api/client.js` (the Axios singleton).

### Vue Patterns

- **Always** use `<script setup>` (Composition API). Never write Options API components.
- Use `ref()` for all reactive primitives; `computed()` for derived values.
- Conditional classes: use array syntax `:class="[base, cond ? 'a' : 'b']"`.
- Use event modifiers in templates (`@click.stop`, `@keydown.enter.exact.prevent`).
- Call `nextTick()` before any DOM-dependent operation after a reactive change.
- `v-if` / `v-else-if` / `v-else` chains for multi-state UI (idle / loading / done / error).

### Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Variables, refs, functions | `camelCase` | `uploadState`, `handleFileUpload` |
| Component files | `PascalCase.vue` | `Home.vue`, `UploadPanel.vue` |
| Pinia store actions | verb-first camelCase | `createNotebook`, `toggleMenu` |
| Store ID | lowercase string | `defineStore('app', ...)` |
| Tailwind color tokens | `notebook-{50..900}` | `bg-notebook-800` |

### State Management

- The Pinia store is the **only** place that calls the API. Components call store actions.
- Module-level `let _pollInterval = null` pattern for polling; always clear on terminal states.

### Error Handling

```js
try {
  await store.uploadPaper(file)
} catch (err) {
  errorMessage.value = err?.response?.data?.detail || 'Upload failed'
} finally {
  isLoading.value = false
}
```

- Extract error messages with optional chaining: `err?.response?.data?.detail || 'fallback'`.
- Use `.catch(() => {})` only for intentional fire-and-forget calls (e.g., cancel a job).
- Surface errors via local `ref` state; there is no global error boundary.

### Styling

- Tailwind CSS utility classes exclusively. Use `@apply` inside `<style scoped>` only for
  multi-rule prose/markdown overrides (see `Home.vue`).
- The custom `notebook-*` palette (defined in `tailwind.config.js`) is the design token system.
  Prefer `notebook-*` over Tailwind's default `gray-*`.

---

## Backend Code Style (Python)

### General

- Target **Python 3.11+**. Use `T | None` union syntax, not `Optional[T]`, for new code.
- 4-space indentation, PEP 8 naming.
- Every service module must start with a docstring explaining its responsibility.

### Imports

```python
# 1. Standard library
import json
import logging
from typing import Any

# 2. Third-party
from fastapi import HTTPException
from pydantic import BaseModel

# 3. Local app modules
from app.config import settings
from app.services import memory_store
```

### Naming Conventions

| Thing | Convention | Example |
|---|---|---|
| Variables, functions, modules | `snake_case` | `upload_paper`, `memory_store` |
| Classes (Pydantic models, etc.) | `PascalCase` | `ChatRequest`, `PaperMetadata` |
| Constants / env config keys | `UPPER_SNAKE_CASE` | `OPENROUTER_API_KEY` |
| Private helpers | leading underscore | `_get_client`, `_extract_json` |
| Router function names | verb-noun | `upload_paper`, `delete_paper` |

### Typing

- Annotate all function signatures: `def foo(x: str, y: int = 5) -> list[dict[str, Any]]:`.
- Use `dict[str, Any]` and `list[dict]` for service layer collection types.
- Define all FastAPI request/response bodies as `pydantic.BaseModel` subclasses inline in the
  router file (not in the empty `schemas/` directory).

### Service Module Pattern

```python
logger = logging.getLogger(__name__)

_client: AsyncOpenAI | None = None   # lazy singleton

def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=settings.OPENROUTER_API_KEY, ...)
    return _client
```

- Use `# --- Section Name ---` dividers to separate logical sections in long files.
- Module-level `logger = logging.getLogger(__name__)` in every service file.

### Error Handling

```python
# FastAPI endpoints
raise HTTPException(status_code=404, detail="Paper not found")

# Service layer — log and re-raise (or return fallback)
try:
    result = await _get_client().embeddings.create(...)
except Exception as e:
    logger.exception("Embedding failed: %s", e)
    raise
```

- Reranker and other non-critical services: `logger.warning(...)` and continue without the
  feature rather than crashing (graceful degradation).
- JSON parsing: implement a 3-tier fallback (direct parse → strip markdown fences → regex).
- Background thread errors: catch broadly, set `job["status"] = "error"`, log with
  `logger.exception`.

### Async Patterns

- All OpenRouter / OpenAI API calls must be `async def` + `await`.
- Parallel calls: `await asyncio.gather(*[_run_one(a) for a in actions], return_exceptions=True)`.
- Blocking SDK calls in background work: use `threading.Thread`, not `asyncio.run_in_executor`.
- Cancellation: check a `_is_cancelled(job_id)` flag between LLM calls inside threads.

### Configuration

- All secrets and tunables come from `backend/app/config.py` (`pydantic-settings BaseSettings`).
- Never hard-code API keys, model names, or URLs. Add new settings to `config.py` and
  `.env.example` together.
