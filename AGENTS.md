# AGENTS.md — OpenLab

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.
Compact, high-signal guide for AI agents working in this repo.
Only things that are non-obvious or frequently-missed are listed.

---

## Repo layout

```
OpenLab/
├── backend/          # FastAPI + SQLAlchemy + Qdrant (Python 3.11+)
│   ├── app/
│   │   ├── config.py          # all settings (pydantic-settings, reads backend/.env)
│   │   ├── main.py            # FastAPI app, lifespan, router registration
│   │   ├── database.py        # SQLAlchemy engine + SessionLocal
│   │   ├── models/            # ORM models
│   │   ├── schemas/           # Pydantic request/response models
│   │   ├── routers/           # auth, chat, generate, health, notebooks, papers, poster
│   │   ├── services/          # embedding, qdrant, reranker, pdf, auth, memory_store
│   │   └── poster_pipeline/   # self-contained PosterAgent pipeline (see below)
│   ├── .env                   # gitignored — copy from .env.example
│   ├── .env.example
│   └── requirements.txt
└── frontend/         # Vue 3 + Vite + Pinia + TailwindCSS (plain JS, no TypeScript)
    ├── src/
    │   ├── api/               # axios wrappers
    │   ├── components/
    │   ├── router/
    │   ├── stores/            # Pinia stores
    │   └── views/
    └── package.json
```

---

## Starting the services

**Backend** — run from `backend/` directory:
```bash
cd backend
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

pip install -r requirements.txt
playwright install chromium   # required after pip install
uvicorn app.main:app --reload --port 8000
```
- Reads settings from `backend/.env` (relative paths in config.py assume CWD is `backend/`)
- Tables are auto-created at startup via `Base.metadata.create_all()` — **do not run alembic migrations**; there is no `alembic/` directory even though `alembic` is listed in requirements.txt

**Frontend** — run from `frontend/` directory:
```bash
cd frontend
npm install
npm run dev       # Vite dev server on :5173
```

---

## Backend — critical non-obvious details

### No test suite
There are no tests. Do not try to run pytest or jest.

### Database
- SQLite at `backend/vibeproject.db` (auto-created; gitignored)
- Qdrant runs **locally on-disk** at `backend/qdrant_storage/` — no Docker, no server to start
- Legacy `memory_store.json` (if present) is auto-migrated to SQLite on first startup; do nothing

### python-pptx — custom git fork required
`requirements.txt` pins:
```
python-pptx @ git+https://github.com/Force1ess/python-pptx@dc356685...
```
The stock PyPI `python-pptx` is **incompatible** with PosterAgent. Never replace it with `pip install python-pptx`.

### bcrypt version constraint
`requirements.txt` requires `bcrypt<4.0.0`. passlib 1.7.4 breaks with bcrypt 4+. Do not upgrade.

### fastembed — ONNX model download on first use
`fastembed-gpu` downloads BAAI/bge-small-en-v1.5 (embeddings) and BAAI/bge-reranker-base (reranker) as `.onnx` files to `fastembed_cache/` on first import. This is expected; the directory is gitignored.

### docling — heavy download on first use
`docling` downloads ~2 GB of IBM layout/OCR models on first use. This is expected and takes several minutes.

### Playwright — post-install step
After `pip install -r requirements.txt`, run:
```bash
playwright install chromium
```
`playwright` is imported at the module level in `wei_utils.py` and will fail at import time if the browser is not installed.

### All LLM calls go through OpenRouter
Set `OPENROUTER_API_KEY` in `backend/.env`. There is no local LLM inference (except fastembed for embeddings/reranking). The API base URL used is `https://openrouter.ai/api/v1`.

### API prefix
All backend endpoints are under `/api/v1` (configured in `config.py` via `API_V1_STR`).

### CORS
Allowed origins are `http://localhost:5173` and `http://localhost:3000` by default. Override via `BACKEND_CORS_ORIGINS` in `.env`.

### Auth
JWT-based. Token valid 7 days by default. Secret read from `SECRET_KEY` in `.env`.

---

## Poster pipeline — runtime directories

Everything under `backend/app/poster_pipeline/` that is **not** Python source is runtime output and is gitignored:

| Path pattern | Contents |
|---|---|
| `*_images_and_tables/` | Extracted images/tables per model run (PNG, HTML, MD, JSON) |
| `contents/` | Intermediate parsed content JSON files |
| `tree_splits/` | Intermediate section-tree JSON files |
| `checkpoints/` | Checkpoint files |
| `outlines/` | Outline files |
| `log/` | Pipeline logs |
| `tmp/` | Temporary files |

These directories exist in the working tree but are **never committed**.

Final `.pptx` outputs go to `../paper2poster_outputs/` (relative to `backend/`), i.e. `OpenLab/paper2poster_outputs/`. Also gitignored.

Paper2Code outputs go to `../paper2code_outputs/` (i.e. `OpenLab/paper2code_outputs/`). Also gitignored.

---

## Frontend — non-obvious details

- **Plain JavaScript** — no TypeScript anywhere. Do not add `.ts` files.
- **Vue 3** with both Options API (older views) and Composition API (`<script setup>`) — either style is acceptable.
- **Pinia** for state management. Stores are in `frontend/src/stores/`.
- **TailwindCSS v3** — utility classes only, no custom CSS framework.
- `npm` is the package manager. There is a `package-lock.json`. Do not use `pnpm` or `yarn`.
- Backend API is proxied through Vite in dev; check `frontend/vite.config.js` for proxy rules.

---

## What is gitignored (non-obvious items)

| Pattern | Reason |
|---|---|
| `backend/app/poster_pipeline/*_images_and_tables/` | Runtime per-model image extraction outputs |
| `backend/app/poster_pipeline/contents/` | Runtime intermediate JSON |
| `backend/app/poster_pipeline/tree_splits/` | Runtime intermediate JSON |
| `backend/app/poster_pipeline/checkpoints/`, `log/`, `tmp/`, `outlines/` | Runtime pipeline state |
| `paper2code_outputs/`, `paper2poster_outputs/` | Pipeline final outputs at repo root |
| `fastembed_cache/`, `.fastembed_cache/`, `*.onnx` | ONNX model cache |
| `backend/vibeproject.db`, `*.db`, `*.sqlite3` | Runtime databases |
| `backend/qdrant_storage/` | Qdrant on-disk index |
| `backend/uploads/` | User-uploaded PDFs |
| `TODO.md`, `DESIGN.md` | Local planning docs, not for the repo |

If `git status` shows a large number of files in `backend/app/poster_pipeline/` as untracked, the `.gitignore` rule for `*_images_and_tables/` covers them — do not add them.
