# VibeProject - NotebookLM-Inspired Research Paper Q&A System

A modern web application for managing, querying, and analyzing research papers. Features a clean NotebookLM-inspired UI with notebook management, paper organization, and AI-powered chat backed by a full RAG pipeline.

## 🎯 Project Status

**Frontend: ✅ Fully Functional** | **Backend: ✅ Functional (RAG Pipeline + Paper to Code Active)**

The frontend is complete with a production-ready UI. The backend is live with a full vision-based RAG pipeline: upload a PDF → pages are extracted by a VLM → chunks are embedded and stored in Qdrant → questions are answered by a VLM reading the relevant page images directly. The **Paper to Code** Lab feature is also fully implemented — a 3-stage LLM pipeline generates a runnable code repository from any uploaded paper, downloadable as a ZIP.

## ✨ Implemented Features

### 📚 Notebook Management System
- **Multiple Notebooks**: Create, rename, delete notebooks (ChatGPT-style sidebar)
- **Isolated Data**: Each notebook has its own sources, chat history, and notes
- **Smart Navigation**: Toggle between "Your Notebooks" and "Sources" views

### 📄 Document Management
- **PDF Upload**: Drag-and-drop upload with real backend processing
- **Paper Actions**: Rename and delete papers via 3-dot menu
- **Upload Feedback**: Shows chunks indexed after successful upload

### 💬 AI Chat (RAG Pipeline)
- **VLM-Powered Q&A**: Questions answered by a vision model reading page images
- **Semantic Search**: fastembed (BAAI/bge-small-en-v1.5) + Qdrant vector search
- **3-Page Context Window**: VLM receives pages N-1, N, N+1 around the best match
- **Markdown Support**: Rich text rendering with `marked.js`
- **Citation Badges**: Clickable [1], [2] badges linked to source pages
- **Message History**: Persistent per-notebook chat history

### 🧪 Lab (Generation Features)
- **Paper to Code** ✅ — 3-stage LLM pipeline (Planning → Analyzing → Coding) generates a runnable code repository from a paper; progress bar during generation; download result as ZIP; cancel support
- **Paper to Poster**, **Paper to Web** — UI complete, generation logic TBD

---

## 🏗️ Architecture

### RAG Pipeline

```
── INGESTION (Upload) ──────────────────────────────────────────────────
PDF
  └─ PyMuPDF → page PNGs saved to disk

  For each page (concurrently):
    VLM (OPENROUTER_VISION_MODEL)
      ← page image
      → plain text  (tables described in prose, not Markdown)

    RecursiveCharacterTextSplitter (chunk_size=500, overlap=75)
      → N chunks

    fastembed (BAAI/bge-small-en-v1.5, 384-dim, local CPU)
      → dense vector per chunk

    Qdrant (local on-disk)
      ← upsert {type, paper_id, page_num, content, page_text, vector}

── RETRIEVAL (Chat) ─────────────────────────────────────────────────────
Question
  └─ fastembed → 384-dim query vector
  └─ Qdrant cosine search → top-5 chunks
  └─ deduplicate by (paper_id, page_num)
  └─ top result at page N
       → load images: page N-1, page N, page N+1 from disk

  VLM (OPENROUTER_ANSWER_MODEL)
    ← 3 page images + question
    → answer with page citations
```

### Key Design Decisions
| Decision | Reason |
|---|---|
| VLM reads images for answering | Avoids lossy text extraction for final answer; model sees original layout |
| Tables described in prose during extraction | Avoids Markdown table embedding issues; description embeds better |
| 3-page window (N-1, N, N+1) | Catches content that spans across a page boundary |
| fastembed local embeddings | No API cost/latency for embeddings; 384-dim fast on CPU |
| Qdrant local on-disk | No Docker needed; resets cleanly on re-upload |
| In-memory metadata store | No PostgreSQL dependency; resets on server restart (acceptable for demo) |

---

## 🛠️ Tech Stack

### Frontend
- **Vue 3** (Composition API with `<script setup>`)
- **Vite** · **Pinia** · **Vue Router** · **Tailwind CSS v3**
- **Lucide Vue Next** · **Marked.js**

### Backend
- **FastAPI** + **Uvicorn** (ASGI)
- **Pydantic / pydantic-settings**
- **PyMuPDF** — PDF → page images
- **fastembed** — local text embeddings (BAAI/bge-small-en-v1.5, 384-dim)
- **Qdrant Client** — local on-disk vector store
- **OpenAI SDK** — OpenRouter-compatible client
- **LangChain Text Splitters** — RecursiveCharacterTextSplitter
- **aiofiles** · **python-multipart**

### AI / Models (via OpenRouter)
| Role | Default Model | Config Key |
|---|---|---|
| Page extraction (VLM) | `google/gemini-flash-1.5` | `OPENROUTER_VISION_MODEL` |
| Answer generation (VLM) | `google/gemini-flash-1.5` | `OPENROUTER_ANSWER_MODEL` |
| Paper to Code generation | `anthropic/claude-3.5-sonnet` | `OPENROUTER_CODE_MODEL` |

---

## 📁 Project Structure

```
VibeProject/
├── frontend/
│   └── src/
│       ├── views/Home.vue          # Main UI (3-column layout)
│       ├── stores/app.js           # Pinia store + API calls
│       └── router/index.js
├── backend/
│   ├── app/
│   │   ├── main.py                 # FastAPI app, CORS, logging
│   │   ├── config.py               # Settings (env vars + defaults)
│   │   ├── routers/
│   │   │   ├── papers.py           # Upload, list, delete, /chunks debug
│   │   │   ├── chat.py             # RAG chat endpoint
│   │   │   └── generate.py         # Paper to Code: start/status/cancel/download
│   │   └── services/
│   │       ├── openrouter_service.py     # VLM extraction + answer generation
│   │       ├── paper2code_service.py     # 3-stage Paper2Code pipeline
│   │       ├── embedding_service.py      # fastembed local embeddings
│   │       ├── qdrant_service.py         # Qdrant local client + search
│   │       ├── memory_store.py           # In-memory notebook/paper metadata
│   │       └── pdf_service.py            # PDF → PIL page images
│   ├── requirements.txt
│   ├── .env                        # API keys (gitignored)
│   └── .env.example                # Template for .env
├── paper2code_outputs/             # Generated repos + ZIPs (outside backend/ to avoid reload)
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
- **Node.js 20+**
- **Python 3.11+**
- **OpenRouter API key** — get one at https://openrouter.ai/keys

### Backend Setup

```bash
cd backend

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and set OPENROUTER_API_KEY=your_key_here

# Start server
uvicorn app.main:app --reload
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
# App at http://localhost:5173
```

### Environment Variables

```env
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_VISION_MODEL=google/gemini-flash-1.5    # for page extraction
OPENROUTER_ANSWER_MODEL=google/gemini-flash-1.5    # for answer generation
OPENROUTER_CODE_MODEL=anthropic/claude-3.5-sonnet  # for Paper to Code generation
```

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Health check |
| `GET` | `/api/v1/notebooks/{id}/papers` | List papers in notebook |
| `POST` | `/api/v1/notebooks/{id}/papers/upload` | Upload PDF (triggers ingestion) |
| `DELETE` | `/api/v1/notebooks/{id}/papers/{pid}` | Delete paper + Qdrant points |
| `POST` | `/api/v1/notebooks/{id}/chat` | Ask a question (RAG) |
| `GET` | `/api/v1/notebooks/{id}/chunks` | Debug: browse indexed chunks |
| `POST` | `/api/v1/notebooks/{id}/papers/{pid}/generate/code` | Start Paper to Code job → returns `job_id` |
| `GET` | `/api/v1/generate/code/{job_id}/status` | Poll job progress (`running`/`done`/`error`/`cancelled`) |
| `POST` | `/api/v1/generate/code/{job_id}/cancel` | Cancel a running job |
| `GET` | `/api/v1/generate/code/{job_id}/download` | Download generated repo as ZIP |

---

## 🔧 Development Notes

### Known Limitations
- **In-memory metadata**: Notebooks and paper metadata reset on server restart (no database)
- **No auth**: Notebook IDs passed directly in URL
- **GPU embeddings**: fastembed-gpu installed but falls back to CPU if no CUDA device detected

### Debug Endpoint
Browse stored chunks at:
```
GET /api/v1/notebooks/{id}/chunks?type=text&limit=20
```

### Logging
Per-request debug logs show which pages are sent to the answer VLM:
```
DEBUG app.services.openrouter_service: images sent: ['page_2.png', 'page_3.png', 'page_4.png']
```

---

## 📝 Conventions

- **Naming**: camelCase for JS/Vue state, snake_case for Python
- **Icons**: Lucide Vue Next throughout
- **Event handling**: `@click.stop` to prevent bubbling on 3-dot menus
- **Async**: All OpenRouter calls are `async/await`; page processing uses `asyncio.gather`

---

**Last Updated**: February 2026
**Status**: Frontend complete · Backend RAG pipeline active · Paper to Code active

