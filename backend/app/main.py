import json
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.database import SessionLocal, engine
from app.models.base import Base
from app.routers import health, papers, chat, generate, poster, web, auth, notebooks

# Third-party loggers stay at INFO; only our app uses DEBUG
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

_MEMORY_STORE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "memory_store.json")
)


def _migrate_json_to_sqlite() -> None:
    """
    One-time migration: import all papers from memory_store.json into SQLite.

    Skipped entirely if:
      - memory_store.json does not exist, or
      - the papers table already contains rows (migration already done).
    """
    if not os.path.exists(_MEMORY_STORE_PATH):
        return

    from app.models.notebook import Notebook
    from app.models.paper import Paper

    db = SessionLocal()
    try:
        if db.query(Paper).count() > 0:
            logger.info("SQLite already contains papers — skipping JSON migration.")
            return

        with open(_MEMORY_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        migrated = 0
        for notebook_id, notebook_data in data.items():
            notebook = db.get(Notebook, notebook_id)
            if notebook is None:
                notebook = Notebook(id=notebook_id, title=notebook_id, user_id=None)
                db.add(notebook)

            for p in notebook_data.get("papers", []):
                filename = p.get("filename", "")
                original_filename = filename.split("_", 1)[-1] if "_" in filename else filename
                meta = p.get("metadata") or {}

                paper = Paper(
                    id=p["id"],
                    notebook_id=notebook_id,
                    title=p.get("title"),
                    original_filename=original_filename,
                    storage_path=filename,
                    page_count=p.get("page_count"),
                    metadata_json=meta if meta else None,
                    year=meta.get("year"),
                    venue=meta.get("venue"),
                    abstract=meta.get("abstract"),
                    description=meta.get("description"),
                    authors=json.dumps(meta["authors"], ensure_ascii=False) if meta.get("authors") else None,
                )
                db.add(paper)
                migrated += 1

        db.commit()
        logger.info("Migrated %d paper(s) from memory_store.json to SQLite.", migrated)

    except Exception:
        logger.exception("JSON→SQLite migration failed — existing data unchanged.")
        db.rollback()
    finally:
        db.close()


@asynccontextmanager
async def lifespan(application: FastAPI):
    # 1. Create all SQLAlchemy tables (no-op if they already exist).
    #    All ORM models are already imported transitively via the routers above.
    Base.metadata.create_all(bind=engine)

    # 2. One-time migration from the legacy memory_store.json (if present).
    _migrate_json_to_sqlite()

    yield


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router,     prefix=settings.API_V1_STR, tags=["auth"])
app.include_router(health.router,   prefix=settings.API_V1_STR, tags=["health"])
app.include_router(notebooks.router, prefix=settings.API_V1_STR, tags=["notebooks"])
app.include_router(papers.router,   prefix=settings.API_V1_STR, tags=["papers"])
app.include_router(chat.router,     prefix=settings.API_V1_STR, tags=["chat"])
app.include_router(generate.router, prefix=settings.API_V1_STR, tags=["generate"])
app.include_router(poster.router,   prefix=settings.API_V1_STR, tags=["poster"])
app.include_router(web.router,      prefix=settings.API_V1_STR, tags=["web"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to VibeProject API",
        "docs": "/docs",
        "version": "0.1.0"
    }
