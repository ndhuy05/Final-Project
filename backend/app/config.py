from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "VibeProject"
    API_V1_STR: str = "/api/v1"

    # Qdrant — local on-disk mode (no Docker needed)
    QDRANT_LOCAL_PATH: str = "./qdrant_storage"

    # OpenRouter API
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_VISION_MODEL: str = "google/gemini-flash-1.5"   # extraction
    OPENROUTER_PLANNER_MODEL: str = "openai/gpt-4o-mini"        # routing + decomposition
    OPENROUTER_ANSWER_MODEL: str = "google/gemini-flash-1.5"   # answer generation
    OPENROUTER_CODE_MODEL: str = "anthropic/claude-3.5-sonnet"  # paper2code generation
    OPENROUTER_EMBEDDING_MODEL: str = "openai/text-embedding-3-small"  # 1536-dim

    # Paper2Code output (kept outside backend/ so uvicorn --reload doesn't watch it)
    PAPER2CODE_OUTPUT_DIR: str = "../paper2code_outputs"

    # Paper2Poster pipeline
    POSTER_MODEL_T: str = "qwen/qwen3.5-plus-02-15"             # OpenRouter model ID for text tasks
    POSTER_MODEL_V: str = "qwen/qwen3.5-plus-02-15"             # OpenRouter model ID for vision tasks
    PAPER2POSTER_DIR: str = "./app/poster_pipeline"              # self-contained pipeline root inside backend/
    PAPER2POSTER_OUTPUT_DIR: str = "../paper2poster_outputs"     # where .pptx files are saved (outside backend/)

    # Database
    DATABASE_URL: str = "sqlite:///./vibeproject.db"

    # Upload storage
    UPLOAD_DIR: str = "./uploads"

    # Page image storage (for OpenRouter vision input)
    IMAGE_DIR: str = "./uploads/images"

    # Authentication (JWT)
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
