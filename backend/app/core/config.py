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
    RAG_VISION_MODEL: str = "qwen/qwen3.6-flash"         # extraction
    RAG_PLANNER_MODEL: str = "qwen/qwen3.6-flash"        # routing + decomposition
    RAG_ANSWER_MODEL: str = "openai/gpt-5.2"             # answer generation
    PAPER2CODE_CODE_MODEL: str = "minimax/minimax-m2.7"         # paper2code generation
    RAG_EMBEDDING_MODEL: str = "qwen/qwen3-embedding-8b"  # 4096-dim

    # Paper2Code output (kept outside backend/ so uvicorn --reload doesn't watch it)
    PAPER2CODE_OUTPUT_DIR: str = "../paper2code_outputs"

    # Paper2Poster pipeline
    PAPER2POSTER_TEXT_MODEL: str = "qwen/qwen3.6-flash"                  # OpenRouter model ID for text tasks
    PAPER2POSTER_VISION_MODEL: str = "qwen/qwen3-vl-32b-instruct"          # OpenRouter model ID for vision tasks
    PAPER2POSTER_DIR: str = "./app/agents"                      # consolidated agents root inside backend/
    PAPER2POSTER_OUTPUT_DIR: str = "../paper2poster_outputs"     # where .pptx files are saved (outside backend/)

    # Paper2Web pipeline
    PAPER2WEB_TEXT_MODEL: str = "qwen/qwen3.6-flash"                     # text/parse/outline/extract model alias
    PAPER2WEB_GENERATOR_MODEL: str = "qwen/qwen3-coder-next"                  # HTML generator model alias
    PAPER2WEB_VISION_MODEL: str = "qwen/qwen3-vl-32b-instruct"             # vision model alias for iterative optimizer
    PAPER2WEB_CODE_MODEL: str = "qwen/qwen3-coder-next"                  # coder model alias for iterative optimizer
    PAPER2WEB_DIR: str = "./app/agents"                         # consolidated agents root inside backend/
    PAPER2WEB_OUTPUT_DIR: str = "../paper2web_outputs"           # where output dirs are saved (outside backend/)

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