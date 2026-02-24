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
    OPENROUTER_ANSWER_MODEL: str = "google/gemini-flash-1.5"   # answer generation

    # Upload storage
    UPLOAD_DIR: str = "./uploads"

    # Page image storage (for OpenRouter vision input)
    IMAGE_DIR: str = "./uploads/images"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
