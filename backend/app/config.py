from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    # Project
    PROJECT_NAME: str = "VibeProject"
    API_V1_STR: str = "/api/v1"

    # Database (for future use)
    DATABASE_URL: str = "postgresql://user:password@localhost:5432/vibeproject"

    # Qdrant — local on-disk mode (no Docker needed)
    QDRANT_LOCAL_PATH: str = "./qdrant_storage"

    # ColQwen2 model
    COLPALI_MODEL_NAME: str = "vidore/colqwen2-v1.0"

    # VLM for answer generation (vision-language model)
    LLM_MODEL_NAME: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Upload storage
    UPLOAD_DIR: str = "./uploads"

    # Page image storage (for VLM input)
    IMAGE_DIR: str = "./uploads/images"

    # CORS
    BACKEND_CORS_ORIGINS: List[str] = ["http://localhost:5173", "http://localhost:3000"]

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
