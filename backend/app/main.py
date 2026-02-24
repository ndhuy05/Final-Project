import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import health, papers, chat

# Third-party loggers stay at INFO; only our app uses DEBUG
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("app").setLevel(logging.DEBUG)

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json"
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
app.include_router(health.router, prefix=settings.API_V1_STR, tags=["health"])
app.include_router(papers.router, prefix=settings.API_V1_STR, tags=["papers"])
app.include_router(chat.router, prefix=settings.API_V1_STR, tags=["chat"])


@app.get("/")
async def root():
    return {
        "message": "Welcome to VibeProject API",
        "docs": "/docs",
        "version": "0.1.0"
    }
