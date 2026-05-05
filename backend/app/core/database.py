"""
SQLAlchemy engine and session factory.

Usage in routers (dependency injection pattern):
    from app.core.database import get_db
    from sqlalchemy.orm import Session

    @router.get("/example")
    def example(db: Session = Depends(get_db)):
        ...

Usage in synchronous (non-FastAPI) code:
    from app.core.database import SessionLocal
    db = SessionLocal()
    try:
        ...
    finally:
        db.close()
"""
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings

# connect_args is required for SQLite to allow cross-thread access.
# Ignored by other dialects (Postgres, MySQL) — safe to leave in.
_connect_args = {"check_same_thread": False} if settings.DATABASE_URL.startswith("sqlite") else {}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=_connect_args,
    echo=False,       # set True locally to log every SQL statement
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a DB session per request and
    guarantees the session is closed on exit (even on exceptions).
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
