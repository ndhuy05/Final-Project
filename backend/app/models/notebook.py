"""
Notebook ORM model.

A named workspace that groups Papers and a single ChatSession together.
Belongs to one User; soft-deleted via deleted_at.
paper_count_cached is a denormalised counter updated by the Papers router
to avoid COUNT(*) on every sidebar render.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Notebook(Base, TimestampMixin):
    __tablename__ = "notebooks"

    # --- Primary key ---
    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # --- Ownership (nullable until auth is introduced) ---
    user_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )

    # --- Display fields ---
    title: Mapped[str] = mapped_column(String(255), nullable=False)

    # --- Denormalised counter (maintained by DB trigger in PostgreSQL; by router in SQLite dev) ---
    paper_count_cached: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # --- Soft delete ---
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # --- Relationships ---
    user: Mapped["User | None"] = relationship("User", back_populates="notebooks")  # type: ignore[name-defined]  # noqa: F821
    papers: Mapped[list["Paper"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Paper", back_populates="notebook", cascade="all, delete-orphan"
    )
    chat_session: Mapped["ChatSession | None"] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "ChatSession", back_populates="notebook", cascade="all, delete-orphan",
        uselist=False,
    )

    # --- Methods ---

    def rename(self, new_title: str) -> None:
        """Update the display title."""
        self.title = new_title

    def delete(self) -> None:
        """Soft-delete the notebook. Papers and sessions cascade via DB."""
        from datetime import timezone
        self.deleted_at = datetime.now(timezone.utc)

    def get_papers(self) -> list["Paper"]:  # type: ignore[name-defined]  # noqa: F821
        """Return active (non-deleted) papers ordered by creation time."""
        return [p for p in self.papers if p.deleted_at is None]

    def get_chat_session(self) -> "ChatSession | None":  # type: ignore[name-defined]  # noqa: F821
        """Return the chat session if it exists and has not been soft-deleted."""
        if self.chat_session and self.chat_session.deleted_at is None:
            return self.chat_session
        return None

    def __repr__(self) -> str:
        return f"<Notebook id={self.id!r} title={self.title!r}>"
