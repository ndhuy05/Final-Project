"""
Notebook ORM model.

A named workspace that groups Papers and ChatSessions together.
Belongs to one User; soft-deleted via deleted_at.
paper_count_cached is a denormalised counter updated by the Papers router
to avoid COUNT(*) on every sidebar render.
"""
import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Notebook(Base, TimestampMixin):
    __tablename__ = "notebooks"

    # --- Primary key ---
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # --- Ownership (nullable until auth is introduced) ---
    user_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True
    )

    # --- Display fields ---
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    color_tag: Mapped[str | None] = mapped_column(String(32), nullable=True)

    # --- Denormalised counter (updated by router, not by trigger) ---
    paper_count_cached: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    # --- Soft delete ---
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # --- Relationships ---
    user: Mapped["User | None"] = relationship("User", back_populates="notebooks")  # type: ignore[name-defined]  # noqa: F821
    papers: Mapped[list["Paper"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Paper", back_populates="notebook", cascade="all, delete-orphan"
    )
    chat_sessions: Mapped[list["ChatSession"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "ChatSession", back_populates="notebook", cascade="all, delete-orphan"
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

    def get_chat_sessions(self) -> list["ChatSession"]:  # type: ignore[name-defined]  # noqa: F821
        """Return active chat sessions ordered by creation time (newest first)."""
        active = [s for s in self.chat_sessions if s.deleted_at is None]
        return sorted(active, key=lambda s: s.created_at, reverse=True)

    def __repr__(self) -> str:
        return f"<Notebook id={self.id!r} title={self.title!r}>"
