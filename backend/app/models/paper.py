"""
Paper ORM model.

Represents an uploaded PDF together with all extracted metadata and
indexing state. One Paper belongs to one Notebook.

Duplicate-upload guard: a UniqueConstraint on (notebook_id, original_filename)
mirrors the HTTP 409 logic currently enforced in the papers router.

storage_path stores the root directory for all page images:
    uploads/{paper_id}/page_0.png, page_1.png, ...
This convention is relied on by ExtractionAgent and AnsweringAgent.
"""
import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class Paper(Base, TimestampMixin):
    __tablename__ = "papers"
    __table_args__ = (
        UniqueConstraint("notebook_id", "original_filename", name="uq_paper_notebook_filename"),
    )

    # --- Primary key ---
    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # --- Parent ---
    notebook_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- File info ---
    original_filename: Mapped[str] = mapped_column(String(512), nullable=False)
    storage_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    file_size_mb: Mapped[float | None] = mapped_column(Float, nullable=True)
    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # --- Extracted metadata (populated by ExtractionAgent) ---
    title: Mapped[str | None] = mapped_column(String(512), nullable=True)
    authors: Mapped[str | None] = mapped_column(String(1024), nullable=True)
    year: Mapped[str | None] = mapped_column(String(8), nullable=True)
    venue: Mapped[str | None] = mapped_column(String(512), nullable=True)
    abstract: Mapped[str | None] = mapped_column(Text, nullable=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Full metadata dict as returned by ExtractionAgent (includes keywords etc.) ---
    metadata_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    # --- Soft delete ---
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # --- Relationships ---
    notebook: Mapped["Notebook"] = relationship("Notebook", back_populates="papers")  # type: ignore[name-defined]  # noqa: F821

    # --- Methods ---

    def get_metadata(self) -> dict[str, Any]:
        """Return the core bibliographic fields used by PlannerAgent."""
        return {
            "title": self.title,
            "authors": self.authors,
            "year": self.year,
            "venue": self.venue,
            "abstract": self.abstract,
            "description": self.description,
        }

    def rename(self, new_title: str) -> None:
        """Update the display title without touching original_filename."""
        self.title = new_title

    def delete(self) -> None:
        """
        Soft-delete the paper record.
        Callers are responsible for removing Qdrant points and page images from disk.
        """
        from datetime import timezone
        self.deleted_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"<Paper id={self.id!r} title={self.title!r}>"
