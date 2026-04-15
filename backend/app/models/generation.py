"""
GenerationJob ORM model.

Tracks a single async background generation task for any of the three
generation agents (CodeAgent, PosterAgent, WebAgent).

status lifecycle:
    queued → running → done
                    ↘ error
                    ↘ cancelled

progress is a float 0.0–1.0 polled by the frontend every 2 s.
output_path is an absolute path to the generated artifact (ZIP / PPTX / bundle).
error_message is populated only when status == "error".
"""
import uuid
from datetime import datetime
from typing import Any

from fastapi.responses import FileResponse
from sqlalchemy import DateTime, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class GenerationJob(Base, TimestampMixin):
    __tablename__ = "generation_jobs"

    # --- Primary key ---
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # --- Parent ---
    paper_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("papers.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- Job identity ---
    job_type: Mapped[str] = mapped_column(String(16), nullable=False)  # "code" | "poster" | "web"

    # --- Status ---
    status: Mapped[str] = mapped_column(
        String(16), nullable=False, default="queued"
    )  # queued | running | done | error | cancelled
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    # --- Output ---
    output_path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Timing ---
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # --- Relationships ---
    paper: Mapped["Paper"] = relationship("Paper", back_populates="generation_jobs")  # type: ignore[name-defined]  # noqa: F821

    # --- Methods ---

    def get_status(self) -> dict[str, Any]:
        """Return the fields polled by the frontend."""
        return {
            "id": self.id,
            "status": self.status,
            "progress": self.progress,
            "error_message": self.error_message,
            "output_path": self.output_path,
        }

    def cancel(self) -> None:
        """
        Mark this job as cancelled.
        The running agent thread checks _is_cancelled() between LLM calls and
        will exit cleanly when it sees this status.
        """
        from datetime import timezone
        self.status = "cancelled"
        self.completed_at = datetime.now(timezone.utc)

    def download(self) -> FileResponse:
        """
        Serve output_path as a file download.
        Raises ValueError if the job is not done or output_path is unset.
        """
        import os
        if self.status != "done":
            raise ValueError(f"Job {self.id} is not done (status={self.status!r})")
        if not self.output_path or not os.path.exists(self.output_path):
            raise ValueError(f"Output file not found: {self.output_path!r}")
        filename = os.path.basename(self.output_path)
        return FileResponse(
            path=self.output_path,
            filename=filename,
            media_type="application/octet-stream",
        )

    def __repr__(self) -> str:
        return f"<GenerationJob id={self.id!r} type={self.job_type!r} status={self.status!r}>"
