"""
ChatSession and ChatMessage ORM models.

ChatSession — a named conversation thread scoped to one Notebook.
ChatMessage — a single turn (user or assistant) inside a session.
              ChatMessage is a value object: all logic lives in ChatSession or the agents.

citations_json stores a JSON array:
    [{"id": str, "title": str, "page": int, "excerpt": str, "score": float}, ...]
This is kept as a TEXT column (not a JSON column) for maximum SQLite compatibility.
"""
import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, String, Text, Uuid
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class ChatSession(Base, TimestampMixin):
    __tablename__ = "chat_sessions"

    # --- Primary key ---
    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # --- Parent ---
    notebook_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("notebooks.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- Display ---
    title: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # --- Soft delete ---
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # --- Relationships ---
    notebook: Mapped["Notebook"] = relationship("Notebook", back_populates="chat_session")  # type: ignore[name-defined]  # noqa: F821
    messages: Mapped[list["ChatMessage"]] = relationship(
        "ChatMessage", back_populates="session",
        cascade="all, delete-orphan",
        order_by="ChatMessage.created_at",
    )

    # --- Methods ---

    def send_message(self, role: str, content: str, citations_json: str | None = None, user_id: str | None = None) -> "ChatMessage":
        """
        Append a new message to this session and return it.
        The caller is responsible for flushing/committing the session to the DB.
        """
        msg = ChatMessage(
            chat_session_id=self.id,
            role=role,
            content=content,
            citations_json=citations_json,
            user_id=user_id,
        )
        self.messages.append(msg)
        return msg

    def get_history(self) -> list["ChatMessage"]:
        """Return all messages ordered by creation time (oldest first)."""
        return sorted(self.messages, key=lambda m: m.created_at)

    def delete(self) -> None:
        """Soft-delete; messages cascade via FK."""
        from datetime import timezone
        self.deleted_at = datetime.now(timezone.utc)

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id!r} title={self.title!r}>"


class ChatMessage(Base):
    """
    A single turn in a conversation.  Pure data — no methods.
    created_at only (no updated_at — messages are immutable once written).
    """
    __tablename__ = "chat_messages"

    # --- Primary key ---
    id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # --- Parent ---
    chat_session_id: Mapped[str] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("chat_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # --- Optional user FK (null for assistant messages) ---
    user_id: Mapped[str | None] = mapped_column(
        Uuid(as_uuid=False), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # --- Content ---
    role: Mapped[str] = mapped_column(String(16), nullable=False)           # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text, nullable=False)
    message_type: Mapped[str] = mapped_column(
        String(32), nullable=False, default="text"
    )  # "text" | "retrieval_summary" | "metadata_block"
    citations_json: Mapped[str | None] = mapped_column(Text, nullable=True)

    # --- Timestamp (immutable) ---
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )

    # --- Relationships ---
    session: Mapped["ChatSession"] = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage id={self.id!r} role={self.role!r}>"
