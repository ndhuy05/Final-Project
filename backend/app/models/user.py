"""
User ORM model.

Represents an authenticated person who owns one or more Notebooks.
Passwords are stored as bcrypt hashes — plain text is never persisted.
Soft-delete via deleted_at; suspended accounts use is_active = False.
"""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, String
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    # --- Primary key ---
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4())
    )

    # --- Identity ---
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # --- Status ---
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # --- Relationships ---
    notebooks: Mapped[list["Notebook"]] = relationship(  # type: ignore[name-defined]  # noqa: F821
        "Notebook", back_populates="user", cascade="all, delete-orphan"
    )

    # --- Methods ---

    @classmethod
    def register(cls, email: str, username: str, password_hash: str, full_name: str | None = None) -> "User":
        """Create a new User record. Caller is responsible for hashing the password."""
        return cls(
            email=email,
            username=username,
            password_hash=password_hash,
            full_name=full_name,
        )

    def login(self) -> bool:
        """Return True if the account is active and not soft-deleted."""
        return self.is_active and self.deleted_at is None

    def logout(self) -> None:
        """Placeholder — JWT invalidation is handled at the auth layer, not here."""
        pass

    def delete(self) -> None:
        """Soft-delete: set deleted_at to now, deactivate account."""
        from datetime import timezone
        self.deleted_at = datetime.now(timezone.utc)
        self.is_active = False

    def __repr__(self) -> str:
        return f"<User id={self.id!r} email={self.email!r}>"
