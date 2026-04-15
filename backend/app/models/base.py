"""
SQLAlchemy 2.0 declarative base and shared mixins.

All ORM models inherit from Base (for table mapping) and TimestampMixin
(for automatic created_at / updated_at columns).
"""
from datetime import datetime

from sqlalchemy import DateTime, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Single declarative base for all ORM models."""
    pass


class TimestampMixin:
    """
    Adds created_at and updated_at columns to any model that inherits it.

    created_at — set once at INSERT time, never changed.
    updated_at — set at INSERT and automatically updated on every UPDATE.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
