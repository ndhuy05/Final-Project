"""
Chat service: session and message persistence for the notebook chat feature.
All SQL operations that touch chat_sessions and chat_messages live here.
"""
import logging
from datetime import datetime, timedelta, timezone

from sqlalchemy.orm import Session

from app.models.notebook import Notebook
from app.models.chat import ChatSession

logger = logging.getLogger(__name__)


def get_or_create_chat_session(db: Session, notebook: Notebook) -> ChatSession:
    """Return the notebook's single ChatSession, creating it if it doesn't exist yet."""
    session = notebook.get_chat_session()
    if session is None:
        session = ChatSession(notebook_id=notebook.id)
        db.add(session)
        db.flush()  # assign session.id without committing the transaction
    return session


def persist_chat_turn(
    db: Session,
    notebook: Notebook,
    user_content: str,
    assistant_content: str,
    citations: list,
    user_id: str | None = None,
) -> None:
    """
    Persist one full chat turn (user + assistant messages) to the notebook's session.
    Commits the transaction. Rolls back and logs on any error.
    """
    try:
        chat_session = get_or_create_chat_session(db, notebook)
        # Stamp both messages from a single snapshot so they never share a timestamp.
        # Tied created_at values cause non-deterministic ORDER BY in PostgreSQL.
        now = datetime.now(timezone.utc)
        user_msg = chat_session.send_message(role="user", content=user_content, user_id=user_id)
        user_msg.created_at = now
        citations_data = [c.model_dump() for c in citations] if citations else None
        asst_msg = chat_session.send_message(role="assistant", content=assistant_content, citations_json=citations_data)
        asst_msg.created_at = now + timedelta(milliseconds=1)
        db.commit()
    except Exception:
        logger.exception("Failed to persist chat messages for notebook %s", notebook.id)
        db.rollback()
