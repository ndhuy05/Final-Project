"""
User service: CRUD operations for User records.
All SQL queries that touch the users table live here.
"""
import logging

from fastapi import HTTPException
from sqlalchemy.orm import Session

from app.models.user import User
from app.services.auth_service import hash_password, verify_password, create_access_token

logger = logging.getLogger(__name__)


def register_user(db: Session, email: str, username: str, password: str) -> str:
    """
    Create a new user account. Returns a JWT access token.
    Raises HTTP 409 if email or username is already taken.
    """
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered.")
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=409, detail="Username already taken.")
    user = User.register(
        email=email,
        username=username,
        password_hash=hash_password(password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return create_access_token({"sub": user.id})


def login_user(db: Session, email: str, password: str) -> str:
    """
    Verify credentials and return a JWT access token.
    Raises HTTP 401 if credentials are invalid, HTTP 403 if account is inactive.
    """
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password.")
    if not user.login():
        raise HTTPException(status_code=403, detail="Account is inactive.")
    return create_access_token({"sub": user.id})
