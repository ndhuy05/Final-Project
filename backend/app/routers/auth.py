"""
Auth router: user registration, login, and current-user introspection.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.user import User
from app.services.auth_service import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

logger = logging.getLogger(__name__)
router = APIRouter()


# --- Request / Response schemas ---

class RegisterRequest(BaseModel):
    username: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    full_name: str | None


# --- Endpoints ---

@router.post("/auth/register", response_model=TokenResponse, status_code=201)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Create a new user account and return a JWT access token."""
    if db.query(User).filter(User.email == request.email).first():
        raise HTTPException(status_code=409, detail="Email already registered.")
    if db.query(User).filter(User.username == request.username).first():
        raise HTTPException(status_code=409, detail="Username already taken.")
    user = User.register(
        email=request.email,
        username=request.username,
        password_hash=hash_password(request.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.id})
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Verify credentials and return a JWT access token."""
    user = db.query(User).filter(User.email == request.email).first()
    if not user or not verify_password(request.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
        )
    if not user.login():
        raise HTTPException(status_code=403, detail="Account is inactive.")
    token = create_access_token({"sub": user.id})
    return TokenResponse(access_token=token)


@router.get("/auth/me", response_model=UserResponse)
async def me(current_user: User = Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return UserResponse(
        id=current_user.id,
        username=current_user.username,
        email=current_user.email,
        full_name=current_user.full_name,
    )
