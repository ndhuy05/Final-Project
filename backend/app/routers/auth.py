"""
Auth router: user registration, login, and current-user introspection.
"""
import logging

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.user import User
from app.services.auth_service import get_current_user
from app.services.user_service import register_user, login_user

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
    token = register_user(db, email=request.email, username=request.username, password=request.password)
    return TokenResponse(access_token=token)


@router.post("/auth/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """Verify credentials and return a JWT access token."""
    token = login_user(db, email=request.email, password=request.password)
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
