from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from app.config import get_settings
from app.database import authenticate_user, create_user
from app.dependencies import current_user
from app.schemas import AuthTokenResponse, UserCreateRequest, UserLoginRequest, UserResponse
from app.services.auth_service import create_access_token


router = APIRouter(prefix="/auth", tags=["auth"])


def _token_response(user: UserResponse) -> AuthTokenResponse:
    settings = get_settings()
    return AuthTokenResponse(
        access_token=create_access_token(
            user_id=user.id,
            email=user.email,
            secret_key=settings.auth_secret_key,
            expires_minutes=settings.access_token_expire_minutes,
        ),
        user=user,
    )


@router.post("/register", response_model=AuthTokenResponse, status_code=status.HTTP_201_CREATED)
def register(request: UserCreateRequest) -> AuthTokenResponse:
    try:
        user = create_user(email=request.email, password=request.password)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return _token_response(user)


@router.post("/login", response_model=AuthTokenResponse)
def login(request: UserLoginRequest) -> AuthTokenResponse:
    user = authenticate_user(email=request.email, password=request.password)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return _token_response(user)


@router.get("/me", response_model=UserResponse)
def me(user: UserResponse = Depends(current_user)) -> UserResponse:
    return user
