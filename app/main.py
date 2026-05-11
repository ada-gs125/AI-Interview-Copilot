from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import close_db, init_db
from app.logging_config import configure_logging
from app.routes.auth import router as auth_router
from app.routes.interview import router as interview_router

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    logger.info("startup", extra={"event": "startup"})
    yield
    close_db()
    logger.info("shutdown", extra={"event": "shutdown"})


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description="AI-powered interview preparation from a resume PDF and target job description.",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(interview_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    latency_ms = int((time.perf_counter() - start) * 1000)

    user_id: str | None = None
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        try:
            from app.services.auth_service import decode_access_token
            payload = decode_access_token(auth[7:], settings.auth_secret_key)
            user_id = payload.get("sub")
        except Exception:
            pass

    logger.info(
        "request",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "latency_ms": latency_ms,
            "user_id": user_id,
        },
    )
    return response


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
