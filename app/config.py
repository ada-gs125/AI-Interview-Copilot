from __future__ import annotations

from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class Settings(BaseSettings):
    app_name: str = "AI Interview Copilot"
    openai_api_key: Optional[str] = Field(default=None, alias="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_MODEL")
    openai_input_cost_per_1m_tokens: float = Field(default=0.0, alias="OPENAI_INPUT_COST_PER_1M_TOKENS")
    openai_output_cost_per_1m_tokens: float = Field(default=0.0, alias="OPENAI_OUTPUT_COST_PER_1M_TOKENS")
    auth_secret_key: str = Field(default="local-dev-secret-change-me", alias="AUTH_SECRET_KEY")
    access_token_expire_minutes: int = Field(default=60 * 24 * 7, alias="ACCESS_TOKEN_EXPIRE_MINUTES")
    session_retention_days: int = Field(default=30, alias="SESSION_RETENTION_DAYS")
    database_url: str = Field(
        default="postgresql://interview_copilot:interview_copilot@localhost:5432/interview_copilot",
        alias="DATABASE_URL",
    )
    cors_origins: list[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
