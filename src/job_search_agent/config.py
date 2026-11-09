"""Application settings, read from environment variables or a .env file.

This is the single place where configuration enters the app. Values are
converted to proper types and validated at startup, so a bad setting fails
immediately with a clear error instead of breaking mid-run.

Environment variable names match field names, ignoring case. For example,
GEMINI_MODEL sets ``gemini_model``. See .env.example for every setting.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Absolute project root, so default paths work no matter which folder the
# app is started from.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Every tunable setting for SpartanScout, with safe defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Model provider (used from step 10).
    gemini_api_key: SecretStr | None = None
    gemini_model: str = "gemini-3.6-flash"
    model_timeout_seconds: float = Field(default=30.0, gt=0)
    model_max_retries: int = Field(default=2, ge=0, le=5)
    use_mock_model: bool = False

    # Agent limits (used from step 13).
    max_tool_calls: int = Field(default=5, ge=1, le=10)

    # Data.
    data_dir: Path = PROJECT_ROOT / "data" / "demo" / "opportunities"


@lru_cache
def get_settings() -> Settings:
    """Build the settings once and reuse them for the life of the process."""
    return Settings()
