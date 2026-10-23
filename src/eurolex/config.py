"""Runtime settings, loaded once from environment variables and an optional `.env` file.

Settings come from the environment (not code constants) so the same image runs locally, in CI
and in Docker Compose, and so secrets never live in the repository. Missing or invalid required
values raise at startup instead of surfacing later as a confusing runtime error.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Raised when settings are missing or invalid, with a hint pointing at `.env.example`."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    # Required: the API must never start with an empty or trivially guessable key.
    api_key: SecretStr = Field(min_length=16)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    # Kept separate from llm_model so the judge and candidate generator can differ from
    # the answering model (SPEC.md section 3 and Phase 4).
    judge_model: str | None = None
    candidate_model: str | None = None

    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"

    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    max_question_chars: int = Field(default=500, gt=0)
    random_seed: int = 42

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Off by default: raw questions may contain personal data (SPEC.md section 9).
    log_queries: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings, failing fast with a readable message."""
    try:
        return Settings()
    except ValidationError as exc:
        fields = ", ".join(".".join(str(p) for p in err["loc"]).upper() for err in exc.errors())
        raise ConfigError(
            f"Invalid or missing settings: {fields}. Copy .env.example to .env and fill it in."
        ) from exc
