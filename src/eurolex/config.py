"""Runtime settings, loaded once from environment variables and an optional `.env` file.

Settings come from the environment (not code constants) so the same image runs locally, in CI
and in Docker Compose, and so secrets never live in the repository. Missing or invalid required
values raise at startup instead of surfacing later as a confusing runtime error.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
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


class _YamlModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ActSpec(_YamlModel):
    act_id: str = Field(pattern=r"^[a-z_]+$")
    celex: str = Field(pattern=r"^\d{5}[A-Z]\d{4}$")
    raw_filename: str
    expected_articles: int = Field(gt=0)
    expected_recitals: int = Field(gt=0)
    # Disabled acts stay in the config so extending the corpus is a one-line change.
    enabled: bool = True


class DownloadConfig(_YamlModel):
    cellar_url_template: str
    source_url_template: str
    timeout_seconds: float = Field(gt=0)

    def source_url(self, celex: str) -> str:
        return self.source_url_template.format(celex=celex)


class ArticleChunkerConfig(_YamlModel):
    max_tokens: int = Field(gt=0)


class FixedChunkerConfig(_YamlModel):
    size_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> "FixedChunkerConfig":
        if self.overlap_tokens >= self.size_tokens:
            raise ValueError("overlap_tokens must be smaller than size_tokens")
        return self


class ChunkingConfig(_YamlModel):
    article: ArticleChunkerConfig
    fixed: FixedChunkerConfig


class IngestConfig(_YamlModel):
    raw_dir: Path
    processed_dir: Path
    acts: list[ActSpec] = Field(min_length=1)
    download: DownloadConfig
    chunking: ChunkingConfig

    @model_validator(mode="after")
    def _at_least_one_enabled(self) -> "IngestConfig":
        if not self.enabled_acts:
            raise ValueError("at least one act must be enabled")
        return self

    @property
    def enabled_acts(self) -> list[ActSpec]:
        return [act for act in self.acts if act.enabled]


def load_ingest_config(path: Path) -> IngestConfig:
    """Load and validate `configs/ingest.yaml` (or another file with the same shape)."""
    with path.open(encoding="utf-8") as fh:
        return IngestConfig.model_validate(yaml.safe_load(fh))


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
