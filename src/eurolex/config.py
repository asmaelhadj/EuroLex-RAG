"""Runtime settings, loaded once from environment variables and an optional `.env` file.

Settings come from the environment (not code constants) so the same image runs locally, in CI
and in Docker Compose, and so secrets never live in the repository. Missing or invalid required
values raise at startup instead of surfacing later as a confusing runtime error.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
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

    # Only the API needs a key, so ingestion and evaluation run without one; the API calls
    # `require_api_key()` at startup so it can never run with an empty or guessable key.
    api_key: SecretStr | None = Field(default=None, min_length=16)

    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None

    ollama_base_url: str = "http://localhost:11434"
    llm_model: str = "qwen2.5:7b-instruct"
    # Kept separate from llm_model so the judge and candidate generator can differ from
    # the answering model (SPEC.md section 3 and Phase 4).
    judge_model: str | None = None
    candidate_model: str | None = None

    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")

    max_question_chars: int = Field(default=500, gt=0)
    random_seed: int = 42

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    # Off by default: raw questions may contain personal data (SPEC.md section 9).
    log_queries: bool = False

    @field_validator("api_key", "qdrant_api_key", mode="before")
    @classmethod
    def _empty_is_unset(cls, value: object) -> object:
        """`.env.example` ships `API_KEY=`; treat an empty value as not set, not as invalid."""
        return None if value == "" else value

    def require_api_key(self) -> SecretStr:
        if self.api_key is None:
            raise ConfigError("API_KEY is not set. Copy .env.example to .env and fill it in.")
        return self.api_key


class _YamlModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())


class ActSpec(_YamlModel):
    act_id: str = Field(pattern=r"^[a-z_]+$")
    celex: str = Field(pattern=r"^\d{5}[A-Z]\d{4}$")
    raw_filename: str
    # Display name used in embedding headers, e.g. "GDPR Article 6 - Lawfulness of processing".
    short_name: str
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


class EmbeddingConfig(_YamlModel):
    model: str
    # Pinned Hugging Face commit so a re-run months later embeds with the same weights.
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    max_length: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    # device used to build the index
    device: Literal["auto", "cpu", "cuda"] = "auto"
    # fp16 only pays off on GPUs with tensor cores; on a GTX 1650 it is 4x slower
    precision: Literal["fp32", "fp16"] = "fp32"

    @property
    def model_id(self) -> str:
        return f"{self.model}@{self.revision[:12]}+{self.precision}"


class CollectionConfig(_YamlModel):
    name: str = Field(pattern=r"^[a-z0-9_-]+$")
    chunker: Literal["article", "fixed"]
    # Prefix chunk text with "GDPR Article 6(1) - <title>" before embedding.
    prepend_header: bool


class IndexConfig(_YamlModel):
    collections: list[CollectionConfig] = Field(min_length=1)
    upsert_batch_size: int = Field(gt=0)


class IngestConfig(_YamlModel):
    raw_dir: Path
    processed_dir: Path
    acts: list[ActSpec] = Field(min_length=1)
    download: DownloadConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig

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


class RetrieverSpec(_YamlModel):
    type: Literal["dense", "sparse", "bm25"]
    top_k: int = Field(default=50, gt=0)
    weight: float = Field(default=1.0, gt=0)


class FusionConfig(_YamlModel):
    k: int = Field(default=60, gt=0)


class RerankConfig(_YamlModel):
    enabled: bool = False
    model: str = "BAAI/bge-reranker-v2-m3"
    revision: str = Field(
        default="953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e", pattern=r"^[0-9a-f]{40}$"
    )
    # how many fused candidates go to the reranker
    candidates: int = Field(default=50, gt=0)
    max_length: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=8, gt=0)
    prepend_header: bool = True
    device: Literal["auto", "cpu", "cuda"] = "auto"
    precision: Literal["fp32", "fp16"] = "fp32"


class RewriteConfig(_YamlModel):
    enabled: bool = False
    # None means LLM_MODEL from the environment
    model: str | None = None
    mode: Literal["replace", "expand"] = "expand"
    # Ollama num_gpu; 0 runs the model on CPU so a small GPU stays free for the reranker
    gpu_layers: int | None = Field(default=None, ge=0)


class RetrievalConfig(_YamlModel):
    collection: str
    # which chunks_<chunker>.jsonl BM25 reads
    chunker: Literal["article", "fixed"]
    retrievers: list[RetrieverSpec] = Field(min_length=1)
    fusion: FusionConfig = FusionConfig()
    rerank: RerankConfig = RerankConfig()
    rewrite: RewriteConfig = RewriteConfig()
    # results returned to evaluation
    top_k: int = Field(default=10, gt=0)
    # CPU by default so a small GPU is left for the reranker
    query_device: Literal["auto", "cpu", "cuda"] = "cpu"


class ExperimentConfig(_YamlModel):
    name: str = Field(pattern=r"^[a-z0-9_]+$")
    description: str
    ingest_config: Path = Path("configs/ingest.yaml")
    retrieval: RetrievalConfig


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open(encoding="utf-8") as fh:
        return ExperimentConfig.model_validate(yaml.safe_load(fh))


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
