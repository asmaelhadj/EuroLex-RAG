from collections.abc import Iterator
from pathlib import Path

import pytest

from eurolex.config import ConfigError, Settings, get_settings


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[None]:
    """Run each test without the developer's real `.env` or environment leaking in."""
    monkeypatch.chdir(tmp_path)
    for name in ("API_KEY", "LLM_MODEL", "LOG_QUERIES", "MAX_QUESTION_CHARS"):
        monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_missing_api_key_fails_fast() -> None:
    with pytest.raises(ConfigError, match="API_KEY"):
        get_settings()


def test_short_api_key_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "short")
    with pytest.raises(ConfigError, match="API_KEY"):
        get_settings()


def test_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY", "x" * 32)
    monkeypatch.setenv("LLM_MODEL", "llama3.1:8b")
    settings = get_settings()
    assert settings.llm_model == "llama3.1:8b"
    assert settings.log_queries is False
    assert settings.max_question_chars == 500
    assert "x" * 32 not in repr(settings)  # secret is masked


def test_reads_dotenv_file() -> None:
    Path(".env").write_text("API_KEY=" + "y" * 20 + "\n", encoding="utf-8")
    assert Settings().api_key.get_secret_value() == "y" * 20
