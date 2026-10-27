"""Optional Langfuse tracing, off by default (LANGFUSE_ENABLED=true to turn on).

Install with `uv sync --extra tracing`. Langfuse reads LANGFUSE_PUBLIC_KEY,
LANGFUSE_SECRET_KEY and LANGFUSE_HOST from the environment. Traces carry the same fields
as the log line; the question text is only sent when LOG_QUERIES=true.
"""

from typing import Any, Protocol

from eurolex.config import ConfigError, Settings


class Tracer(Protocol):
    def trace(self, question: str | None, ids: dict[str, Any], fields: dict[str, Any]) -> None: ...


class NoTracer:
    def trace(self, question: str | None, ids: dict[str, Any], fields: dict[str, Any]) -> None:
        return None


class LangfuseTracer:
    def __init__(self, client: Any) -> None:
        self._client = client

    def trace(self, question: str | None, ids: dict[str, Any], fields: dict[str, Any]) -> None:
        span = self._client.start_observation(
            name="ask",
            input=question,
            output={"refused": fields["refused"], "citations": fields["citations"]},
            metadata={**ids, **fields},
        )
        span.end()


def build_tracer(settings: Settings) -> Tracer:
    if not settings.langfuse_enabled:
        return NoTracer()
    try:
        from langfuse import Langfuse
    except ImportError as exc:
        raise ConfigError(
            "LANGFUSE_ENABLED=true but langfuse is not installed. Run `uv sync --extra tracing`."
        ) from exc
    return LangfuseTracer(Langfuse())
