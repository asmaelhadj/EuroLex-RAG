"""JSON logs for the API.

Questions are logged as a short hash and a length, never as text, unless LOG_QUERIES=true
(local debugging only). The hash lets you spot repeated questions without storing them.
It is not anonymisation: a short, common question can be guessed from its hash.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from typing import Any

# attributes every LogRecord has; anything else was passed through `extra`
_STANDARD = set(logging.makeLogRecord({}).__dict__) | {"message", "asctime"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        entry: dict[str, Any] = {
            "time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        entry.update({k: v for k, v in record.__dict__.items() if k not in _STANDARD})
        if record.exc_info:
            entry["error"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def configure_logging(level: str) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # HTTP clients log every request at INFO (model downloads, Qdrant, Ollama); too noisy
    for name in ("httpx", "httpx2", "httpcore"):
        logging.getLogger(name).setLevel(max(logging.WARNING, root.level))


def question_fields(question: str, log_queries: bool) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "question_sha256": hashlib.sha256(question.encode("utf-8")).hexdigest()[:16],
        "question_chars": len(question),
    }
    if log_queries:
        fields["question"] = question
    return fields
