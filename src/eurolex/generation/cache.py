"""Disk cache for LLM calls, so a stopped run can resume and identical calls are not repeated.

The key covers everything that can change the reply: model, system prompt, prompt, JSON
schema and generation options. Replies are only reused for an exact match.
"""

import hashlib
import json
from pathlib import Path

from eurolex.generation.llm import OllamaClient


def cache_key(model: str, system: str | None, prompt: str, schema: object, options: object) -> str:
    payload = json.dumps(
        {"model": model, "system": system, "prompt": prompt, "schema": schema, "options": options},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class CachedOllama:
    """Wraps OllamaClient; same interface, plus hit/miss counts for the run record."""

    def __init__(self, inner: OllamaClient, path: Path) -> None:
        self.model = inner.model
        self.num_ctx = inner.num_ctx
        self._inner = inner
        self._path = path
        self._entries: dict[str, dict[str, object]] = {}
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    entry = json.loads(line)
                    self._entries[entry["key"]] = entry
        self.hits = 0
        self.misses = 0
        self.last_prompt_tokens: int | None = None

    def has_model(self) -> bool:
        return self._inner.has_model()

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, object] | None = None,
        temperature: float | None = None,
    ) -> str:
        options = self._inner.options_for(temperature)
        key = cache_key(self.model, system, prompt, json_schema, options)
        entry = self._entries.get(key)
        if entry is not None:
            self.hits += 1
            tokens = entry.get("prompt_tokens")
            self.last_prompt_tokens = tokens if isinstance(tokens, int) else None
            return str(entry["response"])
        self.misses += 1
        response = self._inner.generate(
            prompt, system=system, json_schema=json_schema, temperature=temperature
        )
        self.last_prompt_tokens = self._inner.last_prompt_tokens
        entry = {"key": key, "response": response, "prompt_tokens": self.last_prompt_tokens}
        self._entries[key] = entry
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(entry) + "\n")
        return response
