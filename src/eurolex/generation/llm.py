"""Small Ollama client. Phase 5 adds what answer generation needs on top."""

import logging
import time
from typing import Protocol

import httpx

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    pass


class LLMClient(Protocol):
    model: str

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, object] | None = None,
        temperature: float | None = None,
    ) -> str: ...


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_seconds: float = 60.0,
        retries: int = 2,
        temperature: float = 0.0,
        seed: int = 42,
        client: httpx.Client | None = None,
    ) -> None:
        self.model = model
        self._base = base_url.rstrip("/")
        self._url = self._base + "/api/generate"
        self._retries = retries
        self._options = {"temperature": temperature, "seed": seed}
        self._http = client or httpx.Client(timeout=timeout_seconds)

    def has_model(self) -> bool:
        """True if Ollama is reachable and the model is pulled."""
        try:
            response = self._http.get(self._base + "/api/tags")
            response.raise_for_status()
            names = {m.get("name") for m in response.json().get("models", [])}
        except (httpx.HTTPError, ValueError):
            return False
        # "mistral" and "mistral:latest" are the same model
        return self.model in names or f"{self.model}:latest" in names

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, object] | None = None,
        temperature: float | None = None,
    ) -> str:
        options = dict(self._options)
        if temperature is not None:
            options["temperature"] = temperature
        body: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": options,
        }
        if system:
            body["system"] = system
        if json_schema:
            # Ollama structured output: the reply is JSON matching this schema
            body["format"] = json_schema
        last_error: Exception | None = None
        for attempt in range(self._retries + 1):
            try:
                response = self._http.post(self._url, json=body)
                response.raise_for_status()
                return str(response.json()["response"])
            except (httpx.HTTPError, KeyError, ValueError) as exc:
                last_error = exc
                logger.warning("ollama call failed (attempt %d): %s", attempt + 1, exc)
                if attempt < self._retries:
                    time.sleep(0.5 * 2**attempt)
        raise LLMError(f"ollama call to {self.model} failed") from last_error
