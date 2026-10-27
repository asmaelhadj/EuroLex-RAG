"""Talks to the API for the Streamlit UI. Kept apart from app.py so it can be tested."""

from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass(frozen=True)
class Reply:
    ok: bool
    answer: str = ""
    refused: bool = False
    citations: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


ERRORS = {
    401: "The UI's API key was rejected. Check API_KEY in .env.",
    422: "The question was not accepted",
    429: "Too many questions in a short time. Wait a minute and try again.",
    503: "The language model is not available. Is Ollama running?",
}


def ask(
    question: str,
    *,
    api_url: str,
    api_key: str,
    timeout: float = 300,
    client: httpx.Client | None = None,
) -> Reply:
    http = client or httpx.Client(timeout=timeout)
    try:
        response = http.post(
            f"{api_url.rstrip('/')}/ask",
            json={"question": question, "filters": {"act_id": None}},
            headers={"X-API-Key": api_key},
        )
    except httpx.HTTPError:
        return Reply(ok=False, error=f"Could not reach the API at {api_url}. Is it running?")
    finally:
        if client is None:
            http.close()
    if response.status_code != 200:
        message = ERRORS.get(response.status_code, f"The API returned {response.status_code}.")
        if response.status_code == 422:
            detail = response.json().get("detail")
            message += f": {detail}" if isinstance(detail, str) else "."
        return Reply(ok=False, error=message)
    body = response.json()
    return Reply(
        ok=True, answer=body["answer"], refused=body["refused"], citations=body["citations"]
    )
