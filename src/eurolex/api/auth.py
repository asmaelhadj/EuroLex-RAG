"""X-API-Key check."""

import secrets

from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

# auto_error=False so a missing key gives 401 (not 403) and the docs still show the header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def check_key(given: str | None, expected: str) -> None:
    # compare_digest so the time taken does not leak how much of the key matched
    if given is None or not secrets.compare_digest(given.encode(), expected.encode()):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid API key.",
            headers={"WWW-Authenticate": "APIKey"},
        )


def key_from_header(key: str | None = Security(api_key_header)) -> str | None:
    return key
