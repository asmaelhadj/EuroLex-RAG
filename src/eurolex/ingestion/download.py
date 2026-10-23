"""Fetch the official English text of each act into `data/raw/`.

EUR-Lex's HTML endpoint sits behind an AWS WAF bot challenge, so automated requests to it get an
empty `202` response. We do not try to get past that. Instead we use the Publications Office
CELLAR API, the documented machine-access route to the same documents (ADR 002). If that fails
too, the human can save the EUR-Lex HTML page by hand under the configured file name and the
parser reads it from there.
"""

import hashlib
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx

from eurolex.config import ActSpec, DownloadConfig

logger = logging.getLogger(__name__)

# Marker present in every CONVEX-generated act page; guards against saving an error page.
_EXPECTED_MARKER = b'class="eli-main-title"'


class DownloadError(RuntimeError):
    pass


def manual_instructions(act: ActSpec, raw_dir: Path, download: DownloadConfig) -> str:
    return (
        f"Could not download {act.act_id} ({act.celex}). Open "
        f"{download.source_url(act.celex)} in a browser, save the page as HTML, and place it at "
        f"{raw_dir / act.raw_filename}. Then re-run the command."
    )


def fetch_act(act: ActSpec, download: DownloadConfig, client: httpx.Client) -> bytes:
    url = download.cellar_url_template.format(celex=act.celex)
    response = client.get(
        url,
        headers={"Accept": "application/xhtml+xml", "Accept-Language": "eng"},
        follow_redirects=True,
        timeout=download.timeout_seconds,
    )
    if response.headers.get("x-amzn-waf-action"):
        raise DownloadError(f"{url} answered with a bot challenge; not retrying")
    if response.status_code != httpx.codes.OK:
        raise DownloadError(f"{url} returned HTTP {response.status_code}")
    if _EXPECTED_MARKER not in response.content:
        raise DownloadError(f"{url} did not return an act document")
    return response.content


def download_acts(
    acts: list[ActSpec],
    raw_dir: Path,
    download: DownloadConfig,
    *,
    force: bool = False,
    client: httpx.Client | None = None,
) -> list[Path]:
    """Download missing acts and record their checksums in `raw_dir/manifest.json`.

    Existing files are kept unless `force` is set, so a manually placed file is never overwritten.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest: dict[str, dict[str, str]] = (
        json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    )
    owns_client = client is None
    http = client or httpx.Client(headers={"User-Agent": "eurolex-rag (research project)"})
    paths: list[Path] = []
    try:
        for act in acts:
            path = raw_dir / act.raw_filename
            if path.exists() and not force:
                logger.info("Using existing %s", path)
            else:
                try:
                    content = fetch_act(act, download, http)
                except (DownloadError, httpx.HTTPError) as exc:
                    raise DownloadError(
                        f"{exc}\n{manual_instructions(act, raw_dir, download)}"
                    ) from exc
                path.write_bytes(content)
                manifest[act.act_id] = {
                    "celex": act.celex,
                    "file": act.raw_filename,
                    "sha256": hashlib.sha256(content).hexdigest(),
                    "retrieved_at": datetime.now(UTC).isoformat(timespec="seconds"),
                    "source": download.cellar_url_template.format(celex=act.celex),
                }
                logger.info("Downloaded %s (%d bytes)", path, len(content))
            paths.append(path)
    finally:
        if owns_client:
            http.close()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return paths
