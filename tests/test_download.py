import json
from pathlib import Path

import httpx
import pytest

from eurolex.config import ActSpec, DownloadConfig
from eurolex.ingestion.download import DownloadError, download_acts

ACT = ActSpec(
    act_id="gdpr",
    celex="32016R0679",
    raw_filename="gdpr_en.html",
    short_name="GDPR",
    expected_articles=99,
    expected_recitals=173,
)
CONFIG = DownloadConfig(
    cellar_url_template="https://cellar.example.test/celex/{celex}",
    source_url_template="https://eurlex.example.test/{celex}",
    timeout_seconds=5,
)
PAGE = b'<html><p class="eli-main-title">Regulation</p></html>'


def client(handler: httpx.MockTransport) -> httpx.Client:
    return httpx.Client(transport=handler)


def test_downloads_and_writes_manifest(tmp_path: Path) -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=PAGE)

    (path,) = download_acts([ACT], tmp_path, CONFIG, client=client(httpx.MockTransport(handler)))
    assert path.read_bytes() == PAGE
    assert str(seen[0].url) == "https://cellar.example.test/celex/32016R0679"
    assert seen[0].headers["Accept"] == "application/xhtml+xml"
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["gdpr"]["celex"] == "32016R0679"
    assert len(manifest["gdpr"]["sha256"]) == 64


def test_existing_file_is_not_overwritten(tmp_path: Path) -> None:
    (tmp_path / "gdpr_en.html").write_bytes(b"manual copy")

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    download_acts([ACT], tmp_path, CONFIG, client=client(httpx.MockTransport(handler)))
    assert (tmp_path / "gdpr_en.html").read_bytes() == b"manual copy"


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(202, headers={"x-amzn-waf-action": "challenge"}),
        httpx.Response(404, content=b"not found"),
        httpx.Response(200, content=b"<html>error page</html>"),
    ],
)
def test_failure_explains_manual_path(tmp_path: Path, response: httpx.Response) -> None:
    transport = httpx.MockTransport(lambda request: response)
    with pytest.raises(DownloadError, match="place it at") as info:
        download_acts([ACT], tmp_path, CONFIG, client=client(transport))
    assert "gdpr_en.html" in str(info.value)
    assert not (tmp_path / "gdpr_en.html").exists()
