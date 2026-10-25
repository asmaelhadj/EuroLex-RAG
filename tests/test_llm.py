import json

import httpx
import pytest

from eurolex.generation.llm import LLMError, OllamaClient
from eurolex.retrieval.rewrite import QueryRewriter, clean_rewrite


def ollama(handler: httpx.MockTransport, retries: int = 2) -> OllamaClient:
    return OllamaClient(
        "http://ollama.test/", "m", retries=retries, client=httpx.Client(transport=handler)
    )


def test_generate_sends_model_system_and_options() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://ollama.test/api/generate"
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "hello"})

    assert ollama(httpx.MockTransport(handler)).generate("hi", system="be brief") == "hello"
    body = seen[0]
    assert body["model"] == "m"
    assert body["system"] == "be brief"
    assert body["stream"] is False
    assert body["options"] == {"temperature": 0.0, "seed": 42}
    assert "format" not in body


def test_generate_json_schema_and_temperature() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "{}"})

    schema: dict[str, object] = {"type": "object"}
    ollama(httpx.MockTransport(handler)).generate("hi", json_schema=schema, temperature=0.7)
    assert seen[0]["format"] == schema
    assert seen[0]["options"] == {"temperature": 0.7, "seed": 42}


def test_num_gpu_option() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return httpx.Response(200, json={"response": "ok"})

    transport = httpx.MockTransport(handler)
    OllamaClient(
        "http://o.test", "m", num_gpu=0, client=httpx.Client(transport=transport)
    ).generate("hi")
    assert seen[0]["options"] == {"temperature": 0.0, "seed": 42, "num_gpu": 0}


def test_retries_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eurolex.generation.llm.time.sleep", lambda s: None)
    calls = iter([httpx.Response(500), httpx.Response(200, json={"response": "ok"})])
    client = ollama(httpx.MockTransport(lambda request: next(calls)))
    assert client.generate("hi") == "ok"


def test_gives_up_after_retries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("eurolex.generation.llm.time.sleep", lambda s: None)
    count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal count
        count += 1
        return httpx.Response(503)

    with pytest.raises(LLMError):
        ollama(httpx.MockTransport(handler), retries=2).generate("hi")
    assert count == 3


class FakeLLM:
    model = "fake"

    def __init__(self, reply: str | None) -> None:
        self.reply = reply
        self.prompts: list[str] = []

    def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        json_schema: dict[str, object] | None = None,
        temperature: float | None = None,
    ) -> str:
        self.prompts.append(prompt)
        if self.reply is None:
            raise LLMError("down")
        return self.reply


def test_rewrite_modes() -> None:
    llm = FakeLLM("right to erasure of personal data")
    assert QueryRewriter(llm, "replace").rewrite("delete my info?") == (
        "right to erasure of personal data"
    )
    assert QueryRewriter(llm, "expand").rewrite("delete my info?") == (
        "delete my info? right to erasure of personal data"
    )
    assert "delete my info?" in llm.prompts[0]


def test_rewrite_failure_returns_none() -> None:
    assert QueryRewriter(FakeLLM(None)).rewrite("q?") is None
    assert QueryRewriter(FakeLLM("   \n  ")).rewrite("q?") is None


@pytest.mark.parametrize(
    ("name", "expected"),
    [("mistral", True), ("mistral:latest", True), ("qwen2.5:7b-instruct", True), ("llama3", False)],
)
def test_has_model(name: str, expected: bool) -> None:
    tags = {"models": [{"name": "mistral:latest"}, {"name": "qwen2.5:7b-instruct"}]}
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=tags))
    client = OllamaClient("http://o.test", name, client=httpx.Client(transport=transport))
    assert client.has_model() is expected


def test_has_model_when_ollama_is_down() -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(503))
    client = OllamaClient("http://o.test", "mistral", client=httpx.Client(transport=transport))
    assert not client.has_model()


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ('"lawful basis for processing"', "lawful basis for processing"),
        ("Search query: data portability\nBecause...", "data portability"),
        ("\n\n  right to object  \n", "right to object"),
        ("x" * 500, "x" * 300),
    ],
)
def test_clean_rewrite(raw: str, expected: str) -> None:
    assert clean_rewrite(raw) == expected
