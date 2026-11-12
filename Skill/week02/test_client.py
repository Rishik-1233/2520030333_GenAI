"""
Week 2 - tests for llm-client.py.

Offline tests (default):
    pytest -q

Live tests (real API / local model, skipped unless enabled):
    set GROQ_API_KEY in .env, then: pytest -q -k groq_live
    start ollama, then:              pytest -q -k ollama_live --live
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

load_dotenv()


def load_llm_module():
    """llm-client.py has a hyphen, so import it by path."""
    path = Path(__file__).with_name("llm-client.py")
    spec = importlib.util.spec_from_file_location("llm_client", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llm_client = load_llm_module()
LLMClient = llm_client.LLMClient
LLMError = llm_client.LLMError


class _FakeGroqCompletions:
    def __init__(self) -> None:
        self.last_request: dict | None = None

    def create(self, **kwargs):
        self.last_request = kwargs
        message = type("Message", (), {"content": "  hello from groq  "})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


class _FakeGroqClient:
    def __init__(self) -> None:
        completions = _FakeGroqCompletions()
        self.completions = completions
        self.chat = type("Chat", (), {"completions": completions})()


class _FakeOllamaClient:
    def __init__(self) -> None:
        self.last_request: dict | None = None

    def chat(self, **kwargs):
        self.last_request = kwargs
        return {"message": {"role": "assistant", "content": "hello from ollama"}}


@pytest.fixture
def groq_client(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    pytest.importorskip("groq")
    client = LLMClient(provider="groq")
    fake = _FakeGroqClient()
    client._client = fake
    return client, fake


@pytest.fixture
def ollama_client(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2")
    pytest.importorskip("ollama")
    client = LLMClient(provider="ollama", host="127.0.0.1:11434")
    fake = _FakeOllamaClient()
    client._client = fake
    return client, fake


def test_default_provider_is_groq(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert llm_client.DEFAULT_PROVIDER == "groq"


def test_groq_reads_default_model(groq_client):
    client, _ = groq_client
    assert client.provider == "groq"
    assert client.model == llm_client.DEFAULT_GROQ_MODEL


def test_groq_requires_api_key(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    pytest.importorskip("groq")
    with pytest.raises(LLMError, match="GROQ_API_KEY"):
        LLMClient(provider="groq", api_key="")


def test_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        LLMClient(provider="openai")


def test_ollama_host_gets_scheme(ollama_client):
    client, _ = ollama_client
    assert client.host == "http://127.0.0.1:11434"


def test_groq_chat_sends_messages_and_strips(groq_client):
    client, fake = groq_client
    answer = client.chat("Hi", history=[{"role": "user", "content": "earlier"}])

    assert answer == "hello from groq"
    assert fake.completions.last_request["model"] == client.model
    assert fake.completions.last_request["messages"] == [
        {"role": "user", "content": "earlier"},
        {"role": "user", "content": "Hi"},
    ]


def test_system_prompt_is_prepended(ollama_client):
    client, fake = ollama_client
    client.chat("Hi", system="You are terse.")

    messages = fake.last_request["messages"]
    assert messages[0] == {"role": "system", "content": "You are terse."}
    assert messages[-1] == {"role": "user", "content": "Hi"}
    assert fake.last_request["options"]["temperature"] == client.temperature


def test_ollama_chat_returns_text(ollama_client):
    client, _ = ollama_client
    assert client.chat("Hi") == "hello from ollama"


def test_empty_prompt_rejected(groq_client):
    client, _ = groq_client
    with pytest.raises(ValueError, match="non-empty"):
        client.chat("   ")


def test_provider_failure_wrapped(ollama_client):
    client, fake = ollama_client

    def boom(**kwargs):
        raise ConnectionError("connection refused")

    fake.chat = boom
    with pytest.raises(LLMError, match="ollama request failed"):
        client.chat("Hi")


def test_from_env_uses_provider_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    pytest.importorskip("ollama")
    assert LLMClient.from_env(provider="ollama").provider == "ollama"


class _FakeStdin:
    def __init__(self, interactive: bool) -> None:
        self._interactive = interactive

    def isatty(self) -> bool:
        return self._interactive


def load_main_module():
    path = Path(__file__).with_name("main.py")
    spec = importlib.util.spec_from_file_location("week02_main", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_choose_provider_accepts_number(monkeypatch):
    main = load_main_module()
    monkeypatch.setattr("builtins.input", lambda prompt="": "2")
    assert main.choose_provider("ollama", stdin=_FakeStdin(True)) == "groq"


def test_choose_provider_accepts_name(monkeypatch):
    main = load_main_module()
    monkeypatch.setattr("builtins.input", lambda prompt="": "OLLAMA")
    assert main.choose_provider("groq", stdin=_FakeStdin(True)) == "ollama"


def test_choose_provider_reasks_until_valid(monkeypatch):
    main = load_main_module()
    answers = iter(["banana", "1"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(answers))
    assert main.choose_provider("groq", stdin=_FakeStdin(True)) == "ollama"


def test_choose_provider_uses_default_on_empty_input(monkeypatch):
    main = load_main_module()
    monkeypatch.setattr("builtins.input", lambda prompt="": "  ")
    assert main.choose_provider("groq", stdin=_FakeStdin(True)) == "groq"


def test_choose_provider_skips_prompt_when_not_interactive(monkeypatch):
    main = load_main_module()

    def boom(prompt=""):
        raise AssertionError("input() must not be called for piped input")

    monkeypatch.setattr("builtins.input", boom)
    assert main.choose_provider("groq", stdin=_FakeStdin(False)) is None


def test_ollama_is_running_false_when_unreachable():
    assert llm_client.ollama_is_running("127.0.0.1:9", timeout=0.5) is False


def test_ensure_ollama_running_true_when_already_up(monkeypatch):
    monkeypatch.setattr(llm_client, "ollama_is_running", lambda *a, **k: True)
    assert llm_client.ensure_ollama_running("http://127.0.0.1:11434") is True


def test_ensure_ollama_running_false_without_binary(monkeypatch):
    monkeypatch.setattr(llm_client, "ollama_is_running", lambda *a, **k: False)
    monkeypatch.setattr(llm_client.shutil, "which", lambda name: None)
    assert llm_client.ensure_ollama_running("http://127.0.0.1:11434") is False


# --- live tests -----------------------------------------------------------
# These hit a real service, so they are opt-in.


@pytest.mark.skipif(
    not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here",
    reason="GROQ_API_KEY not configured",
)
def test_groq_live():
    pytest.importorskip("groq")
    answer = LLMClient(provider="groq").chat("Reply with the single word: pong")
    assert answer


@pytest.mark.skipif(
    os.getenv("RUN_OLLAMA_TESTS", "").lower() not in {"1", "true", "yes"},
    reason="set RUN_OLLAMA_TESTS=1 to run local model tests",
)
def test_ollama_live():
    pytest.importorskip("ollama")
    answer = LLMClient.from_env(provider="ollama").chat(
        "Reply with the single word: pong"
    )
    assert answer
