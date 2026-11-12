"""
Week 2 - one LLM client, two providers.

* groq   -> hosted API (needs GROQ_API_KEY in .env)
* ollama -> local models (needs `ollama serve` running on your machine)

Filename note: this file keeps the requested hyphenated name, so a normal
`import llm_client` will *not* work. main.py and test_client.py load it with
importlib (see load_llm_module() in those files).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import time
import urllib.request
from collections.abc import Mapping, Sequence

from dotenv import load_dotenv

# Loaded once, at import time, so tests and callers can override env vars.
load_dotenv()

DEFAULT_PROVIDER = "groq"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-20b"
DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"

Message = dict[str, str]


class LLMError(RuntimeError):
    """Raised when the client is misconfigured or the provider call fails."""


def _normalise_host(host: str) -> str:
    host = host.strip().rstrip("/")
    if not host.startswith(("http://", "https://")):
        host = f"http://{host}"
    return host


def ollama_is_running(host: str | None = None, timeout: float = 2.0) -> bool:
    """Return True when an Ollama server answers at host."""
    url = _normalise_host(
        host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
    ) + "/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def ensure_ollama_running(
    host: str | None = None, wait_seconds: float = 20.0
) -> bool:
    """Make sure an Ollama server is up, starting `ollama serve` in the
    background if it is not. Returns True once the server responds."""
    if ollama_is_running(host):
        return True

    executable = shutil.which("ollama")
    if not executable:
        return False

    spawn_kwargs: dict[str, object] = {}
    if os.name == "nt":
        spawn_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        spawn_kwargs["start_new_session"] = True

    try:
        subprocess.Popen(
            [executable, "serve"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **spawn_kwargs,
        )
    except OSError:
        return False

    deadline = time.monotonic() + wait_seconds
    while time.monotonic() < deadline:
        if ollama_is_running(host, timeout=1.0):
            return True
        time.sleep(0.25)
    return False


def _extract_content(response: object) -> str:
    """Read the answer out of a Groq or Ollama response object."""
    choices = getattr(response, "choices", None)
    if choices:  # Groq / OpenAI style
        message = choices[0].message
        return (getattr(message, "content", None) or "").strip()

    message = getattr(response, "message", None)
    if message is None and isinstance(response, Mapping):
        message = response.get("message")
    if isinstance(message, Mapping):
        return (message.get("content") or "").strip()
    return (getattr(message, "content", None) or "").strip()


class LLMClient:
    """Small wrapper that gives Groq and Ollama the same .chat() interface."""

    def __init__(
        self,
        provider: str = DEFAULT_PROVIDER,
        model: str | None = None,
        api_key: str | None = None,
        host: str | None = None,
        temperature: float = 0.7,
        system: str | None = None,
    ) -> None:
        self.provider = (provider or DEFAULT_PROVIDER).strip().lower()
        self.temperature = temperature
        self.system = system

        if self.provider == "groq":
            self.api_key = api_key or os.getenv("GROQ_API_KEY", "").strip()
            if not self.api_key or self.api_key.startswith("your_"):
                raise LLMError(
                    "GROQ_API_KEY is missing or still the placeholder. "
                    "Put GROQ_API_KEY=your_real_key in week02/.env."
                )
            self.model = model or os.getenv("GROQ_MODEL") or DEFAULT_GROQ_MODEL
            self.host = None
            from groq import Groq

            self._client = Groq(api_key=self.api_key)

        elif self.provider == "ollama":
            self.model = model or os.getenv("OLLAMA_MODEL") or DEFAULT_OLLAMA_MODEL
            self.host = _normalise_host(
                host or os.getenv("OLLAMA_HOST") or DEFAULT_OLLAMA_HOST
            )
            self.api_key = None
            from ollama import Client

            self._client = Client(host=self.host)

        else:
            raise ValueError(
                f"Unknown provider {self.provider!r}. Use 'groq' or 'ollama'."
            )

    @classmethod
    def from_env(
        cls, provider: str | None = None, model: str | None = None, **kwargs
    ) -> "LLMClient":
        """Build a client from LLM_PROVIDER / .env, with explicit args winning."""
        chosen = provider or os.getenv("LLM_PROVIDER") or DEFAULT_PROVIDER
        return cls(provider=chosen, model=model, **kwargs)

    def _build_messages(
        self, prompt: str, history: Sequence[Message] | None
    ) -> list[Message]:
        messages: list[Message] = []
        system = self.system
        if system:
            messages.append({"role": "system", "content": system})
        if history:
            messages.extend(dict(item) for item in history)
        messages.append({"role": "user", "content": prompt})
        return messages

    def chat(
        self,
        prompt: str,
        *,
        system: str | None = None,
        history: Sequence[Message] | None = None,
    ) -> str:
        """Send one prompt and return the assistant's reply as plain text."""
        if not prompt or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        messages = self._build_messages(prompt, history)
        system = system or self.system
        if system and messages[0]["role"] != "system":
            messages.insert(0, {"role": "system", "content": system})

        try:
            if self.provider == "groq":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                )
            else:
                response = self._client.chat(
                    model=self.model,
                    messages=messages,
                    options={"temperature": self.temperature},
                    stream=False,
                )
        except Exception as exc:  # noqa: BLE001 - surface a clear, single error type
            raise LLMError(f"{self.provider} request failed: {exc}") from exc

        answer = _extract_content(response)
        if not answer:
            raise LLMError(f"{self.provider} returned an empty response")
        return answer

    def __repr__(self) -> str:
        return f"LLMClient(provider={self.provider!r}, model={self.model!r})"
