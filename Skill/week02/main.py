"""
Week 2 - run the same prompt against Groq (hosted) or Ollama (local).

Examples:
    python main.py
    python main.py "Explain embeddings in one paragraph."
    python main.py --provider ollama --model llama3.2 "What is RAG?"
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path


def load_llm_module():
    """Import llm-client.py (hyphenated name) as a normal module."""
    path = Path(__file__).with_name("llm-client.py")
    spec = importlib.util.spec_from_file_location("llm_client", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


llm_client = load_llm_module()
LLMClient = llm_client.LLMClient
LLMError = llm_client.LLMError

DEFAULT_PROMPT = "What is Artificial Intelligence? Explain in simple words."

# Accept a number, the provider name, or a partial match as the answer.
PROVIDER_CHOICES = {"1": "ollama", "2": "groq", "ollama": "ollama", "groq": "groq"}


def configured_provider() -> str:
    """Provider from .env (LLM_PROVIDER), or the built-in default."""
    return (os.getenv("LLM_PROVIDER") or llm_client.DEFAULT_PROVIDER).strip().lower()


def choose_provider(default: str, stdin=None) -> str | None:
    """Ask which provider (ollama or groq) should answer.

    Returns None when the session is not interactive, so piped input still
    works and scripts never block.
    """
    stdin = stdin or sys.stdin
    try:
        if not stdin.isatty():
            return None
    except (AttributeError, ValueError, OSError):
        return None

    print("\nWhich model should answer?")
    print("  1) ollama - local model on your machine")
    print("  2) groq   - hosted API")

    while True:
        try:
            answer = input(f"Choose [1/2] (default: {default}): ")
        except (EOFError, KeyboardInterrupt):
            print()
            return default
        answer = answer.strip().lower()
        if not answer:
            return default
        if answer in PROVIDER_CHOICES:
            return PROVIDER_CHOICES[answer]
        print("Please type 1 for ollama or 2 for groq.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chat with Groq or Ollama.")
    parser.add_argument(
        "prompt",
        nargs="*",
        help=f"Question to ask (default: {DEFAULT_PROMPT!r})",
    )
    parser.add_argument(
        "--provider",
        choices=["groq", "ollama"],
        default=None,
        help="Override LLM_PROVIDER from .env",
    )
    parser.add_argument("--model", default=None, help="Override the model name")
    parser.add_argument("--system", default=None, help="Optional system prompt")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prompt = " ".join(args.prompt).strip() or DEFAULT_PROMPT

    provider = args.provider or choose_provider(configured_provider())

    try:
        client = LLMClient.from_env(provider=provider, model=args.model)
    except (LLMError, ValueError) as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
        return 1

    if client.provider == "ollama" and not llm_client.ollama_is_running(client.host):
        print("Ollama is not running - starting it in the background...", file=sys.stderr)
        if not llm_client.ensure_ollama_running(client.host):
            print(
                f"Error: could not reach Ollama at {client.host}. "
                "Start it yourself with `ollama serve`.",
                file=sys.stderr,
            )
            return 1

    print(f"Provider: {client.provider} | Model: {client.model}\n")

    try:
        answer = client.chat(prompt, system=args.system)
    except LLMError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        if client.provider == "ollama":
            print(
                "Hint: make sure `ollama serve` is running and run "
                f"`ollama pull {client.model}` first.",
                file=sys.stderr,
            )
        return 1

    print(answer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
