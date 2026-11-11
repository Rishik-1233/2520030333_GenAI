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

    try:
        client = LLMClient.from_env(provider=args.provider, model=args.model)
    except (LLMError, ValueError) as exc:
        print(f"Setup error: {exc}", file=sys.stderr)
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
