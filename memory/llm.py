"""LLM seam: a single ``complete`` call over the OpenRouter chat API.

Every component that needs an LLM (extraction, dedup judging, ...) goes through
this one function so tests can monkeypatch ``memory.llm.complete`` wholesale.
The OpenAI client is created lazily on first use, so importing this module makes
no network call and requires no API key.
"""
from __future__ import annotations

import os
from typing import Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_ENV_PATH = "/workspace/AIEHackathon/.env"


def _load_env_file(path: str = _ENV_PATH) -> None:
    """Load ``KEY=value`` pairs from a .env file into ``os.environ`` (no override)."""
    try:
        from dotenv import load_dotenv  # type: ignore
    except ImportError:
        if not os.path.exists(path):
            return
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                value = value.strip()
                # strip a matched pair of surrounding quotes (python-dotenv does this)
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                os.environ.setdefault(key.strip(), value)
    else:
        load_dotenv(path, override=False)


# Resolve the default model after .env is available so OPENROUTER_MODEL is honored.
_load_env_file()
DEFAULT_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-4.6")

_client = None  # module singleton, created on first complete() call


def _get_client():
    """Lazily build the OpenAI-SDK client pointed at OpenRouter (cached)."""
    global _client
    if _client is None:
        from openai import OpenAI

        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is not set (env or /workspace/AIEHackathon/.env)."
            )
        _client = OpenAI(
            api_key=api_key,
            base_url=OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": "https://github.com/xesws/AIEHackathon",
                "X-Title": "Engram",
            },
        )
    return _client


def complete(
    messages: list[dict],
    *,
    model: Optional[str] = None,
    temperature: float = 0.0,
    response_format: Optional[dict] = None,
) -> str:
    """Run one chat completion and return the assistant message text.

    Args:
        messages:        chat messages, e.g. ``[{"role": "user", "content": ...}]``.
        model:           model id; defaults to ``DEFAULT_MODEL`` when omitted.
        temperature:     sampling temperature (0.0 = deterministic).
        response_format: optional OpenAI ``response_format`` (e.g. JSON mode).

    Raises:
        RuntimeError: if ``OPENROUTER_API_KEY`` is missing when first called.
    """
    client = _get_client()
    kwargs: dict = {
        "model": model or DEFAULT_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    completion = client.chat.completions.create(**kwargs)
    return completion.choices[0].message.content
