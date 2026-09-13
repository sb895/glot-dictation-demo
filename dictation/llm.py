"""LLM backends for the polish step.

Every backend is a plain function: (system, user, model) -> str. `call()` picks
one by name and dispatches to it. Adding a new provider means adding one
function and one registry entry — nothing else in the package needs to change.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


class LLMUnavailable(RuntimeError):
    """Raised when the selected backend isn't usable (missing key, package, or service)."""


def _call_openai(system: str, user: str, model: str | None = None) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise LLMUnavailable("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise LLMUnavailable("the 'openai' package is not installed") from exc

    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model or os.environ.get("OPENAI_MODEL", "gpt-5.4-nano"),
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return response.choices[0].message.content.strip()


def _call_gemini(system: str, user: str, model: str | None = None) -> str:
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise LLMUnavailable("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set")
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMUnavailable("the 'google-genai' package is not installed") from exc

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model or os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite"),
        contents=user,
        config=types.GenerateContentConfig(system_instruction=system),
    )
    return response.text.strip()


def _call_ollama_chat(model_name: str, system: str, user: str) -> str:
    """Shared HTTP call to Ollama's /api/chat — used by every local model backend."""
    payload = json.dumps(
        {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
        }
    ).encode()
    request = urllib.request.Request(
        f"{OLLAMA_HOST}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        # 180s: first load of a multi-GB model into memory can take over a minute.
        with urllib.request.urlopen(request, timeout=180) as resp:
            body = json.loads(resp.read())
    except urllib.error.URLError as exc:
        raise LLMUnavailable(
            f"couldn't reach Ollama at {OLLAMA_HOST} (is it running? try `ollama serve`)"
        ) from exc
    try:
        return body["message"]["content"].strip()
    except (KeyError, TypeError) as exc:
        raise LLMUnavailable(f"unexpected response from Ollama: {body}") from exc


def _call_local(system: str, user: str, model: str | None = None) -> str:
    """Local model served by Ollama (Gemma) — no API key, runs on-device."""
    model_name = model or os.environ.get("OLLAMA_MODEL", "gemma3")
    return _call_ollama_chat(model_name, system, user)


def _call_local_llama(system: str, user: str, model: str | None = None) -> str:
    """Local model served by Ollama (Llama 3, small variant) — no API key, runs on-device.

    3b, not 1b: verified empirically that 1b is unreliable for this specific
    task (resolving self-corrections) — across a small sample it flipped the
    speaker's actual final choice, and once fabricated an instruction that was
    never said. 3b resolved the same test correctly 5/5 times.
    """
    model_name = model or os.environ.get("OLLAMA_MODEL_LLAMA", "llama3.2:3b")
    return _call_ollama_chat(model_name, system, user)


_BACKENDS = {
    "openai": _call_openai,
    "gemini": _call_gemini,
    "local": _call_local,
    "local_llama": _call_local_llama,
}


def resolve_backend(backend: str) -> str:
    """"default" -> the configured DEFAULT_LLM_BACKEND; anything else passes through."""
    if backend != "default":
        return backend
    return os.environ.get("DEFAULT_LLM_BACKEND", "local")


def call(system: str, user: str, backend: str = "default", model: str | None = None) -> str:
    """Dispatch to the requested backend. "default" resolves via DEFAULT_LLM_BACKEND."""
    resolved = resolve_backend(backend)
    fn = _BACKENDS.get(resolved)
    if fn is None:
        raise LLMUnavailable(f"unknown backend '{resolved}'; choose one of {list(_BACKENDS)}")
    return fn(system, user, model)
