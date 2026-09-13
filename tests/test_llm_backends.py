import io
import json
from unittest.mock import MagicMock, patch

import pytest

from dictation import llm
from dictation.polish import polish


def test_resolve_backend_default_maps_to_configured_backend(monkeypatch):
    monkeypatch.setenv("DEFAULT_LLM_BACKEND", "gemini")
    assert llm.resolve_backend("default") == "gemini"
    assert llm.resolve_backend("openai") == "openai"


def test_unknown_backend_raises_llm_unavailable():
    with pytest.raises(llm.LLMUnavailable):
        llm.call("sys", "user", backend="not-a-real-backend")


def test_openai_backend_missing_key_raises(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(llm.LLMUnavailable):
        llm.call("sys", "user", backend="openai")


def test_openai_backend_calls_sdk_with_expected_shape(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch("openai.OpenAI") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Cleaned text."))]
        mock_client.chat.completions.create.return_value = mock_response

        result = llm.call("system prompt", "raw text", backend="openai")

    assert result == "Cleaned text."
    _, kwargs = mock_client.chat.completions.create.call_args
    assert kwargs["messages"][0] == {"role": "system", "content": "system prompt"}
    assert kwargs["messages"][1] == {"role": "user", "content": "raw text"}


def test_gemini_backend_missing_key_raises(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(llm.LLMUnavailable):
        llm.call("sys", "user", backend="gemini")


def test_gemini_backend_calls_sdk_with_expected_shape(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("google.genai.Client") as mock_cls:
        mock_client = mock_cls.return_value
        mock_response = MagicMock(text="Cleaned text.")
        mock_client.models.generate_content.return_value = mock_response

        result = llm.call("system prompt", "raw text", backend="gemini")

    assert result == "Cleaned text."
    _, kwargs = mock_client.models.generate_content.call_args
    assert kwargs["contents"] == "raw text"


def test_local_backend_calls_ollama_http_api(monkeypatch):
    fake_response_body = json.dumps({"message": {"content": "Cleaned text."}}).encode()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResponse(fake_response_body)) as mock_open:
        result = llm.call("system prompt", "raw text", backend="local")

    assert result == "Cleaned text."
    request_obj = mock_open.call_args[0][0]
    body = json.loads(request_obj.data)
    assert body["messages"][1]["content"] == "raw text"


def test_local_backend_unreachable_raises_llm_unavailable():
    import urllib.error

    with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
        with pytest.raises(llm.LLMUnavailable):
            llm.call("sys", "user", backend="local")


def test_local_llama_backend_calls_ollama_with_its_own_model(monkeypatch):
    monkeypatch.delenv("OLLAMA_MODEL_LLAMA", raising=False)
    fake_response_body = json.dumps({"message": {"content": "Cleaned text."}}).encode()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch("urllib.request.urlopen", return_value=FakeResponse(fake_response_body)) as mock_open:
        result = llm.call("system prompt", "raw text", backend="local_llama")

    assert result == "Cleaned text."
    request_obj = mock_open.call_args[0][0]
    body = json.loads(request_obj.data)
    assert body["model"] == "llama3.2:3b"
    assert body["messages"][1]["content"] == "raw text"


def test_local_and_local_llama_use_independent_model_env_vars(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "gemma3:4b")
    monkeypatch.setenv("OLLAMA_MODEL_LLAMA", "llama3.2:3b")
    fake_response_body = json.dumps({"message": {"content": "ok"}}).encode()

    class FakeResponse(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "urllib.request.urlopen", side_effect=lambda *a, **kw: FakeResponse(fake_response_body)
    ) as mock_open:
        llm.call("sys", "user", backend="local")
        gemma_model = json.loads(mock_open.call_args[0][0].data)["model"]

        llm.call("sys", "user", backend="local_llama")
        llama_model = json.loads(mock_open.call_args[0][0].data)["model"]

    assert gemma_model == "gemma3:4b"
    assert llama_model == "llama3.2:3b"


def test_polish_falls_back_on_provider_error_not_just_missing_key(monkeypatch):
    """A configured-but-broken backend (bad key, rate limit, etc.) should degrade
    to the offline cleanup rather than raising and crashing the request."""
    monkeypatch.setenv("OPENAI_API_KEY", "fake-key")
    with patch("openai.OpenAI") as mock_cls:
        mock_cls.return_value.chat.completions.create.side_effect = RuntimeError("401 unauthorized")
        text, method = polish("um hello there", backend="openai")

    assert method == "regex_fallback"
    assert text == "Hello there."
