from unittest.mock import MagicMock, patch

from dictation.polish import (
    DIARIZE_INFER_INSTRUCTION,
    DIARIZE_PRESERVE_INSTRUCTION,
    polish,
    regex_cleanup,
)


def test_regex_cleanup_strips_fillers_and_tidies_casing():
    raw = "um so basically i think, like, we should ship it"
    cleaned = regex_cleanup(raw)
    assert "um" not in cleaned.lower().split()
    assert "like" not in cleaned.lower()
    assert cleaned[0].isupper()
    assert cleaned.endswith(".")


def test_regex_cleanup_empty_input():
    assert regex_cleanup("") == ""


def test_polish_falls_back_to_regex_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    text, method = polish("um hello there", backend="gemini")
    assert method == "regex_fallback"
    assert text == "Hello there."


def test_polish_empty_transcript_short_circuits():
    text, method = polish("")
    assert text == ""
    assert method == "regex_fallback"


def test_polish_diarize_regex_fallback_labels_single_speaker(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    text, method = polish("um hello there", backend="gemini", diarize=True)
    assert method == "regex_fallback"
    assert text == "Person 1: Hello there."


def test_polish_diarize_appends_instruction_to_system_prompt(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("google.genai.Client") as mock_cls:
        mock_response = MagicMock(text="Person 1: Hi.\nPerson 2: Hello.")
        mock_cls.return_value.models.generate_content.return_value = mock_response

        text, method = polish("hi hello", backend="gemini", diarize=True)

    assert method == "llm:gemini"
    assert text == "Person 1: Hi.\nPerson 2: Hello."
    _, kwargs = mock_cls.return_value.models.generate_content.call_args
    assert DIARIZE_INFER_INSTRUCTION in kwargs["config"].system_instruction
    assert DIARIZE_PRESERVE_INSTRUCTION not in kwargs["config"].system_instruction


def test_polish_without_diarize_omits_instruction(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("google.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.return_value = MagicMock(text="Hi hello.")

        polish("hi hello", backend="gemini", diarize=False)

    _, kwargs = mock_cls.return_value.models.generate_content.call_args
    assert DIARIZE_INFER_INSTRUCTION not in kwargs["config"].system_instruction
    assert DIARIZE_PRESERVE_INSTRUCTION not in kwargs["config"].system_instruction


def test_polish_prediarized_uses_preserve_instruction(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("google.genai.Client") as mock_cls:
        mock_response = MagicMock(text="Person 1: Hi.\nPerson 2: Hello.")
        mock_cls.return_value.models.generate_content.return_value = mock_response

        text, method = polish(
            "Person 1: um hi\nPerson 2: hello",
            backend="gemini",
            diarize=True,
            prediarized=True,
        )

    assert method == "llm:gemini"
    assert text == "Person 1: Hi.\nPerson 2: Hello."
    _, kwargs = mock_cls.return_value.models.generate_content.call_args
    assert DIARIZE_PRESERVE_INSTRUCTION in kwargs["config"].system_instruction
    assert DIARIZE_INFER_INSTRUCTION not in kwargs["config"].system_instruction


def test_polish_prediarized_regex_fallback_cleans_each_line(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    text, method = polish(
        "Person 1: um hello there\nPerson 2: like i think so",
        backend="gemini",
        diarize=True,
        prediarized=True,
    )
    assert method == "regex_fallback"
    assert text == "Person 1: Hello there.\nPerson 2: I think so."


def test_polish_flags_quota_exceeded_distinctly_from_other_failures(monkeypatch):
    """Regression test for a real Gemini 429 hit in production: the fallback
    still runs, but the UI needs to tell "you're out of quota today" apart
    from a generic/unexplained backend failure."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    quota_error = Exception(
        "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, "
        "'message': 'Quota exceeded for metric: generate_content_free_tier_requests'}}"
    )
    with patch("google.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.side_effect = quota_error
        text, method = polish("um hello there", backend="gemini")

    assert method == "regex_fallback_quota"
    assert text == "Hello there."


def test_polish_other_failures_use_plain_regex_fallback(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    with patch("google.genai.Client") as mock_cls:
        mock_cls.return_value.models.generate_content.side_effect = Exception("network unreachable")
        text, method = polish("um hello there", backend="gemini")

    assert method == "regex_fallback"
