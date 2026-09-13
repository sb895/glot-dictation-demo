import io
from unittest.mock import patch

import pytest

from dictation.asr import AudioDecodeError
from dictation.pipeline import DictationResult
from webapp.app import app


@pytest.fixture
def client():
    return app.test_client()


def test_index_page_loads(client):
    resp = client.get("/")
    assert resp.status_code == 200


def test_full_mode_still_calls_polish_pipeline(client):
    fake_result = DictationResult(
        raw_text="raw", text="polished", language="en", polish_method="regex_fallback"
    )
    with patch("webapp.app._pipeline.run", return_value=fake_result) as mock_run:
        resp = client.post(
            "/api/transcribe",
            data={"audio": (io.BytesIO(b"fake-audio-bytes"), "audio.webm"), "backend": "default"},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 200
    assert resp.get_json()["text"] == "polished"
    mock_run.assert_called_once()


def test_diarize_defaults_to_false_when_field_absent(client):
    """The frontend only ever sends the "diarize" field when the toggle is
    checked — confirms the route treats "field not sent" as False, not just
    "field sent as 0", matching what an unchecked checkbox actually produces."""
    fake_result = DictationResult(
        raw_text="raw", text="polished", language="en", polish_method="regex_fallback"
    )
    with patch("webapp.app._pipeline.run", return_value=fake_result) as mock_run:
        client.post(
            "/api/transcribe",
            data={"audio": (io.BytesIO(b"fake-audio-bytes"), "audio.webm"), "backend": "gemini"},
            content_type="multipart/form-data",
        )
    _, kwargs = mock_run.call_args
    assert kwargs["diarize"] is False


def test_diarize_true_only_when_field_is_exactly_1(client):
    fake_result = DictationResult(
        raw_text="raw", text="polished", language="en", polish_method="regex_fallback"
    )
    with patch("webapp.app._pipeline.run", return_value=fake_result) as mock_run:
        client.post(
            "/api/transcribe",
            data={
                "audio": (io.BytesIO(b"fake-audio-bytes"), "audio.webm"),
                "backend": "gemini",
                "diarize": "1",
            },
            content_type="multipart/form-data",
        )
    _, kwargs = mock_run.call_args
    assert kwargs["diarize"] is True


def test_full_mode_returns_400_on_decode_error(client):
    with patch("webapp.app._pipeline.run", side_effect=AudioDecodeError("bad audio")):
        resp = client.post(
            "/api/transcribe",
            data={"audio": (io.BytesIO(b"garbage"), "audio.webm")},
            content_type="multipart/form-data",
        )
    assert resp.status_code == 400
