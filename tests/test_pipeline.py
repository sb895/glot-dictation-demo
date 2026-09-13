from unittest.mock import MagicMock, patch

from dictation.asr import Segment
from dictation.diarize import DiarizationUnavailable, SpeakerTurn
from dictation.pipeline import Pipeline


def test_run_calls_asr_transcribe_with_just_the_audio():
    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = MagicMock(text="raw text", language="en")
    pipeline = Pipeline(asr=fake_asr)

    pipeline.run(b"audio-bytes", backend="local")

    fake_asr.transcribe.assert_called_once_with(b"audio-bytes")


def test_diarizer_never_invoked_when_diarize_is_false():
    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = MagicMock(text="hello there", language="en")
    fake_diarizer = MagicMock()
    fake_diarizer.diarize.side_effect = AssertionError(
        "diarizer.diarize() must never be called when diarize=False"
    )
    pipeline = Pipeline(asr=fake_asr, diarizer=fake_diarizer)

    result = pipeline.run(b"audio-bytes", backend="local", diarize=False)

    fake_diarizer.diarize.assert_not_called()
    fake_asr.transcribe_segments.assert_not_called()  # the segment-timing path is diarize-only
    assert "Person" not in result.text  # no speaker labels should ever appear


def test_diarizer_never_invoked_when_diarize_omitted_entirely():
    """diarize defaults to False — confirms the default itself is safe, not just
    an explicit diarize=False."""
    fake_asr = MagicMock()
    fake_asr.transcribe.return_value = MagicMock(text="hello there", language="en")
    fake_diarizer = MagicMock()
    fake_diarizer.diarize.side_effect = AssertionError(
        "diarizer.diarize() must never be called when diarize is omitted (defaults False)"
    )
    pipeline = Pipeline(asr=fake_asr, diarizer=fake_diarizer)

    pipeline.run(b"audio-bytes", backend="local")

    fake_diarizer.diarize.assert_not_called()
    fake_asr.transcribe_segments.assert_not_called()


def test_run_diarized_formats_dialogue_from_real_speaker_turns():
    fake_asr = MagicMock()
    fake_asr.transcribe_segments.return_value = (
        [Segment(start=0.0, end=1.0, text="hi"), Segment(start=1.0, end=2.0, text="hello")],
        "en",
    )
    fake_diarizer = MagicMock()
    fake_diarizer.diarize.return_value = [
        SpeakerTurn(start=0.0, end=1.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=1.0, end=2.0, speaker="SPEAKER_01"),
    ]
    pipeline = Pipeline(asr=fake_asr, diarizer=fake_diarizer)

    with patch("dictation.pipeline.polish", return_value=("Person 1: Hi.\nPerson 2: Hello.", "llm:gemini")) as mock_polish, \
         patch("dictation.pipeline.convert_to_wav", return_value="/tmp/fake-converted.wav"), \
         patch("dictation.pipeline.Path.unlink"):
        result = pipeline.run(b"audio-bytes", backend="gemini", diarize=True)

    assert result.raw_text == "Person 1: hi\nPerson 2: hello"
    assert result.text == "Person 1: Hi.\nPerson 2: Hello."
    mock_polish.assert_called_once_with(
        "Person 1: hi\nPerson 2: hello",
        context="general",
        backend="gemini",
        diarize=True,
        prediarized=True,
    )


def test_run_diarized_falls_back_to_text_inference_when_diarization_unavailable():
    fake_asr = MagicMock()
    fake_asr.transcribe_segments.return_value = (
        [Segment(start=0.0, end=1.0, text="hi there")],
        "en",
    )
    fake_diarizer = MagicMock()
    fake_diarizer.diarize.side_effect = DiarizationUnavailable("HF_TOKEN is not set")
    pipeline = Pipeline(asr=fake_asr, diarizer=fake_diarizer)

    with patch("dictation.pipeline.polish", return_value=("Person 1: Hi there.", "llm:gemini")) as mock_polish, \
         patch("dictation.pipeline.convert_to_wav", return_value="/tmp/fake-converted.wav"), \
         patch("dictation.pipeline.Path.unlink"):
        result = pipeline.run(b"audio-bytes", backend="gemini", diarize=True)

    assert result.raw_text == "hi there"
    mock_polish.assert_called_once_with(
        "hi there", context="general", backend="gemini", diarize=True, prediarized=False
    )
