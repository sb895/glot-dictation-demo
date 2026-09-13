import pytest

from dictation.asr import Segment
from dictation.diarize import DiarizationUnavailable, SpeakerDiarizer, SpeakerTurn, format_dialogue


def test_format_dialogue_aligns_segments_to_speaker_turns():
    segments = [
        Segment(start=0.0, end=3.0, text="Are you coming to the meeting tomorrow?"),
        Segment(start=3.0, end=6.0, text="Yeah, I will be there at ten a.m."),
        Segment(start=6.0, end=8.0, text="Great, see you then."),
    ]
    turns = [
        SpeakerTurn(start=0.03, end=1.94, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.75, end=4.86, speaker="SPEAKER_01"),
        SpeakerTurn(start=5.62, end=7.17, speaker="SPEAKER_00"),
    ]

    result = format_dialogue(segments, turns)

    assert result == (
        "Person 1: Are you coming to the meeting tomorrow?\n"
        "Person 2: Yeah, I will be there at ten a.m.\n"
        "Person 1: Great, see you then."
    )


def test_format_dialogue_groups_consecutive_same_speaker_segments():
    segments = [
        Segment(start=0.0, end=1.0, text="Hello"),
        Segment(start=1.0, end=2.0, text="there."),
    ]
    turns = [SpeakerTurn(start=0.0, end=2.0, speaker="SPEAKER_00")]

    assert format_dialogue(segments, turns) == "Person 1: Hello there."


def test_format_dialogue_no_turns_labels_everything_person_1():
    segments = [Segment(start=0.0, end=1.0, text="Hello there.")]
    assert format_dialogue(segments, []) == "Person 1: Hello there."


def test_format_dialogue_empty_segments_returns_empty_string():
    assert format_dialogue([], []) == ""


def test_diarizer_raises_when_no_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    diarizer = SpeakerDiarizer()
    with pytest.raises(DiarizationUnavailable):
        diarizer.diarize("some/audio.wav")
