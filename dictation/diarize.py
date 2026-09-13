"""Speaker diarization: who spoke when, from the audio itself (not text).

Uses pyannote.audio's pretrained pipeline — real acoustic speaker recognition,
not guessing from phrasing. This is a genuinely heavy, optional dependency
(torch + pyannote.audio, ~1-2GB installed); nothing in this module is
imported at package-import time, so `import dictation` stays cheap if this
feature is never used.

Requires:
  - the `pyannote.audio` package installed (see the "diarization" extra)
  - an HF_TOKEN (Hugging Face access token) whose account has accepted the
    usage conditions for pyannote/speaker-diarization-3.1 and
    pyannote/segmentation-3.0 on huggingface.co — each a one-time, free
    click, not an API cost. Without acceptance, loading the pipeline raises
    a clear error rather than a cryptic one.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional

from .asr import Segment


class DiarizationUnavailable(RuntimeError):
    """Raised when diarization can't run (no token, package missing, model access denied)."""


@dataclass
class SpeakerTurn:
    start: float
    end: float
    speaker: str  # pyannote's raw label, e.g. "SPEAKER_00"


class SpeakerDiarizer:
    """Thin wrapper around pyannote.audio. Model is lazy-loaded on first use."""

    def __init__(self, model_name: Optional[str] = None):
        self._model_name = model_name or os.environ.get(
            "DIARIZATION_MODEL", "pyannote/speaker-diarization-3.1"
        )
        self._pipeline = None

    def _load(self):
        if self._pipeline is None:
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                raise DiarizationUnavailable("HF_TOKEN is not set")
            try:
                from pyannote.audio import Pipeline
            except ImportError as exc:
                raise DiarizationUnavailable(
                    "the 'pyannote.audio' package is not installed"
                ) from exc
            try:
                pipeline = Pipeline.from_pretrained(self._model_name, use_auth_token=hf_token)
            except Exception as exc:
                raise DiarizationUnavailable(f"could not load diarization model: {exc}") from exc
            if pipeline is None:
                raise DiarizationUnavailable(
                    "diarization pipeline failed to load — check that the HF account "
                    "behind HF_TOKEN has accepted the model's usage conditions"
                )
            self._pipeline = pipeline
        return self._pipeline

    def diarize(self, audio_path: str) -> List[SpeakerTurn]:
        pipeline = self._load()
        diarization = pipeline(audio_path)
        return [
            SpeakerTurn(start=turn.start, end=turn.end, speaker=speaker)
            for turn, _, speaker in diarization.itertracks(yield_label=True)
        ]


def format_dialogue(segments: List[Segment], turns: List[SpeakerTurn]) -> str:
    """Combines Whisper's transcribed segments with pyannote's speaker turns.

    For each transcribed segment, picks whichever diarization turn overlaps it
    the most in time, then groups consecutive segments from the same speaker
    into one line. Raw pyannote labels ("SPEAKER_00") are renumbered to
    "Person 1", "Person 2", etc. in order of first appearance.
    """
    if not turns:
        text = " ".join(s.text for s in segments if s.text).strip()
        return f"Person 1: {text}" if text else ""

    def overlap(seg: Segment, turn: SpeakerTurn) -> float:
        return max(0.0, min(seg.end, turn.end) - max(seg.start, turn.start))

    speaker_order: List[str] = []
    lines: List[List[str]] = []  # list of [speaker_label, text, text, ...]

    for seg in segments:
        if not seg.text:
            continue
        best_turn = max(turns, key=lambda t: overlap(seg, t))
        raw_speaker = best_turn.speaker
        if raw_speaker not in speaker_order:
            speaker_order.append(raw_speaker)
        person_label = f"Person {speaker_order.index(raw_speaker) + 1}"

        if lines and lines[-1][0] == person_label:
            lines[-1].append(seg.text)
        else:
            lines.append([person_label, seg.text])

    return "\n".join(f"{label}: {' '.join(parts)}" for label, *parts in lines)
