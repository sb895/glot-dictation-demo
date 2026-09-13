"""High-level entry point: audio in, polished text out."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .asr import AudioDecodeError, AudioInput, WhisperASR, convert_to_wav, resolve_audio_path
from .diarize import DiarizationUnavailable, SpeakerDiarizer, format_dialogue
from .polish import polish

logger = logging.getLogger(__name__)


@dataclass
class DictationResult:
    raw_text: str
    text: str
    language: Optional[str]
    polish_method: str  # "llm:<backend>" or "regex_fallback"


class Pipeline:
    """The full press-speak-release pipeline: audio -> raw transcript -> polished text."""

    def __init__(
        self, asr: Optional[WhisperASR] = None, diarizer: Optional[SpeakerDiarizer] = None
    ):
        self._asr = asr or WhisperASR()
        self._diarizer = diarizer or SpeakerDiarizer()

    def run(
        self,
        audio: AudioInput,
        context: str = "general",
        backend: str = "default",
        diarize: bool = False,
    ) -> DictationResult:
        if not diarize:
            transcript = self._asr.transcribe(audio)
            polished_text, method = polish(transcript.text, context=context, backend=backend)
            return DictationResult(
                raw_text=transcript.text,
                text=polished_text,
                language=transcript.language,
                polish_method=method,
            )
        return self._run_diarized(audio, context=context, backend=backend)

    def _run_diarized(self, audio: AudioInput, context: str, backend: str) -> DictationResult:
        audio_path, tmp_path = resolve_audio_path(audio)
        wav_path = None
        try:
            segments, language = self._asr.transcribe_segments(audio_path)
            try:
                # pyannote's torchaudio backend can't read webm/Opus, unlike faster-whisper.
                wav_path = convert_to_wav(audio_path)
                turns = self._diarizer.diarize(wav_path)
                raw_text = format_dialogue(segments, turns)
                prediarized = True
            except (DiarizationUnavailable, AudioDecodeError) as exc:
                # No real diarization available — fall back to text-based inference.
                logger.warning(
                    "audio diarization unavailable, falling back to text-based inference: %s",
                    exc,
                )
                raw_text = " ".join(s.text for s in segments).strip()
                prediarized = False
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)
            if wav_path:
                Path(wav_path).unlink(missing_ok=True)

        polished_text, method = polish(
            raw_text, context=context, backend=backend, diarize=True, prediarized=prediarized
        )
        return DictationResult(
            raw_text=raw_text, text=polished_text, language=language, polish_method=method
        )
