"""Speech-to-text backend.

Uses faster-whisper (CTranslate2) for local, offline transcription — no API key
required for this stage, which keeps the demo usable with zero setup friction.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

AudioInput = Union[str, Path, bytes]


class AudioDecodeError(RuntimeError):
    """Raised when the given audio can't be decoded (corrupt, empty, or unsupported)."""


@dataclass
class Transcript:
    text: str
    language: Optional[str] = None


@dataclass
class Segment:
    start: float
    end: float
    text: str


def resolve_audio_path(audio: AudioInput) -> Tuple[str, Optional[str]]:
    """Returns (path_to_use, temp_path_to_clean_up_or_None).

    Writes bytes to a temp file if needed; a caller that gets a non-None
    second value is responsible for deleting it when done. Shared by
    WhisperASR and the diarization path, which both need a real file path
    and — when diarizing — need that *same* path for two separate calls.
    """
    if isinstance(audio, (bytes, bytearray)):
        fd, tmp_path = tempfile.mkstemp(suffix=".audio")
        with os.fdopen(fd, "wb") as f:
            f.write(audio)
        return tmp_path, tmp_path
    return str(audio), None


def convert_to_wav(input_path: str) -> str:
    """Decodes any audio PyAV/ffmpeg can read and writes it to a new temp WAV file.

    Needed only for diarization: browsers record audio/webm (Opus), which PyAV
    (used by faster-whisper) decodes fine, but torchaudio's soundfile backend
    (used internally by pyannote.audio) can't read at all — confirmed via
    LibsndfileError: "Format not recognised." WAV is the one format both sides
    of this pipeline can always read, so diarization gets its own converted
    copy rather than depending on the original upload's format. The caller
    owns the returned path and must delete it when done.
    """
    import av

    fd, wav_path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        input_container = av.open(input_path)
        output_container = av.open(wav_path, mode="w")
        output_stream = output_container.add_stream("pcm_s16le", rate=16000)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
        try:
            for frame in input_container.decode(audio=0):
                for resampled_frame in resampler.resample(frame):
                    for packet in output_stream.encode(resampled_frame):
                        output_container.mux(packet)
            for packet in output_stream.encode(None):
                output_container.mux(packet)
        finally:
            output_container.close()
            input_container.close()
    except Exception as exc:
        Path(wav_path).unlink(missing_ok=True)
        raise AudioDecodeError(f"Could not convert audio for diarization: {exc}") from exc
    return wav_path


class WhisperASR:
    """Thin wrapper around faster-whisper. Model is lazy-loaded on first use."""

    def __init__(
        self,
        model_size: Optional[str] = None,
        device: str = "cpu",
        compute_type: str = "int8",
    ):
        self._model_size = model_size or os.environ.get("WHISPER_MODEL", "base")
        self._device = device
        self._compute_type = compute_type
        self._model = None
        # beam_size=1 can smooth over or mis-hear disfluencies; 5 is more faithful.
        self._beam_size = int(os.environ.get("WHISPER_BEAM_SIZE", "5"))

    def _load(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(
                self._model_size, device=self._device, compute_type=self._compute_type
            )
        return self._model

    def _transcribe_raw(self, audio: AudioInput) -> Tuple[List[Segment], Optional[str]]:
        model = self._load()
        audio_path, tmp_path = resolve_audio_path(audio)
        try:
            try:
                raw_segments, info = model.transcribe(
                    audio_path, beam_size=self._beam_size, vad_filter=True
                )
                segments = [
                    Segment(start=s.start, end=s.end, text=s.text.strip()) for s in raw_segments
                ]
            except Exception as exc:
                raise AudioDecodeError(f"Could not decode audio: {exc}") from exc
            return segments, info.language
        finally:
            if tmp_path:
                Path(tmp_path).unlink(missing_ok=True)

    def transcribe(self, audio: AudioInput) -> Transcript:
        segments, language = self._transcribe_raw(audio)
        text = " ".join(s.text for s in segments).strip()
        return Transcript(text=text, language=language)

    def transcribe_segments(self, audio: AudioInput) -> Tuple[List[Segment], Optional[str]]:
        """Like transcribe(), but keeps each segment's start/end time instead of
        joining them into one string — needed to align text with speaker turns."""
        return self._transcribe_raw(audio)
