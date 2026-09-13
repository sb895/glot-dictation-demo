import os
import tempfile

import pytest

from dictation.asr import AudioDecodeError, WhisperASR, convert_to_wav


def test_transcribe_raises_clean_error_on_invalid_audio():
    asr = WhisperASR(model_size="tiny")
    with pytest.raises(AudioDecodeError):
        asr.transcribe(b"this is not audio data")


def test_convert_to_wav_raises_clean_error_on_invalid_audio():
    fd, path = tempfile.mkstemp(suffix=".audio")
    try:
        os.write(fd, b"this is not audio data")
        os.close(fd)
        with pytest.raises(AudioDecodeError):
            convert_to_wav(path)
    finally:
        os.unlink(path)


def test_convert_to_wav_handles_webm_opus_like_a_browser_recording():
    """Regression test: browsers record audio/webm (Opus). faster-whisper's PyAV
    decoder reads that fine, but pyannote.audio's torchaudio/soundfile backend
    can't — confirmed via LibsndfileError: "Format not recognised" against a
    real deployment. convert_to_wav exists specifically to bridge that gap, so
    its output must be a WAV file that soundfile actually accepts."""
    av = pytest.importorskip("av")
    soundfile = pytest.importorskip("soundfile")

    fd, webm_path = tempfile.mkstemp(suffix=".webm")
    os.close(fd)
    try:
        input_container = av.open("tests/fixtures/sample_dictation.wav")
        output_container = av.open(webm_path, mode="w")
        output_stream = output_container.add_stream("libopus", rate=48000)
        resampler = av.AudioResampler(format="s16", layout="mono", rate=48000)
        for frame in input_container.decode(audio=0):
            for resampled_frame in resampler.resample(frame):
                for packet in output_stream.encode(resampled_frame):
                    output_container.mux(packet)
        for packet in output_stream.encode(None):
            output_container.mux(packet)
        output_container.close()
        input_container.close()

        wav_path = convert_to_wav(webm_path)
        try:
            with soundfile.SoundFile(wav_path) as f:
                assert f.frames > 0
        finally:
            os.unlink(wav_path)
    finally:
        os.unlink(webm_path)
