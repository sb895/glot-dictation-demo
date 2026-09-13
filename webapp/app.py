"""Minimal web demo for the dictation pipeline: hold the button, speak, release."""

from __future__ import annotations

import os
import secrets
from functools import wraps

from flask import Flask, Response, jsonify, render_template, request

from dictation import Pipeline
from dictation.asr import AudioDecodeError

app = Flask(__name__)
_pipeline = Pipeline()  # Whisper model loads lazily on first request, not at import time

# Auth is opt-in — unset means no auth. Set both on any public deployment with real keys.
_BASIC_AUTH_USER = os.environ.get("BASIC_AUTH_USER")
_BASIC_AUTH_PASS = os.environ.get("BASIC_AUTH_PASS")


def require_auth(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not _BASIC_AUTH_USER:
            return view(*args, **kwargs)
        auth = request.authorization
        valid = (
            auth
            and secrets.compare_digest(auth.username, _BASIC_AUTH_USER)
            and secrets.compare_digest(auth.password, _BASIC_AUTH_PASS)
        )
        if not valid:
            return Response(
                "Authentication required", 401, {"WWW-Authenticate": 'Basic realm="Dictation demo"'}
            )
        return view(*args, **kwargs)

    return wrapped


@app.route("/")
@require_auth
def index():
    resp = Response(render_template("index.html"))
    # Avoid a stale cached copy of the page shadowing UI/JS changes after a redeploy.
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/api/transcribe", methods=["POST"])
@require_auth
def transcribe():
    audio_file = request.files.get("audio")
    if audio_file is None:
        return jsonify({"error": "no audio uploaded"}), 400

    audio_bytes = audio_file.read()

    context = request.form.get("context", "general")
    backend = request.form.get("backend", "default")
    diarize = request.form.get("diarize") == "1"
    try:
        result = _pipeline.run(audio_bytes, context=context, backend=backend, diarize=diarize)
    except AudioDecodeError:
        return jsonify({"error": "Could not process that recording, please try again."}), 400

    return jsonify(
        {
            "raw_text": result.raw_text,
            "text": result.text,
            "language": result.language,
            "polish_method": result.polish_method,
        }
    )


def main():
    # 0.0.0.0 + $PORT so this also works unmodified on Render/Railway/etc.
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)


if __name__ == "__main__":
    main()
