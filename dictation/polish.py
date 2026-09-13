"""Turns a raw ASR transcript into clean, paste-ready text.

This is the part that makes the output feel authored rather than transcribed:
filler-word removal, self-correction resolution, punctuation/paragraphing, and
light context-aware formatting.

Two paths:
  - LLM path (used when a backend is configured and reachable): a single
    prompt handles all of the above, including the parts a regex can't do
    (resolving "call John, actually Sarah" requires understanding, not
    pattern matching).
  - Regex fallback (used when no backend is available): strips common filler
    words and normalizes punctuation/casing. It cannot resolve self-corrections
    or restructure sentences — that's the real quality difference between the
    two modes, and is called out in the README rather than hidden.
"""

from __future__ import annotations

import logging
import re
from typing import Tuple

from . import llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You clean up raw speech-to-text transcripts so they read like the \
speaker typed them deliberately. Rules:
- Remove filler words and disfluencies (um, uh, like, you know, so yeah) unless removing \
them would change the meaning.
- Resolve self-corrections: if the speaker restates or corrects themselves mid-sentence \
(e.g. "call John — actually, call Sarah instead"), keep only the final intended meaning.
- Add correct punctuation, capitalization, and paragraph breaks. Split run-on dictation \
into sentences where a pause or topic shift is implied.
- Never invent facts, names, or content that was not said. Do not answer questions in the \
transcript or add commentary — only reformat what was said.
- Preserve the speaker's meaning, tone, and register; do not make it more formal than intended.
- Output only the cleaned text, nothing else: no preamble, no quotes, no explanation."""

DIARIZE_INFER_INSTRUCTION = """
This recording may contain more than one speaker. Identify distinct speakers from \
turn-taking and phrasing (e.g. a question followed by an answer, or a shift in \
perspective) and label them generically as "Person 1", "Person 2", etc., in order of \
first appearance — never guess real names. Format the entire output as a dialogue: one \
line per speaker turn, each starting with "Person N: " followed by that turn's cleaned \
text. If you cannot confidently identify more than one speaker, output the whole thing \
as a single "Person 1: " line."""

# Used instead of DIARIZE_INFER_INSTRUCTION once real diarization already split turns.
DIARIZE_PRESERVE_INSTRUCTION = """
This transcript has already been split into speaker turns using real audio-based \
speaker detection (not a guess), with each line starting "Person N: ". Preserve this \
exact speaker structure and labels exactly as given — do not merge, reorder, split, \
add, or relabel any turns, even if the content seems to suggest otherwise. Within each \
line, apply the same cleanup rules as above. Output the same "Person N: " structure, \
one line per input line, nothing else."""

CONTEXT_HINTS = {
    "general": "",
    "email": "Format as a short email body. Only add a greeting/sign-off if the speaker said one.",
    "chat": "Keep it casual and short, like a chat message. Do not over-formalize.",
    "notes": "Format as concise notes. Use short bullet points if multiple distinct points were dictated.",
    "code_comment": "Format as a terse code comment: one or two sentences, no filler, imperative mood.",
}

# Offline fallback only — the LLM path handles this contextually instead.
_FILLER_RE = re.compile(
    r"\b(um+|uh+|erm+|like|you know|i mean|sort of|kind of|basically)\b,?\s*",
    re.IGNORECASE,
)


def _is_quota_error(exc: Exception) -> bool:
    """Best-effort, cross-provider check for "you're rate-limited/out of quota".

    OpenAI and Gemini each raise their own exception types for this, and
    matching both by type would mean importing each SDK's error classes into
    this module just to check one thing. Checking the
    stringified error for the markers each of them actually uses (verified
    against a real Gemini 429 in testing: "429 RESOURCE_EXHAUSTED... quota")
    is less elegant but doesn't couple this module to three vendors' SDKs.
    """
    text = str(exc).lower()
    return any(marker in text for marker in ("429", "resource_exhausted", "rate_limit", "quota"))


def regex_cleanup(text: str) -> str:
    """Cheap, offline filler-word strip + punctuation/casing tidy-up."""
    cleaned = _FILLER_RE.sub("", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        if cleaned[-1] not in ".!?":
            cleaned += "."
    return cleaned


def _regex_cleanup_dialogue(raw_text: str) -> str:
    """regex_cleanup, applied per-line to a "Person N: ..." formatted transcript
    so the (already-correct, audio-based) speaker split survives the offline
    fallback instead of being flattened away."""
    lines = []
    for line in raw_text.split("\n"):
        label, sep, text = line.partition(": ")
        lines.append(f"{label}{sep}{regex_cleanup(text)}" if sep else regex_cleanup(line))
    return "\n".join(lines)


def polish(
    raw_text: str,
    context: str = "general",
    backend: str = "default",
    diarize: bool = False,
    prediarized: bool = False,
) -> Tuple[str, str]:
    """Returns (polished_text, method).

    method is "llm:<backend>" (e.g. "llm:gemini", "llm:local") on success,
    "regex_fallback" if the backend is unavailable (no key, package, or
    service running), or "regex_fallback_quota" specifically when the
    provider rejected the call for being rate-limited/out of quota — the
    offline cleanup still runs either way so the pipeline never just fails,
    but the UI shows a distinct message for "you've hit today's limit" vs.
    a generic failure.

    diarize labels speaker turns as "Person 1"/"Person 2"/etc. prediarized
    distinguishes *how*: when True, raw_text already has real audio-based
    speaker boundaries (see dictation/diarize.py) and the LLM's job is only to
    clean up the text within each given turn, not decide who's talking. When
    False (diarize alone), the LLM has to infer turns from the text itself —
    used when real diarization isn't available (no HF_TOKEN, package missing,
    gated model access not granted). The regex fallback can't infer speakers
    at all: with prediarized text it cleans each line in place, keeping the
    real speaker split; otherwise it just labels everything "Person 1".
    """
    if not raw_text.strip():
        return "", "regex_fallback"

    hint = CONTEXT_HINTS.get(context, "")
    user_prompt = f"[Context: {hint}]\n\n{raw_text}" if hint else raw_text

    if diarize:
        instruction = DIARIZE_PRESERVE_INSTRUCTION if prediarized else DIARIZE_INFER_INSTRUCTION
        system_prompt = SYSTEM_PROMPT + "\n" + instruction
    else:
        system_prompt = SYSTEM_PROMPT

    try:
        cleaned = llm.call(system=system_prompt, user=user_prompt, backend=backend)
        return cleaned, f"llm:{llm.resolve_backend(backend)}"
    except Exception as exc:
        logger.warning("polish backend '%s' failed, using regex fallback: %s", backend, exc)
        method = "regex_fallback_quota" if _is_quota_error(exc) else "regex_fallback"
        if prediarized:
            return _regex_cleanup_dialogue(raw_text), method
        cleaned = regex_cleanup(raw_text)
        return (f"Person 1: {cleaned}" if diarize else cleaned), method
