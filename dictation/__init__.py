"""Reverse-engineered Wispr Flow style dictation pipeline: audio in, polished text out."""

from .pipeline import DictationResult, Pipeline

__all__ = ["Pipeline", "DictationResult"]
__version__ = "0.1.0"
