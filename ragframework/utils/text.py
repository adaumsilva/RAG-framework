"""Lightweight text utilities used across the framework."""

from __future__ import annotations

import re


def normalize_whitespace(text: str) -> str:
    """Replace sequences of whitespace with a single space and strip ends."""
    return re.sub(r"\s+", " ", text).strip()


def truncate(text: str, max_chars: int, suffix: str = "...") -> str:
    """Truncate *text* to *max_chars*, appending *suffix* if truncated."""
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def word_count(text: str) -> int:
    """Return the number of whitespace-separated tokens in *text*."""
    return len(text.split())
