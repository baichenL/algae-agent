from __future__ import annotations

import re
import unicodedata

from app.services.intent.routing_models import NormalizedInput


_PUNCTUATION_TRANSLATION = str.maketrans(
    {
        "，": ",",
        "；": ";",
        "：": ":",
        "。": ".",
        "、": ",",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
    }
)


def normalize_text_value(value: str) -> str:
    """Return the conservative, lossless-for-matching text view."""
    normalized = unicodedata.normalize("NFKC", value or "")
    normalized = normalized.translate(_PUNCTUATION_TRANSLATION)
    return re.sub(r"\s+", " ", normalized).strip()


def normalize_input(value: str) -> NormalizedInput:
    original = value or ""
    normalized = normalize_text_value(original)
    return NormalizedInput(
        original_text=original,
        normalized_text=normalized,
        match_text=normalized.casefold(),
    )


def normalized_contains(text: NormalizedInput, keyword: str) -> bool:
    return normalize_text_value(keyword).casefold() in text.match_text
