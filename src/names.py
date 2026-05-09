"""Ticker display-name helpers."""

from __future__ import annotations

from .config import TICKER_DISPLAY_NAMES


def clean_display_name(value: object) -> str | None:
    """Return a stripped non-empty display name."""
    if value is None:
        return None

    text = str(value).strip()
    return text or None


def ticker_display_name(ticker: str, quote: object | None = None) -> str:
    """Resolve a readable display name with ticker as the final fallback."""
    mapped = clean_display_name(TICKER_DISPLAY_NAMES.get(ticker))
    if mapped:
        return mapped

    for attribute in ("long_name", "short_name", "display_name"):
        name = clean_display_name(getattr(quote, attribute, None))
        if name:
            return name

    return ticker
