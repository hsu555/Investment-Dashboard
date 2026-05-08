"""Formatting helpers for Streamlit display."""

from __future__ import annotations


def fmt_currency(value: float | None, currency: str | None = None) -> str:
    if value is None or value != value:
        return "N/A"
    prefix = f"{currency} " if currency else ""
    return f"{prefix}{value:,.2f}"


def fmt_percent(value: float | None, digits: int = 2) -> str:
    if value is None or value != value:
        return "N/A"
    return f"{value * 100:,.{digits}f}%"


def fmt_signed(value: float | None, digits: int = 2) -> str:
    if value is None or value != value:
        return "N/A"
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:,.{digits}f}"
