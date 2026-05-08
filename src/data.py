"""Data access helpers for Yahoo Finance via yfinance."""

from __future__ import annotations

import re
from dataclasses import dataclass
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from .config import CACHE_TTL_SECONDS, FX_TICKER, YAHOO_TW_RSS_FEEDS


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float | None
    previous_close: float | None
    day_change: float | None
    day_change_pct: float | None
    currency: str | None
    long_name: str | None
    dividend_yield: float | None
    trailing_annual_dividend_rate: float | None


def _as_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _fast_info_value(info: object, key: str) -> object:
    try:
        return getattr(info, key)
    except Exception:
        pass
    try:
        return info.get(key)  # type: ignore[union-attr]
    except Exception:
        return None


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_history(tickers: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    """Load adjusted close history for all tracked tickers."""
    if not tickers:
        return pd.DataFrame()

    try:
        raw = yf.download(
            list(tickers),
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
            group_by="column",
            threads=True,
        )
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        else:
            return pd.DataFrame()
    else:
        prices = raw[["Close"]].rename(columns={"Close": tickers[0]})

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index().dropna(how="all")
    return prices


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_dividends(ticker: str) -> pd.Series:
    """Load dividend series for one ticker."""
    try:
        dividends = yf.Ticker(ticker).dividends
    except Exception:
        return pd.Series(dtype=float)

    if dividends is None or dividends.empty:
        return pd.Series(dtype=float)

    dividends.index = pd.to_datetime(dividends.index).tz_localize(None)
    return dividends.sort_index()


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_quotes(tickers: tuple[str, ...]) -> dict[str, Quote]:
    """Load current quote metadata for tickers."""
    quotes: dict[str, Quote] = {}
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).fast_info
        except Exception:
            info = {}

        price = _as_float(_fast_info_value(info, "last_price"))
        previous_close = _as_float(
            _fast_info_value(info, "previous_close")
            or _fast_info_value(info, "regular_market_previous_close")
        )

        day_change = None
        day_change_pct = None
        if price is not None and previous_close not in (None, 0):
            day_change = price - previous_close
            day_change_pct = day_change / previous_close

        quotes[ticker] = Quote(
            ticker=ticker,
            price=price,
            previous_close=previous_close,
            day_change=day_change,
            day_change_pct=day_change_pct,
            currency=str(_fast_info_value(info, "currency") or ""),
            long_name=None,
            dividend_yield=None,
            trailing_annual_dividend_rate=None,
        )
    return quotes


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_fx_rate() -> float | None:
    """Load USD/TWD exchange rate using Yahoo Finance TWD=X."""
    try:
        history = yf.download(
            FX_TICKER,
            period="5d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    except Exception:
        return None

    if history.empty:
        return None

    if isinstance(history.columns, pd.MultiIndex):
        if "Close" not in history.columns.get_level_values(0):
            return None
        close = history["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in history:
            return None
        close = history["Close"]

    close = close.dropna()
    if close.empty:
        return None
    return _as_float(close.iloc[-1])


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_news(tickers: tuple[str, ...], limit: int = 8) -> list[dict[str, str]]:
    """Load Traditional Chinese Yahoo Taiwan Finance RSS news."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    ticker_terms = {ticker.replace(".TW", "") for ticker in tickers}

    for category, url in YAHOO_TW_RSS_FEEDS.items():
        category_count = 0
        try:
            response = requests.get(
                url,
                timeout=10,
                headers={
                    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.6",
                    "User-Agent": "Mozilla/5.0 Investment Dashboard",
                },
            )
            response.raise_for_status()
            root = ET.fromstring(response.content)
        except Exception:
            continue

        for item in root.findall(".//item"):
            title = _clean_html(item.findtext("title", default="").strip())
            link = item.findtext("link", default="").strip()
            summary = _clean_html(item.findtext("description", default="").strip())
            matched_terms = sorted(term for term in ticker_terms if term and term in title + summary)

            if not title or title in seen:
                continue

            seen.add(title)
            items.append(
                {
                    "ticker": "、".join(matched_terms) if matched_terms else category,
                    "title": title,
                    "url": link,
                    "publisher": "Yahoo奇摩股市",
                    "summary": summary,
                }
            )
            category_count += 1
            if len(items) >= limit:
                return items
            if category_count >= 2:
                break
    return items
