"""Data access helpers for Yahoo Finance via yfinance."""

from __future__ import annotations

import re
import contextlib
from dataclasses import dataclass
import io
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import yfinance.cache as yf_cache

from .config import CACHE_TTL_SECONDS, FX_TICKER, YAHOO_TW_RSS_FEEDS

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_HEADERS = {
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.6",
    "User-Agent": "Mozilla/5.0 Investment Dashboard",
}

YFINANCE_CACHE_DIR = Path(__file__).resolve().parents[1] / ".yfinance-cache"
try:
    YFINANCE_CACHE_DIR.mkdir(exist_ok=True)
    yf_cache.set_cache_location(str(YFINANCE_CACHE_DIR))
except Exception:
    pass


@dataclass(frozen=True)
class Quote:
    ticker: str
    price: float | None
    previous_close: float | None
    day_change: float | None
    day_change_pct: float | None
    currency: str | None
    long_name: str | None
    short_name: str | None
    dividend_yield: float | None
    trailing_annual_dividend_rate: float | None


@dataclass(frozen=True)
class SecurityProfile:
    ticker: str
    quote_type: str | None
    currency: str | None
    long_name: str | None
    short_name: str | None
    sector: str | None
    industry: str | None
    exchange: str | None
    market_cap: float | None
    total_assets: float | None
    trailing_pe: float | None
    forward_pe: float | None
    price_to_book: float | None
    dividend_yield: float | None
    trailing_eps: float | None
    forward_eps: float | None
    beta: float | None
    fifty_two_week_low: float | None
    fifty_two_week_high: float | None
    average_volume: float | None
    expense_ratio: float | None
    fund_family: str | None
    category: str | None
    nav_price: float | None
    summary: str | None


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


def _quiet_yfinance(callable_object):
    """Run noisy yfinance calls without letting Yahoo 401 messages leak to the terminal."""
    stdout = io.StringIO()
    stderr = io.StringIO()
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        return callable_object()


def _fast_info_snapshot(ticker: str) -> dict[str, object]:
    keys = [
        "currency",
        "last_price",
        "previous_close",
        "regular_market_previous_close",
        "market_cap",
        "year_low",
        "year_high",
        "three_month_average_volume",
        "ten_day_average_volume",
        "exchange",
        "quote_type",
    ]

    def load() -> dict[str, object]:
        info = yf.Ticker(ticker).fast_info
        return {key: _fast_info_value(info, key) for key in keys}

    return _quiet_yfinance(load)


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _info_value(info: dict[str, object], *keys: str) -> object:
    for key in keys:
        value = info.get(key)
        if value not in (None, ""):
            return value
    return None


def _quote_search_result(ticker: str) -> dict[str, object]:
    """Fetch search metadata without using Yahoo endpoints that require crumb auth."""
    try:
        response = requests.get(
            YAHOO_SEARCH_URL,
            params={"q": ticker, "quotesCount": 6, "newsCount": 0},
            timeout=8,
            headers=YAHOO_HEADERS,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
    except Exception:
        return {}

    if not isinstance(quotes, list):
        return {}

    ticker_upper = ticker.upper()
    exact_matches = [
        quote
        for quote in quotes
        if isinstance(quote, dict) and str(quote.get("symbol", "")).upper() == ticker_upper
    ]
    candidates = exact_matches or [quote for quote in quotes if isinstance(quote, dict)]
    if not candidates:
        return {}

    return candidates[0]


def _quote_name_from_search(ticker: str) -> tuple[str | None, str | None]:
    """Fetch display names without using Yahoo endpoints that require crumb auth."""
    selected = _quote_search_result(ticker)
    if not selected:
        return None, None

    long_name = _as_text(selected.get("longname") or selected.get("longName"))
    short_name = _as_text(selected.get("shortname") or selected.get("shortName"))
    return long_name, short_name


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
def load_ohlcv_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Load OHLCV history for one ticker."""
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    columns = [column for column in ["Open", "High", "Low", "Close", "Volume"] if column in raw]
    if not columns or "Close" not in columns:
        return pd.DataFrame()

    history = raw[columns].copy()
    history.index = pd.to_datetime(history.index)
    return history.sort_index().dropna(how="all")


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
            info = _fast_info_snapshot(ticker)
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

        long_name, short_name = _quote_name_from_search(ticker)
        quotes[ticker] = Quote(
            ticker=ticker,
            price=price,
            previous_close=previous_close,
            day_change=day_change,
            day_change_pct=day_change_pct,
            currency=str(_fast_info_value(info, "currency") or ""),
            long_name=long_name,
            short_name=short_name,
            dividend_yield=None,
            trailing_annual_dividend_rate=None,
        )
    return quotes


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def load_security_profile(ticker: str) -> SecurityProfile:
    """Load quote summary and fundamental fields for one stock or ETF."""
    search = _quote_search_result(ticker)
    try:
        fast_info = _fast_info_snapshot(ticker)
    except Exception:
        fast_info = {}

    # Avoid Ticker.info here: Yahoo's quoteSummary endpoint often returns 401
    # and yfinance prints those errors directly to the Streamlit terminal.
    info: dict[str, object] = {}

    long_name = _as_text(search.get("longname") or search.get("longName"))
    short_name = _as_text(search.get("shortname") or search.get("shortName"))
    long_name = long_name or _as_text(_info_value(info, "longName", "longBusinessSummary"))
    short_name = short_name or _as_text(_info_value(info, "shortName", "symbol"))
    currency = _as_text(_info_value(info, "currency", "financialCurrency")) or _as_text(_fast_info_value(fast_info, "currency"))
    exchange = (
        _as_text(_info_value(info, "fullExchangeName", "exchange"))
        or _as_text(search.get("exchDisp") or search.get("exchange"))
        or _as_text(_fast_info_value(fast_info, "exchange"))
    )

    return SecurityProfile(
        ticker=ticker,
        quote_type=_as_text(_info_value(info, "quoteType", "typeDisp")) or _as_text(search.get("quoteType") or search.get("typeDisp")),
        currency=currency,
        long_name=long_name,
        short_name=short_name,
        sector=_as_text(_info_value(info, "sector")) or _as_text(search.get("sectorDisp") or search.get("sector")),
        industry=_as_text(_info_value(info, "industry")) or _as_text(search.get("industryDisp") or search.get("industry")),
        exchange=exchange,
        market_cap=_as_float(_info_value(info, "marketCap")) or _as_float(_fast_info_value(fast_info, "market_cap")),
        total_assets=_as_float(_info_value(info, "totalAssets", "netAssets")),
        trailing_pe=_as_float(_info_value(info, "trailingPE")),
        forward_pe=_as_float(_info_value(info, "forwardPE")),
        price_to_book=_as_float(_info_value(info, "priceToBook")),
        dividend_yield=_as_float(_info_value(info, "dividendYield", "yield")),
        trailing_eps=_as_float(_info_value(info, "trailingEps")),
        forward_eps=_as_float(_info_value(info, "forwardEps")),
        beta=_as_float(_info_value(info, "beta")),
        fifty_two_week_low=_as_float(_info_value(info, "fiftyTwoWeekLow")) or _as_float(_fast_info_value(fast_info, "year_low")),
        fifty_two_week_high=_as_float(_info_value(info, "fiftyTwoWeekHigh")) or _as_float(_fast_info_value(fast_info, "year_high")),
        average_volume=(
            _as_float(_info_value(info, "averageVolume", "averageDailyVolume10Day"))
            or _as_float(_fast_info_value(fast_info, "three_month_average_volume"))
            or _as_float(_fast_info_value(fast_info, "ten_day_average_volume"))
        ),
        expense_ratio=_as_float(_info_value(info, "annualReportExpenseRatio", "expenseRatio", "netExpenseRatio")),
        fund_family=_as_text(_info_value(info, "fundFamily")),
        category=_as_text(_info_value(info, "category")),
        nav_price=_as_float(_info_value(info, "navPrice")),
        summary=_as_text(_info_value(info, "longBusinessSummary")),
    )


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
                headers=YAHOO_HEADERS,
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
