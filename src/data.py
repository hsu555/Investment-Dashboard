"""Data access helpers for Yahoo Finance via yfinance."""

from __future__ import annotations

import re
import contextlib
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import pandas as pd
import requests
import streamlit as st
import yfinance as yf
import yfinance.cache as yf_cache
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import (
    CACHE_TTL_HISTORY,
    CACHE_TTL_NEWS,
    CACHE_TTL_PROFILE,
    FX_TICKER,
    YAHOO_TW_RSS_FEEDS,
)

YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
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

TIMING_LOG = Path(__file__).resolve().parents[1] / "timing.log"
QUOTE_CACHE_FILE = Path(__file__).resolve().parents[1] / "quote_cache.json"


def clear_timing_log() -> None:
    """Call once at app startup to start a fresh session log."""
    try:
        TIMING_LOG.write_text(
            f"=== session {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n",
            encoding="utf-8",
        )
    except Exception:
        pass


def read_timing_log() -> str:
    try:
        return TIMING_LOG.read_text(encoding="utf-8")
    except Exception:
        return ""


@contextlib.contextmanager
def _timed(label: str):
    """Context manager that appends one timing line to timing.log."""
    t0 = time.perf_counter()
    meta: dict[str, object] = {}
    try:
        yield meta
    finally:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        size_str = str(meta.get("size", ""))
        line = f"  {datetime.datetime.now().strftime('%H:%M:%S')}  {label:<45}  {elapsed_ms:7.0f} ms"
        if size_str:
            line += f"  [{size_str}]"
        line += "\n"
        try:
            with TIMING_LOG.open("a", encoding="utf-8") as f:
                f.write(line)
        except Exception:
            pass

# 網路錯誤重試：最多 3 次，指數退避 2→4→8 秒
_NETWORK_RETRY = retry(
    retry=retry_if_exception_type((
        requests.exceptions.ConnectionError,
        requests.exceptions.Timeout,
        requests.exceptions.ChunkedEncodingError,
    )),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=8),
    reraise=True,
)


@_NETWORK_RETRY
def _http_get(url: str, **kwargs) -> requests.Response:
    """requests.get 加重試，由呼叫端處理 raise_for_status 及例外。"""
    return requests.get(url, **kwargs)


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


_FAST_INFO_KEYS = [
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


@_NETWORK_RETRY
def _fetch_fast_info(ticker: str) -> dict[str, object]:
    """yf.Ticker fast_info fetch with network retry (reraises on exhaustion)."""
    with _timed(f"fast_info  {ticker}") as meta:
        def load() -> dict[str, object]:
            info = yf.Ticker(ticker).fast_info
            return {key: _fast_info_value(info, key) for key in _FAST_INFO_KEYS}
        result = _quiet_yfinance(load)
        meta["size"] = f"{sum(1 for v in result.values() if v is not None)}/{len(_FAST_INFO_KEYS)} fields"
    return result


def _fast_info_snapshot(ticker: str) -> dict[str, object]:
    try:
        return _fetch_fast_info(ticker)
    except Exception:
        return {}


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _read_quote_cache() -> dict[str, object]:
    try:
        payload = json.loads(QUOTE_CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_quote_cache(payload: dict[str, object]) -> None:
    try:
        QUOTE_CACHE_FILE.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except OSError:
        pass


def _quote_from_cache(ticker: str, cached: object) -> Quote | None:
    if not isinstance(cached, dict):
        return None
    return Quote(
        ticker=ticker,
        price=_as_float(cached.get("price")),
        previous_close=_as_float(cached.get("previous_close")),
        day_change=_as_float(cached.get("day_change")),
        day_change_pct=_as_float(cached.get("day_change_pct")),
        currency=_as_text(cached.get("currency")) or _currency_from_ticker(ticker),
        long_name=_as_text(cached.get("long_name")),
        short_name=_as_text(cached.get("short_name")),
        dividend_yield=_as_float(cached.get("dividend_yield")),
        trailing_annual_dividend_rate=_as_float(cached.get("trailing_annual_dividend_rate")),
    )


def _quote_has_price(quote: Quote | None) -> bool:
    return quote is not None and quote.price is not None and quote.price == quote.price


def _info_value(info: dict[str, object], *keys: str) -> object:
    for key in keys:
        value = info.get(key)
        if value not in (None, ""):
            return value
    return None


def _quote_search_result(ticker: str) -> dict[str, object]:
    """Fetch search metadata without using Yahoo endpoints that require crumb auth."""
    with _timed(f"search     {ticker}") as meta:
        try:
            response = _http_get(
                YAHOO_SEARCH_URL,
                params={"q": ticker, "quotesCount": 6, "newsCount": 0},
                timeout=8,
                headers=YAHOO_HEADERS,
            )
            response.raise_for_status()
            quotes = response.json().get("quotes", [])
            meta["size"] = f"{len(response.content)} bytes"
        except Exception:
            meta["size"] = "error"
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


def _chart_quote_snapshot(ticker: str) -> dict[str, object]:
    """Fetch recent quote data from Yahoo's chart endpoint without crumb auth."""
    with _timed(f"chart quote {ticker}") as meta:
        try:
            response = _http_get(
                YAHOO_CHART_URL.format(ticker=ticker),
                params={"range": "5d", "interval": "1d"},
                timeout=8,
                headers=YAHOO_HEADERS,
            )
            response.raise_for_status()
            result = response.json().get("chart", {}).get("result", [])
            meta["size"] = f"{len(response.content)} bytes"
        except Exception:
            meta["size"] = "error"
            return {}

    if not result:
        return {}

    chart = result[0]
    chart_meta = chart.get("meta", {}) if isinstance(chart, dict) else {}
    indicators = chart.get("indicators", {}) if isinstance(chart, dict) else {}
    quotes = indicators.get("quote", []) if isinstance(indicators, dict) else []
    quote_values = quotes[0] if quotes and isinstance(quotes[0], dict) else {}
    closes = quote_values.get("close", []) if isinstance(quote_values, dict) else []
    valid_closes = [_as_float(value) for value in closes]
    valid_closes = [value for value in valid_closes if value is not None]

    last_price = _as_float(chart_meta.get("regularMarketPrice")) or (valid_closes[-1] if valid_closes else None)
    previous_close = _as_float(chart_meta.get("chartPreviousClose"))
    if previous_close is None and len(valid_closes) >= 2:
        previous_close = valid_closes[-2]

    return {
        "price": last_price,
        "previous_close": previous_close,
        "currency": _as_text(chart_meta.get("currency")),
        "long_name": _as_text(chart_meta.get("longName")),
        "short_name": _as_text(chart_meta.get("shortName")),
    }


def _clean_html(value: str) -> str:
    text = re.sub(r"<[^>]+>", "", value)
    text = text.replace("\xa0", " ")
    return " ".join(text.split())


@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def load_history(tickers: tuple[str, ...], period: str = "5y") -> pd.DataFrame:
    """Load adjusted close history for all tracked tickers."""
    if not tickers:
        return pd.DataFrame()

    with _timed(f"yf.download history {period} ×{len(tickers)}") as meta:
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
            meta["size"] = f"{raw.shape[0]} rows × {raw.shape[1]} cols" if not raw.empty else "empty"
        except Exception:
            meta["size"] = "error"
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


@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def load_ohlcv_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """Load OHLCV history for one ticker."""
    with _timed(f"yf.download ohlcv {ticker} {period}/{interval}") as meta:
        try:
            raw = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            meta["size"] = f"{raw.shape[0]} rows" if not raw.empty else "empty"
        except Exception:
            meta["size"] = "error"
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


@st.cache_data(ttl=CACHE_TTL_HISTORY, show_spinner=False)
def load_dividends(ticker: str) -> pd.Series:
    """Load dividend series for one ticker."""
    with _timed(f"dividends  {ticker}") as meta:
        try:
            dividends = yf.Ticker(ticker).dividends
            meta["size"] = f"{len(dividends)} records" if dividends is not None else "none"
        except Exception:
            meta["size"] = "error"
            return pd.Series(dtype=float)

    if dividends is None or dividends.empty:
        return pd.Series(dtype=float)

    dividends.index = pd.to_datetime(dividends.index).tz_localize(None)
    return dividends.sort_index()


def _currency_from_ticker(ticker: str) -> str:
    """Derive currency from ticker suffix — avoids a per-ticker fast_info call."""
    suffix_map = {
        ".TW": "TWD", ".TWO": "TWD",
        ".HK": "HKD",
        ".L": "GBP",
        ".DE": "EUR", ".PA": "EUR", ".AS": "EUR", ".MI": "EUR",
        ".T": "JPY",
    }
    for suffix, currency in suffix_map.items():
        if ticker.upper().endswith(suffix.upper()):
            return currency
    return "USD"


def load_quotes(tickers: tuple[str, ...]) -> dict[str, Quote]:
    """Load current quote metadata using one batch yf.download + parallel name searches."""
    # One batch request for all tickers' closing prices
    prices_last: dict[str, float | None] = {t: None for t in tickers}
    prices_prev: dict[str, float | None] = {t: None for t in tickers}
    with _timed(f"yf.download quotes 5d ×{len(tickers)}") as meta:
        try:
            raw = yf.download(
                list(tickers),
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
                group_by="column",
                threads=True,
            )
            meta["size"] = f"{raw.shape[0]} rows × {raw.shape[1]} cols" if not raw.empty else "empty"
        except Exception:
            meta["size"] = "error"
            raw = pd.DataFrame()

    if not raw.empty:
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw["Close"].copy() if "Close" in raw.columns.get_level_values(0) else pd.DataFrame()
        elif "Close" in raw.columns:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})
        else:
            close = pd.DataFrame()

        if not close.empty:
            close = close.sort_index().dropna(how="all")
            for ticker in tickers:
                col = ticker if ticker in close.columns else (close.columns[0] if len(close.columns) == 1 else None)
                if col is not None:
                    valid = close[col].dropna()
                    if len(valid) >= 1:
                        prices_last[ticker] = _as_float(valid.iloc[-1])
                    if len(valid) >= 2:
                        prices_prev[ticker] = _as_float(valid.iloc[-2])

    missing_price_tickers = [ticker for ticker in tickers if prices_last[ticker] is None]
    chart_snapshots: dict[str, dict[str, object]] = {}
    if missing_price_tickers:
        with ThreadPoolExecutor(max_workers=min(len(missing_price_tickers), 8)) as executor:
            futures = {ticker: executor.submit(_chart_quote_snapshot, ticker) for ticker in missing_price_tickers}
            chart_snapshots = {ticker: fut.result() for ticker, fut in futures.items()}
        for ticker, snapshot in chart_snapshots.items():
            prices_last[ticker] = _as_float(snapshot.get("price"))
            prices_prev[ticker] = _as_float(snapshot.get("previous_close"))

    # Fetch display names for all tickers in parallel
    with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
        name_futures = {ticker: executor.submit(_quote_name_from_search, ticker) for ticker in tickers}
        names = {ticker: fut.result() for ticker, fut in name_futures.items()}

    quotes: dict[str, Quote] = {}
    cache_payload = _read_quote_cache()
    cached_quotes = cache_payload.get("quotes", {})
    for ticker in tickers:
        last_price = prices_last[ticker]
        previous_close = prices_prev[ticker]
        chart_snapshot = chart_snapshots.get(ticker, {})
        long_name, short_name = names.get(ticker, (None, None))
        long_name = long_name or _as_text(chart_snapshot.get("long_name"))
        short_name = short_name or _as_text(chart_snapshot.get("short_name"))

        day_change = None
        day_change_pct = None
        if last_price is not None and previous_close not in (None, 0):
            day_change = last_price - previous_close
            day_change_pct = day_change / previous_close

        quotes[ticker] = Quote(
            ticker=ticker,
            price=last_price,
            previous_close=previous_close,
            day_change=day_change,
            day_change_pct=day_change_pct,
            currency=_as_text(chart_snapshot.get("currency")) or _currency_from_ticker(ticker),
            long_name=long_name,
            short_name=short_name,
            dividend_yield=None,
            trailing_annual_dividend_rate=None,
        )
        if not _quote_has_price(quotes[ticker]) and isinstance(cached_quotes, dict):
            cached_quote = _quote_from_cache(ticker, cached_quotes.get(ticker))
            if _quote_has_price(cached_quote):
                quotes[ticker] = cached_quote

    fresh_quotes = {ticker: quote for ticker, quote in quotes.items() if _quote_has_price(quote)}
    if fresh_quotes:
        cache_payload["quotes"] = {
            **(cached_quotes if isinstance(cached_quotes, dict) else {}),
            **{ticker: asdict(quote) for ticker, quote in fresh_quotes.items()},
        }
        cache_payload["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        _write_quote_cache(cache_payload)
    return quotes


@st.cache_data(ttl=CACHE_TTL_PROFILE, show_spinner=False)
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


def load_fx_rate() -> float | None:
    """Load USD/TWD exchange rate using Yahoo Finance TWD=X."""
    def cached_fx_rate() -> float | None:
        return _as_float(_read_quote_cache().get("fx_rate"))

    def remember_fx_rate(value: float | None) -> float | None:
        if value is None:
            return cached_fx_rate()
        payload = _read_quote_cache()
        payload["fx_rate"] = value
        payload["updated_at"] = datetime.datetime.now().isoformat(timespec="seconds")
        _write_quote_cache(payload)
        return value

    with _timed("yf.download fx TWD=X 5d") as meta:
        try:
            history = yf.download(
                FX_TICKER,
                period="5d",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            meta["size"] = f"{len(history)} rows" if not history.empty else "empty"
        except Exception:
            meta["size"] = "error"
            history = pd.DataFrame()

    if history.empty:
        snapshot = _chart_quote_snapshot(FX_TICKER)
        return remember_fx_rate(_as_float(snapshot.get("price")))

    if isinstance(history.columns, pd.MultiIndex):
        if "Close" not in history.columns.get_level_values(0):
            snapshot = _chart_quote_snapshot(FX_TICKER)
            return remember_fx_rate(_as_float(snapshot.get("price")))
        close = history["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
    else:
        if "Close" not in history:
            snapshot = _chart_quote_snapshot(FX_TICKER)
            return remember_fx_rate(_as_float(snapshot.get("price")))
        close = history["Close"]

    close = close.dropna()
    if close.empty:
        snapshot = _chart_quote_snapshot(FX_TICKER)
        return remember_fx_rate(_as_float(snapshot.get("price")))
    return remember_fx_rate(_as_float(close.iloc[-1]))


@st.cache_data(ttl=CACHE_TTL_NEWS, show_spinner=False)
def load_news(tickers: tuple[str, ...], limit: int = 8) -> list[dict[str, str]]:
    """Load Traditional Chinese Yahoo Taiwan Finance RSS news."""
    items: list[dict[str, str]] = []
    seen: set[str] = set()
    ticker_terms = {ticker.replace(".TW", "") for ticker in tickers}

    for category, url in YAHOO_TW_RSS_FEEDS.items():
        category_count = 0
        with _timed(f"rss        {category}") as meta:
            try:
                response = _http_get(
                    url,
                    timeout=10,
                    headers=YAHOO_HEADERS,
                )
                response.raise_for_status()
                root = ET.fromstring(response.content)
                meta["size"] = f"{len(response.content)} bytes"
            except Exception:
                meta["size"] = "error"
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
