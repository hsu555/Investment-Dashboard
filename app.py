from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import hmac
import json
from html import escape
from pathlib import Path
import time

import pandas as pd
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from src.analytics import (
    growth_curve,
    metrics_table,
    technical_indicators,
    technical_signal_table,
    yearly_dividends,
)
from src.charts import (
    allocation_pie,
    comparison_chart,
    dividend_chart,
    growth_chart,
    momentum_chart,
    technical_price_chart,
)
from src.config import CACHE_TTL_NEWS, CACHE_TTL_QUOTES, DEFAULT_TICKERS
from src.data import (
    load_dividends,
    load_fx_rate,
    load_history,
    load_news,
    load_ohlcv_history,
    load_quotes,
    load_security_profile,
)
from src.formatting import fmt_currency, fmt_percent, fmt_signed
from src.names import ticker_display_name


PORTFOLIO_FILE = Path(__file__).with_name("portfolio.json")
PASSWORD_SECRET_KEY = "dashboard_password"
_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300  # 5分鐘鎖定


@st.cache_resource
def background_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)


@st.cache_resource
def _auth_store() -> dict:
    """Server-side 鎖定狀態，跨 session / 重新整理均有效。"""
    return {"lockout_until": 0.0, "failed_attempts": 0}


def parse_tickers(value: str) -> list[str]:
    separators = str.maketrans({",": " ", "，": " ", "\n": " "})
    tickers = [ticker.strip().upper() for ticker in value.translate(separators).split()]
    return list(dict.fromkeys(ticker for ticker in tickers if ticker))


def default_holdings() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order": range(1, len(DEFAULT_TICKERS) + 1),
            "ticker": DEFAULT_TICKERS,
            "quantity": [0.0] * len(DEFAULT_TICKERS),
            "purchase_price": [0.0] * len(DEFAULT_TICKERS),
        }
    )


def clean_holdings(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["order", "ticker", "quantity", "purchase_price"])

    cleaned = frame.copy()
    for column in ["order", "ticker", "quantity", "purchase_price"]:
        if column not in cleaned:
            cleaned[column] = "" if column == "ticker" else None

    cleaned["ticker"] = cleaned["ticker"].fillna("").astype(str).map(lambda value: value.strip().upper())
    cleaned = cleaned[cleaned["ticker"] != ""]
    cleaned["order"] = pd.to_numeric(cleaned["order"], errors="coerce")
    cleaned["quantity"] = pd.to_numeric(cleaned["quantity"], errors="coerce").fillna(0.0).clip(lower=0)
    cleaned["purchase_price"] = pd.to_numeric(cleaned["purchase_price"], errors="coerce").fillna(0.0).clip(lower=0)
    cleaned = cleaned.drop_duplicates(subset="ticker", keep="last")
    cleaned = cleaned.sort_values(["order", "ticker"], na_position="last").reset_index(drop=True)
    cleaned["order"] = range(1, len(cleaned) + 1)
    return cleaned[["order", "ticker", "quantity", "purchase_price"]]


def load_holdings() -> pd.DataFrame:
    try:
        payload = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_holdings()

    if not isinstance(payload, list):
        return default_holdings()
    return clean_holdings(pd.DataFrame(payload))


def save_holdings(holdings: pd.DataFrame) -> None:
    records = clean_holdings(holdings).to_dict(orient="records")
    PORTFOLIO_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def clear_sidebar_editor_state() -> None:
    prefixes = (
        "ticker_input_",
        "quantity_input_",
        "purchase_input_",
        "move_up_holding_",
        "move_down_holding_",
        "delete_holding_",
    )
    for key in list(st.session_state.keys()):
        if key.startswith(prefixes):
            del st.session_state[key]


def check_password() -> bool:
    if st.session_state.get("password_authenticated", False):
        return True

    st.title("投資儀表板")
    st.caption("請先輸入密碼，通過後才會載入持倉與投資資料。")

    # 鎖定檢查：使用 server-side store，重新整理無法繞過
    store = _auth_store()
    if store["lockout_until"] > time.time():
        remaining = int(store["lockout_until"] - time.time())
        st.error(f"登入嘗試次數過多，請等待 {remaining} 秒後再試。")
        return False

    with st.form("login_form"):
        password = st.text_input("密碼", type="password")
        submitted = st.form_submit_button("登入", type="primary")

    try:
        configured_password = st.secrets.get(PASSWORD_SECRET_KEY, "")
    except StreamlitSecretNotFoundError:
        configured_password = ""

    if submitted:
        if not configured_password:
            st.error(f"尚未設定登入密碼。請在 Streamlit Secrets 新增 `{PASSWORD_SECRET_KEY}`。")
            return False
        if hmac.compare_digest(password, str(configured_password)):
            st.session_state.password_authenticated = True
            store["failed_attempts"] = 0
            st.rerun()

        # 登入失敗：累計 server-side 次數
        store["failed_attempts"] += 1
        st.session_state.login_failed = True
        if store["failed_attempts"] >= _MAX_FAILED_ATTEMPTS:
            store["lockout_until"] = time.time() + _LOCKOUT_SECONDS
            store["failed_attempts"] = 0
            st.rerun()

    if st.session_state.get("login_failed", False):
        remaining_attempts = _MAX_FAILED_ATTEMPTS - store["failed_attempts"]
        st.error(f"密碼錯誤，無法存取儀表板。（還剩 {remaining_attempts} 次機會）")
    return False


def require_password() -> None:
    if not check_password():
        st.stop()


st.set_page_config(
    page_title="投資儀表板",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        color-scheme: dark;
    }
    .stApp {
        background: #0b1020;
        color: #e5e7eb;
    }
    section[data-testid="stSidebar"] {
        background: #111827;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 0.45rem;
    }
    div[data-testid="stSidebarUserContent"] {
        padding-top: 0.2rem !important;
    }
    div[data-testid="stSidebarHeader"] {
        height: 0 !important;
        min-height: 0 !important;
        padding: 0 !important;
    }
    div[data-testid="stSidebarHeader"] button {
        position: absolute;
        right: 0.4rem;
        top: 0.35rem;
        z-index: 2;
    }
    section[data-testid="stSidebar"] h1 {
        margin-top: 0;
        margin-bottom: 0.35rem;
    }
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(17, 24, 39, 0.48);
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlockBorderWrapper"] > div {
        padding: 5px 8px 6px;
    }
    section[data-testid="stSidebar"] div[data-testid="stVerticalBlock"] {
        gap: 0.42rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stHorizontalBlock"] {
        flex-wrap: nowrap !important;
        gap: 0 !important;
    }
    section[data-testid="stSidebar"] div[data-testid="column"] {
        flex-shrink: 1 !important;
        min-width: 0 !important;
        overflow: hidden;
        padding: 0 3px !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stTextInput"],
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] {
        min-width: 0;
    }
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input,
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        line-height: 1.2;
        min-height: 34px;
        padding: 2px 6px;
    }
    section[data-testid="stSidebar"] div[data-testid="stTextInput"] input {
        font-size: 0.95rem;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] input {
        font-size: 0.9rem;
    }
    section[data-testid="stSidebar"] div[data-baseweb="input"] {
        min-height: 34px;
    }
    section[data-testid="stSidebar"] div[data-baseweb="input"] > div {
        min-height: 34px;
    }
    section[data-testid="stSidebar"] div[data-testid="stNumberInput"] button {
        display: none;
    }
    section[data-testid="stSidebar"] button[kind="secondary"] {
        min-height: 34px;
        padding: 2px 8px;
    }
    section[data-testid="stSidebar"] button[kind="tertiary"] {
        font-size: 0.78rem;
        min-height: 28px;
        min-width: 20px;
        padding: 0 1px;
        white-space: nowrap;
    }
    .sidebar-row-guide {
        color: #94a3b8;
        display: grid;
        font-size: 0.76rem;
        gap: 0;
        grid-template-columns: 1.2fr 0.8fr 0.86fr;
        margin: 0 0 4px;
    }
    .sidebar-action-line {
        align-items: center;
        color: #cbd5e1;
        display: flex;
        font-size: 0.78rem;
        gap: 8px;
        line-height: 1.25;
        margin-top: 0;
        min-width: 0;
    }
    .sidebar-action-line .market {
        flex: 1 1 auto;
        font-size: 0.92rem;
        font-weight: 700;
        min-width: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .sidebar-action-line .positive {
        color: #22c55e;
    }
    .sidebar-action-line .negative {
        color: #fb7185;
    }
    .sidebar-action-line .muted {
        color: #94a3b8;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1560px;
    }
    .news-item {
        border-bottom: 1px solid rgba(148, 163, 184, 0.16);
        padding: 12px 0;
    }
    .news-item a {
        color: #93c5fd;
        text-decoration: none;
        font-weight: 650;
    }
    .news-meta {
        color: #94a3b8;
        font-size: 0.82rem;
        margin-top: 3px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def percent_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return df.map(lambda value: fmt_percent(value) if pd.notna(value) else "N/A")


def fmt_number(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):,.{digits}f}"


def fmt_compact(value: float | None, currency: str | None = None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    amount = float(value)
    abs_amount = abs(amount)
    units = [
        (1_000_000_000_000, "T"),
        (1_000_000_000, "B"),
        (1_000_000, "M"),
        (1_000, "K"),
    ]
    prefix = f"{currency} " if currency else ""
    for divisor, suffix in units:
        if abs_amount >= divisor:
            return f"{prefix}{amount / divisor:,.2f}{suffix}"
    return f"{prefix}{amount:,.2f}"


def value_or_fallback(*values):
    for value in values:
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return value
    return None


def add_display_name_column(df: pd.DataFrame, quotes) -> pd.DataFrame:
    display = df.copy()
    display.insert(
        0,
        "名稱",
        [ticker_display_name(str(ticker), quotes.get(str(ticker))) for ticker in display.index],
    )
    return display


def sidebar_market_summary_html(ticker: str, quote) -> str:
    if quote is None:
        return (
            '<div class="sidebar-action-line">'
            '<span class="market muted">行情更新中...</span>'
            "</div>"
        )

    currency = quote_currency(ticker, quote)
    price = fmt_currency(quote.price, currency)
    change = fmt_signed(quote.day_change)
    change_pct = fmt_percent(quote.day_change_pct)
    direction = "positive" if quote.day_change is not None and quote.day_change >= 0 else "negative"
    return (
        '<div class="sidebar-action-line">'
        f'<span class="market">{escape(price)} '
        f'<span class="{direction}">{escape(change)} ({escape(change_pct)})</span></span>'
        "</div>"
    )


def latest_indicator_value(indicators: pd.DataFrame, column: str) -> float | None:
    if indicators.empty or column not in indicators:
        return None
    values = indicators[column].dropna()
    return float(values.iloc[-1]) if not values.empty else None


def resample_ohlcv(history: pd.DataFrame, frequency: str) -> pd.DataFrame:
    if history.empty:
        return history

    columns = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    available = {column: method for column, method in columns.items() if column in history}
    return history.resample(frequency).agg(available).dropna(subset=["Close"])


def filter_prices_by_years(prices: pd.DataFrame, years: int) -> pd.DataFrame:
    if prices.empty:
        return prices
    end = prices.dropna(how="all").index.max()
    if pd.isna(end):
        return prices
    start = end - pd.DateOffset(years=years)
    return prices.loc[prices.index >= start]


def security_summary_signal(profile, quote, indicators: pd.DataFrame) -> str:
    clean = indicators.dropna(subset=["Close"]) if not indicators.empty else pd.DataFrame()
    latest = clean.iloc[-1] if not clean.empty else {}
    close = latest.get("Close") if hasattr(latest, "get") else None
    ma20 = latest.get("MA20") if hasattr(latest, "get") else None
    ma60 = latest.get("MA60") if hasattr(latest, "get") else None
    rsi = latest.get("RSI14") if hasattr(latest, "get") else None

    valuation = "估值資料不足"
    if profile.trailing_pe:
        valuation = "估值偏高" if profile.trailing_pe >= 30 else "估值合理" if profile.trailing_pe >= 12 else "估值偏低"
    elif profile.expense_ratio:
        valuation = f"費用率 {fmt_percent(profile.expense_ratio)}"

    trend = "趨勢資料不足"
    if pd.notna(close) and pd.notna(ma20) and pd.notna(ma60):
        trend = "短中期趨勢偏多" if close > ma20 > ma60 else "短中期趨勢偏弱" if close < ma20 < ma60 else "短中期盤整"

    momentum = "動能中性"
    if pd.notna(rsi):
        momentum = "RSI 偏熱" if rsi >= 70 else "RSI 偏弱" if rsi <= 30 else "動能中性"

    name = ticker_display_name(profile.ticker, quote)
    return f"{name} 目前呈現{trend}，{valuation}，{momentum}。"


def history_profile_stats(history: pd.DataFrame) -> dict[str, float | None]:
    if history.empty or "Close" not in history:
        return {"fifty_two_week_low": None, "fifty_two_week_high": None, "average_volume": None}

    recent = history.tail(252)
    stats = {
        "fifty_two_week_low": float(recent["Close"].min()) if not recent["Close"].dropna().empty else None,
        "fifty_two_week_high": float(recent["Close"].max()) if not recent["Close"].dropna().empty else None,
        "average_volume": None,
    }
    if "Volume" in recent and not recent["Volume"].dropna().empty:
        stats["average_volume"] = float(recent["Volume"].tail(60).mean())
    return stats


def profile_table(profile, quote=None, history_stats: dict[str, float | None] | None = None) -> pd.DataFrame:
    history_stats = history_stats or {}
    is_fund = (profile.quote_type or "").upper() in {"ETF", "MUTUALFUND"} or profile.expense_ratio is not None
    currency = value_or_fallback(profile.currency, getattr(quote, "currency", None), "TWD" if profile.ticker.endswith(".TW") else "USD")
    low_52w = value_or_fallback(profile.fifty_two_week_low, history_stats.get("fifty_two_week_low"))
    high_52w = value_or_fallback(profile.fifty_two_week_high, history_stats.get("fifty_two_week_high"))
    average_volume = value_or_fallback(profile.average_volume, history_stats.get("average_volume"))
    fields = [
        ("名稱", value_or_fallback(profile.long_name, profile.short_name)),
        ("代號", profile.ticker),
        ("類型", profile.quote_type),
        ("交易所", profile.exchange),
        ("幣別", currency),
        ("52週低點", fmt_currency(low_52w, currency)),
        ("52週高點", fmt_currency(high_52w, currency)),
        ("平均成交量", fmt_compact(average_volume)),
    ]
    if is_fund:
        fields.extend(
            [
                ("基金公司", profile.fund_family),
                ("分類", profile.category),
                ("總資產", fmt_compact(profile.total_assets, currency)),
                ("費用率", fmt_percent(profile.expense_ratio)),
                ("NAV", fmt_currency(profile.nav_price, currency)),
                ("配息率", fmt_percent(profile.dividend_yield)),
            ]
        )
    else:
        fields.extend(
            [
                ("產業", profile.sector),
                ("細分產業", profile.industry),
                ("市值", fmt_compact(profile.market_cap, currency)),
                ("Trailing P/E", fmt_number(profile.trailing_pe)),
                ("Forward P/E", fmt_number(profile.forward_pe)),
                ("P/B", fmt_number(profile.price_to_book)),
                ("EPS", fmt_number(profile.trailing_eps)),
                ("Beta", fmt_number(profile.beta)),
                ("配息率", fmt_percent(profile.dividend_yield)),
            ]
        )
    rows = [{"項目": label, "資料": value if value not in (None, "") else "N/A"} for label, value in fields]
    return pd.DataFrame(rows)


def format_signal_table(signals: pd.DataFrame) -> pd.DataFrame:
    if signals.empty:
        return signals
    formatted = signals.copy()
    formatted["數值"] = formatted["數值"].map(
        lambda value: fmt_percent(value) if pd.notna(value) and abs(float(value)) <= 2 else fmt_number(value)
    )
    return formatted


def render_security_analysis(
    selected: list[str],
    quotes,
    holdings_summary: pd.DataFrame,
    prefetched_security: dict[str, object] | None = None,
) -> None:
    st.subheader("個股 / ETF 分析")
    choices = selected
    default_ticker = selected[0] if selected else choices[0]
    cols = st.columns([0.35, 0.65])
    picked = cols[0].selectbox("選擇追蹤標的", choices, index=choices.index(default_ticker))
    manual = cols[1].text_input("或輸入 Yahoo Finance 代號", value=picked, placeholder="例如 AAPL、SPY、2330.TW")
    ticker = manual.strip().upper() or picked

    prefetched_ticker = str(prefetched_security.get("ticker")) if prefetched_security else None
    if prefetched_security and ticker == prefetched_ticker:
        profile = prefetched_security["profile"]
        quote = quotes.get(ticker)
        daily_history = prefetched_security["daily_history"]
    else:
        with st.spinner(f"正在載入 {ticker} 個股分析資料..."):
            profile = load_security_profile(ticker)
            quote_map = quotes if ticker in quotes else load_quotes((ticker,))
            quote = quote_map.get(ticker)
            daily_history = load_ohlcv_history(ticker, period="20y")

    if daily_history.empty:
        st.warning("目前無法取得這個標的的歷史價格，請確認代號是否符合 Yahoo Finance 格式。")
        return

    daily_indicators = technical_indicators(daily_history)
    latest_close = latest_indicator_value(daily_indicators, "Close")
    latest_rsi = latest_indicator_value(daily_indicators, "RSI14")
    latest_atr = latest_indicator_value(daily_indicators, "ATR14")
    history_stats = history_profile_stats(daily_history)
    low_52w = value_or_fallback(profile.fifty_two_week_low, history_stats.get("fifty_two_week_low"))
    high_52w = value_or_fallback(profile.fifty_two_week_high, history_stats.get("fifty_two_week_high"))
    currency = value_or_fallback((quote.currency if quote else None), profile.currency, "TWD" if ticker.endswith(".TW") else "USD")
    day_change = quote.day_change if quote else None
    day_change_pct = quote.day_change_pct if quote else None

    st.caption(security_summary_signal(profile, quote, daily_indicators))
    metric_cols = st.columns(6)
    metric_cols[0].metric("現價", fmt_currency(latest_close or (quote.price if quote else None), currency), fmt_signed(day_change) if day_change is not None else None)
    metric_cols[1].metric("日漲跌幅", fmt_percent(day_change_pct))
    metric_cols[2].metric("52 週區間", f"{fmt_number(low_52w)} - {fmt_number(high_52w)}")
    metric_cols[3].metric("市值 / 資產", fmt_compact(profile.market_cap or profile.total_assets, currency))
    metric_cols[4].metric("P/E 或費用率", fmt_number(profile.trailing_pe) if profile.trailing_pe else fmt_percent(profile.expense_ratio))
    metric_cols[5].metric("RSI / ATR", f"{fmt_number(latest_rsi)} / {fmt_number(latest_atr)}")

    position = holdings_summary.loc[[ticker]] if ticker in holdings_summary.index else pd.DataFrame()
    if not position.empty:
        row = position.iloc[0]
        st.info(
            f"此標的已在追蹤清單中：數量 {fmt_number(row['數量'], 0)}，"
            f"未實現損益 {fmt_currency(row['未實現損益(TWD)'], 'TWD')}，"
            f"配置比例 {fmt_percent(row['配置比例'])}。"
        )

    info_col, signal_col = st.columns([0.42, 0.58])
    with info_col:
        st.markdown("##### 基本面資料")
        st.dataframe(profile_table(profile, quote, history_stats), hide_index=True, width="stretch", height=400)
    with signal_col:
        st.markdown("##### 技術分析指標")
        signals = technical_signal_table(daily_indicators)
        st.dataframe(format_signal_table(signals), hide_index=True, width="stretch", height=360)

    chart_tabs = st.tabs(["日線", "週線", "月線", "動能"])
    periods = [
        ("日線", daily_indicators.tail(260)),
        ("週線", technical_indicators(resample_ohlcv(daily_history, "W-FRI")).tail(260)),
        ("月線", technical_indicators(resample_ohlcv(daily_history, "ME")).tail(240)),
    ]
    for tab, (label, indicator_frame) in zip(chart_tabs[:3], periods):
        with tab:
            st.plotly_chart(
                technical_price_chart(indicator_frame, ticker, f"{ticker} {label}價格、均線與量能"),
                width="stretch",
            )
    with chart_tabs[3]:
        st.plotly_chart(momentum_chart(daily_indicators.tail(260), f"{ticker} RSI / MACD"), width="stretch")

    if profile.summary:
        with st.expander("公司 / ETF 摘要"):
            st.write(profile.summary)


def render_sidebar() -> tuple[pd.DataFrame, dict, list[tuple[str, object]]]:
    st.sidebar.title("追蹤清單")
    if st.sidebar.button("登出", width="stretch"):
        st.session_state.password_authenticated = False
        st.session_state.login_failed = False
        st.rerun()

    if "holdings" not in st.session_state:
        st.session_state.holdings = load_holdings()
    if "latest_quotes" not in st.session_state:
        st.session_state.latest_quotes = {}

    sidebar_quotes = st.session_state.latest_quotes

    st.sidebar.caption("數量填 0 代表只觀察。每張卡片可編輯、排序，並顯示即時行情。")

    rows = []
    quote_slots = []
    pending_action = None
    current_holdings = clean_holdings(st.session_state.holdings)
    last_index = len(current_holdings) - 1
    for index, row in enumerate(current_holdings.itertuples(index=False)):
        row_key = str(row.ticker)
        with st.sidebar.container(border=True):
            st.markdown(
                """
                <div class="sidebar-row-guide">
                    <span>標的</span><span>數量</span><span>買入價</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            input_cols = st.columns([1.2, 0.8, 0.86], gap=None, vertical_alignment="center")
            ticker = input_cols[0].text_input(
                "標的",
                value=str(row.ticker),
                label_visibility="collapsed",
                key=f"ticker_input_{row_key}",
            )
            quantity = input_cols[1].number_input(
                "數量",
                min_value=0.0,
                value=float(row.quantity),
                step=1.0,
                format="%.0f",
                label_visibility="collapsed",
                key=f"quantity_input_{row_key}",
            )
            purchase_price = input_cols[2].number_input(
                "買入價",
                min_value=0.0,
                value=float(row.purchase_price),
                step=0.01,
                format="%.2f",
                label_visibility="collapsed",
                key=f"purchase_input_{row_key}",
            )

            clean_ticker = ticker.strip().upper()
            quote = sidebar_quotes.get(clean_ticker)

            action_cols = st.columns([1, 0.09, 0.09, 0.14], gap=None, vertical_alignment="center")
            quote_slot = action_cols[0].empty()
            quote_slot.markdown(sidebar_market_summary_html(clean_ticker, quote), unsafe_allow_html=True)
            quote_slots.append((clean_ticker, quote_slot))
            if action_cols[1].button(
                "↑",
                key=f"move_up_holding_{index}",
                help="上移",
                disabled=index == 0,
                type="tertiary",
                width="content",
            ):
                pending_action = ("up", index)
            if action_cols[2].button(
                "↓",
                key=f"move_down_holding_{index}",
                help="下移",
                disabled=index == last_index,
                type="tertiary",
                width="content",
            ):
                pending_action = ("down", index)
            if action_cols[3].button(
                "刪",
                key=f"delete_holding_{index}",
                help="刪除這個標的",
                type="tertiary",
                width="content",
            ):
                pending_action = ("delete", index)

            rows.append(
                {
                    "order": index + 1,
                    "ticker": clean_ticker,
                    "quantity": quantity,
                    "purchase_price": purchase_price,
                }
            )

    if pending_action is not None:
        action, target_index = pending_action
        if action == "delete":
            rows.pop(target_index)
        elif action == "up" and target_index > 0:
            rows[target_index - 1], rows[target_index] = rows[target_index], rows[target_index - 1]
        elif action == "down" and target_index < len(rows) - 1:
            rows[target_index + 1], rows[target_index] = rows[target_index], rows[target_index + 1]
        for order, item in enumerate(rows, start=1):
            item["order"] = order
        st.session_state.holdings = clean_holdings(pd.DataFrame(rows))
        clear_sidebar_editor_state()
        st.rerun()

    holdings = clean_holdings(pd.DataFrame(rows))
    st.session_state.holdings = holdings
    quotes = {
        ticker: sidebar_quotes[ticker]
        for ticker in holdings["ticker"].tolist()
        if ticker in sidebar_quotes
    }

    add_cols = st.sidebar.columns([0.68, 0.32])
    new_ticker = add_cols[0].text_input(
        "新增標的",
        value="",
        placeholder="例如 AAPL",
        label_visibility="collapsed",
        key="new_ticker_input",
    )
    if add_cols[1].button("新增", width="stretch"):
        candidate = new_ticker.strip().upper()
        if candidate:
            next_row = {
                "order": len(holdings) + 1,
                "ticker": candidate,
                "quantity": 0.0,
                "purchase_price": 0.0,
            }
            st.session_state.holdings = clean_holdings(pd.concat([holdings, pd.DataFrame([next_row])]))
            clear_sidebar_editor_state()
            st.rerun()

    if st.sidebar.button("儲存持倉", width="stretch"):
        save_holdings(holdings)
        st.sidebar.success("已儲存，下次開啟會自動載入。")

    st.sidebar.divider()
    st.sidebar.caption("價格資料來源：Yahoo Finance / yfinance。新聞來源：Yahoo奇摩股市。資料每次開啟頁面更新，並快取 30 分鐘。")
    return holdings, quotes, quote_slots


def quote_currency(ticker: str, quote) -> str:
    currency = (quote.currency if quote else None) or ("TWD" if ticker.endswith(".TW") else "USD")
    return currency.upper()


def twd_fx_rate(ticker: str, quote, fx_rate: float | None) -> float | None:
    currency = quote_currency(ticker, quote)
    if currency == "TWD":
        return 1.0
    if currency == "USD":
        return fx_rate
    return None


def portfolio_weights(holdings: pd.DataFrame, quotes, fx_rate: float | None) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in holdings.itertuples(index=False):
        quote = quotes.get(row.ticker)
        price = quote.price if quote else None
        conversion = twd_fx_rate(row.ticker, quote, fx_rate)
        values[row.ticker] = (
            float(row.quantity) * float(price) * conversion
            if price is not None and conversion is not None
            else 0.0
        )

    total = sum(values.values())
    if total <= 0:
        return {ticker: 0.0 for ticker in holdings["ticker"]}
    return {ticker: value / total for ticker, value in values.items()}


def render_holdings_summary(holdings: pd.DataFrame, quotes, fx_rate: float | None) -> pd.DataFrame:
    rows = []
    weights = portfolio_weights(holdings, quotes, fx_rate)
    for row in holdings.itertuples(index=False):
        quote = quotes.get(row.ticker)
        current_price = quote.price if quote else None
        currency = quote_currency(row.ticker, quote)
        conversion = twd_fx_rate(row.ticker, quote, fx_rate)
        quantity = float(row.quantity)
        purchase_price = float(row.purchase_price)
        market_value = (
            quantity * current_price * conversion
            if current_price is not None and conversion is not None
            else None
        )
        cost = (
            quantity * purchase_price * conversion
            if purchase_price > 0 and conversion is not None
            else None
        )
        unrealized_gain = market_value - cost if market_value is not None and cost is not None else None
        unrealized_return = unrealized_gain / cost if unrealized_gain is not None and cost not in (None, 0) else None
        rows.append(
            {
                "名稱": ticker_display_name(row.ticker, quote),
                "標的": row.ticker,
                "幣別": currency,
                "數量": quantity,
                "買入價": purchase_price if purchase_price > 0 else None,
                "現價": current_price,
                "匯率": conversion,
                "市值(TWD)": market_value,
                "成本(TWD)": cost,
                "未實現損益(TWD)": unrealized_gain,
                "未實現報酬率": unrealized_return,
                "配置比例": weights.get(row.ticker, 0.0),
            }
        )
    return pd.DataFrame(rows).set_index("標的") if rows else pd.DataFrame()


def render_position_metrics(holdings_summary: pd.DataFrame, fx_rate: float | None) -> None:
    if holdings_summary.empty:
        total_market_value = None
        total_cost = None
        total_gain = None
        total_return = None
    else:
        total_market_value = holdings_summary["市值(TWD)"].dropna().sum()
        total_cost = holdings_summary["成本(TWD)"].dropna().sum()
        total_gain = (
            total_market_value - total_cost
            if total_market_value > 0 and total_cost > 0
            else None
        )
        total_return = total_gain / total_cost if total_gain is not None and total_cost else None

    cols = st.columns(5)
    cols[0].metric("持有市值(TWD)", fmt_currency(float(total_market_value), "TWD") if total_market_value else "N/A")
    cols[1].metric("投入成本(TWD)", fmt_currency(float(total_cost), "TWD") if total_cost else "N/A")
    cols[2].metric("未實現損益(TWD)", fmt_currency(float(total_gain), "TWD") if total_gain is not None else "N/A")
    cols[3].metric("未實現報酬率", fmt_percent(float(total_return)) if total_return is not None else "N/A")
    cols[4].metric("USD/TWD", fmt_currency(fx_rate, "TWD"))


def update_sidebar_quote_slots(quote_slots: list[tuple[str, object]], quotes) -> None:
    for ticker, slot in quote_slots:
        slot.markdown(sidebar_market_summary_html(ticker, quotes.get(ticker)), unsafe_allow_html=True)


def render_dividend_summary(selected: list[str], quotes, dividends: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for ticker in selected:
        quote = quotes.get(ticker)
        currency = quote_currency(ticker, quote)
        series = dividends.get(ticker, pd.Series(dtype=float))
        latest = series.iloc[-1] if not series.empty else None
        latest_date = series.index[-1].date().isoformat() if not series.empty else "N/A"
        trailing_start = pd.Timestamp.today().tz_localize(None) - pd.Timedelta(days=365)
        trailing_dividend = (
            float(series.loc[series.index >= trailing_start].sum()) if not series.empty else None
        )
        dividend_yield = (
            trailing_dividend / quote.price
            if quote and quote.price not in (None, 0) and trailing_dividend is not None
            else None
        )
        rows.append(
            {
                "名稱": ticker_display_name(ticker, quote),
                "標的": ticker,
                "幣別": currency,
                "Trailing Annual Dividend": trailing_dividend,
                "Dividend Yield": dividend_yield,
                "Latest Dividend": latest,
                "Latest Date": latest_date,
            }
        )

    return pd.DataFrame(rows).set_index("標的")


def render_news(news: list[dict[str, str]]) -> None:
    st.subheader("財經新聞摘要")
    if not news:
        st.info("目前沒有抓到 Yahoo奇摩股市新聞。")
        return

    for item in news:
        summary = item["summary"][:180] + "..." if len(item["summary"]) > 180 else item["summary"]
        title = item["title"]
        url = item["url"]
        link = f'<a href="{url}" target="_blank">{title}</a>' if url else title
        st.markdown(
            f"""
            <div class="news-item">
                {link}
                <div class="news-meta">{item["ticker"]} · {item["publisher"]}</div>
                <div style="margin-top:6px; color:#cbd5e1;">{summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def build_observation_data(tickers: tuple[str, ...]) -> dict[str, object]:
    prices = load_history(tickers, period="20y")
    metrics = metrics_table(prices) if not prices.empty else pd.DataFrame()
    return {"prices": prices, "metrics": metrics}


def build_dividend_data(tickers: tuple[str, ...]) -> dict[str, object]:
    dividends = {ticker: load_dividends(ticker) for ticker in tickers}
    return {"dividends": dividends, "annual_dividends": yearly_dividends(dividends)}


def build_security_data(ticker: str) -> dict[str, object]:
    return {
        "ticker": ticker,
        "profile": load_security_profile(ticker),
        "daily_history": load_ohlcv_history(ticker, period="20y"),
    }


def ensure_prefetch_jobs(tickers: tuple[str, ...]) -> dict[str, Future]:
    cache_window = int(time.time() // CACHE_TTL_NEWS)
    prefetch_key = f"{cache_window}|{'|'.join(tickers)}"
    if st.session_state.get("prefetch_key") != prefetch_key:
        st.session_state.prefetch_key = prefetch_key
        st.session_state.prefetch_jobs = {}

    jobs = st.session_state.prefetch_jobs
    if "observation" not in jobs:
        jobs["observation"] = background_executor().submit(build_observation_data, tickers)
    if "dividends" not in jobs:
        jobs["dividends"] = background_executor().submit(build_dividend_data, tickers)
    if "news" not in jobs:
        jobs["news"] = background_executor().submit(load_news, tickers)
    if tickers and "security" not in jobs:
        jobs["security"] = background_executor().submit(build_security_data, tickers[0])
    return jobs


def wait_for_prefetch(jobs: dict[str, Future], name: str, label: str):
    future = jobs.get(name)
    if future is None:
        return None
    if future.done():
        try:
            return future.result()
        except Exception:
            return None

    with st.status(f"{label}仍在背景載入...", expanded=True) as status:
        status.write("首屏已先顯示，這裡接續等待同一個背景工作完成。")
        try:
            result = future.result()
            status.update(label=f"{label}已載入", state="complete", expanded=False)
            return result
        except Exception:
            status.update(label=f"{label}背景載入未完成，改由目前頁面載入", state="error", expanded=False)
            return None


def prefetch_status_caption(jobs: dict[str, Future]) -> str:
    labels = {
        "observation": "觀察指標",
        "security": "個股分析",
        "dividends": "配息資訊",
        "news": "新聞摘要",
    }
    pending = [label for key, label in labels.items() if key in jobs and not jobs[key].done()]
    if not pending:
        return "其他檢視資料已在背景預載完成。"
    return f"正在背景預載：{'、'.join(pending)}。你可以先查看持有資產。"


def main() -> None:
    require_password()

    holdings, quotes, quote_slots = render_sidebar()
    selected = holdings["ticker"].tolist()
    if not selected:
        st.warning("請至少選擇一個追蹤標的。")
        return

    tickers = tuple(selected)
    st.title("投資儀表板")
    st.caption("即時價格與買入價保留原幣別；市值、成本、損益與配置比例統一換算為台幣。新聞取自 Yahoo奇摩股市，快取時間：30 分鐘。")

    load_status = st.status("正在準備投資儀表板...", expanded=True)
    load_status.write("更新追蹤清單即時報價與日漲跌。")
    quotes = load_quotes(tickers)
    st.session_state.latest_quotes = quotes
    update_sidebar_quote_slots(quote_slots, quotes)
    load_status.write("取得 USD/TWD 匯率，換算台幣市值與損益。")
    fx_rate = load_fx_rate()
    load_status.update(label="核心資料已載入", state="complete", expanded=False)

    weights = portfolio_weights(holdings, quotes, fx_rate)
    holdings_summary = render_holdings_summary(holdings, quotes, fx_rate)
    held_tickers = holdings.loc[holdings["quantity"] > 0, "ticker"].tolist()
    held_summary = holdings_summary.reindex(held_tickers).dropna(how="all") if held_tickers else pd.DataFrame()

    render_position_metrics(held_summary, fx_rate)

    active_view = st.radio(
        "檢視",
        ["持有資產", "觀察指標", "個股分析", "配息資訊", "新聞摘要"],
        horizontal=True,
        label_visibility="collapsed",
    )
    prefetch_jobs = ensure_prefetch_jobs(tickers)

    if active_view == "持有資產":
        if held_summary.empty:
            st.info("目前沒有數量大於 0 的持有標的。")
        else:
            st.plotly_chart(allocation_pie(weights), width="stretch")
            formatted_holdings = held_summary.copy()
            formatted_holdings["買入價"] = formatted_holdings.apply(
                lambda row: fmt_currency(row["買入價"], row["幣別"]),
                axis=1,
            )
            formatted_holdings["現價"] = formatted_holdings.apply(
                lambda row: fmt_currency(row["現價"], row["幣別"]),
                axis=1,
            )
            formatted_holdings["匯率"] = formatted_holdings["匯率"].map(
                lambda value: f"{value:,.4f}" if pd.notna(value) else "N/A"
            )
            for column in ["市值(TWD)", "成本(TWD)", "未實現損益(TWD)"]:
                formatted_holdings[column] = formatted_holdings[column].map(lambda value: fmt_currency(value, "TWD"))
            for column in ["未實現報酬率", "配置比例"]:
                formatted_holdings[column] = formatted_holdings[column].map(fmt_percent)
            st.dataframe(formatted_holdings, width="stretch", height=320)
        st.caption(prefetch_status_caption(prefetch_jobs))

    elif active_view == "觀察指標":
        observation_data = wait_for_prefetch(prefetch_jobs, "observation", "觀察指標資料")
        if observation_data is None:
            with st.status("正在載入追蹤標的歷史價格...", expanded=True) as history_status:
                history_status.write("讀取 20 年歷史價格，用於成長曲線、CAGR 與波動比較。")
                prices = load_history(tickers, period="20y")
                observation_metrics = metrics_table(prices) if not prices.empty else pd.DataFrame()
                history_status.update(label="追蹤標的歷史價格已載入", state="complete", expanded=False)
        else:
            prices = observation_data["prices"]
            observation_metrics = observation_data["metrics"]
        if prices.empty:
            st.error("目前無法取得價格歷史資料，請稍後重新整理。")
            return

        growth_window = st.radio(
            "成長曲線期間",
            options=["1年", "5年", "20年"],
            index=0,
            horizontal=True,
        )
        growth_years = {"1年": 1, "5年": 5, "20年": 20}[growth_window]
        observation_prices = filter_prices_by_years(prices, growth_years)
        observation_growth = growth_curve(observation_prices)
        observation_window_metrics = metrics_table(observation_prices)
        st.plotly_chart(
            growth_chart(observation_growth, observation_window_metrics, title=f"成長曲線：近 {growth_window}"),
            width="stretch",
        )
        st.plotly_chart(comparison_chart(observation_metrics), width="stretch")
        st.subheader("追蹤標的歷史指標")
        display_metrics = observation_metrics.copy()
        st.dataframe(
            add_display_name_column(percent_dataframe(display_metrics), quotes),
            width="stretch",
            height=300,
        )

    elif active_view == "個股分析":
        security_data = wait_for_prefetch(prefetch_jobs, "security", "個股分析資料")
        render_security_analysis(selected, quotes, holdings_summary, security_data)

    elif active_view == "配息資訊":
        dividend_data = wait_for_prefetch(prefetch_jobs, "dividends", "配息資料")
        if dividend_data is None:
            with st.status("正在載入配息資料...", expanded=True) as dividend_status:
                dividends = {}
                for ticker in selected:
                    dividend_status.write(f"讀取 {ticker} 配息紀錄。")
                    dividends[ticker] = load_dividends(ticker)
                annual_dividends = yearly_dividends(dividends)
                dividend_status.update(label="配息資料已載入", state="complete", expanded=False)
        else:
            dividends = dividend_data["dividends"]
            annual_dividends = dividend_data["annual_dividends"]

        summary = render_dividend_summary(selected, quotes, dividends)
        formatted = summary.copy()
        formatted["Dividend Yield"] = formatted["Dividend Yield"].map(fmt_percent)
        for column in ["Trailing Annual Dividend", "Latest Dividend"]:
            formatted[column] = formatted.apply(
                lambda row: fmt_currency(row[column], row["幣別"]),
                axis=1,
            )
        st.dataframe(formatted, width="stretch", height=260)

        if annual_dividends.empty:
            st.info("目前沒有可用的配息歷史資料。")
        else:
            st.plotly_chart(dividend_chart(annual_dividends), width="stretch")

    elif active_view == "新聞摘要":
        news = wait_for_prefetch(prefetch_jobs, "news", "新聞資料")
        if news is None:
            with st.status("正在抓取 Yahoo奇摩股市新聞...", expanded=True) as news_status:
                news_status.write("讀取 RSS 分類並比對追蹤標的關鍵字。")
                news = load_news(tickers)
                news_status.update(label="新聞資料已載入", state="complete", expanded=False)
        render_news(news)


if __name__ == "__main__":
    main()
