from __future__ import annotations

from html import escape

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src.formatting import fmt_currency, fmt_percent, fmt_signed
from src.names import ticker_display_name


def configure_page() -> None:
    st.set_page_config(
        page_title="投資儀表板",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    components.html(
        """
        <script>
        document.documentElement.lang = "zh-TW";
        document.documentElement.classList.add("notranslate");
        const meta = document.createElement("meta");
        meta.name = "google";
        meta.content = "notranslate";
        document.head.appendChild(meta);
        </script>
        """,
        height=0,
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


def quote_currency(ticker: str, quote) -> str:
    currency = (quote.currency if quote else None) or ("TWD" if ticker.endswith(".TW") else "USD")
    return currency.upper()


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
