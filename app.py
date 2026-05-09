from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd
import streamlit as st

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
from src.config import DEFAULT_TICKERS
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
    div[data-testid="stMetric"] {
        background: #111827;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        padding: 14px 16px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.55rem;
    }
    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2rem;
        max-width: 1560px;
    }
    .ticker-card {
        padding: 10px 12px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 8px;
        background: rgba(17, 24, 39, 0.82);
        margin-bottom: 8px;
    }
    .ticker-card strong {
        font-size: 0.95rem;
    }
    .ticker-card span {
        color: #94a3b8;
        font-size: 0.8rem;
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
) -> None:
    st.subheader("個股 / ETF 分析")
    choices = list(dict.fromkeys(selected + ["AAPL", "TSLA", "SPY", "QQQ", "0050.TW", "2330.TW"]))
    default_ticker = selected[0] if selected else choices[0]
    cols = st.columns([0.35, 0.65])
    picked = cols[0].selectbox("選擇追蹤標的", choices, index=choices.index(default_ticker))
    manual = cols[1].text_input("或輸入 Yahoo Finance 代號", value=picked, placeholder="例如 AAPL、SPY、2330.TW")
    ticker = manual.strip().upper() or picked

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


def render_sidebar() -> pd.DataFrame:
    st.sidebar.title("追蹤清單")
    if "holdings" not in st.session_state:
        st.session_state.holdings = load_holdings()

    st.sidebar.caption("數量填 0 代表只觀察，不列入持有資產。買入價請輸入該標的原幣別價格。")
    edited = st.sidebar.data_editor(
        st.session_state.holdings,
        column_config={
            "order": st.column_config.NumberColumn("序號", min_value=1, step=1, format="%d"),
            "ticker": st.column_config.TextColumn("標的代號", help="Yahoo Finance 代號，例如 AAPL、TSLA、2330.TW。"),
            "quantity": st.column_config.NumberColumn("數量", min_value=0, step=1, format="%d"),
            "purchase_price": st.column_config.NumberColumn("買入價(原幣)", min_value=0.0, step=0.01, format="%.2f"),
        },
        hide_index=True,
        num_rows="dynamic",
        width="stretch",
        key="holdings_editor",
    )

    holdings = clean_holdings(edited)
    st.session_state.holdings = holdings

    if st.sidebar.button("儲存持倉", width="stretch"):
        save_holdings(holdings)
        st.sidebar.success("已儲存，下次開啟會自動載入。")

    st.sidebar.divider()
    st.sidebar.caption("價格資料來源：Yahoo Finance / yfinance。新聞來源：Yahoo奇摩股市。資料每次開啟頁面更新，並快取 5 分鐘。")
    return holdings


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


def render_ticker_cards(selected: list[str], quotes) -> None:
    st.sidebar.subheader("即時價格")
    for ticker in selected:
        quote = quotes.get(ticker)
        name = ticker_display_name(ticker, quote)
        price = fmt_currency(quote.price, quote.currency) if quote else "N/A"
        change = fmt_signed(quote.day_change) if quote else "N/A"
        change_pct = fmt_percent(quote.day_change_pct) if quote else "N/A"
        color = "#22c55e" if quote and quote.day_change and quote.day_change >= 0 else "#fb7185"
        st.sidebar.markdown(
            f"""
            <div class="ticker-card">
                <strong>{escape(ticker)}</strong><br>
                <span>{escape(name)}</span><br>
                <div style="margin-top:6px;">{escape(price)}</div>
                <div style="color:{color}; font-size:0.86rem;">{escape(change)} ({escape(change_pct)})</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


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


def main() -> None:
    holdings = render_sidebar()
    selected = holdings["ticker"].tolist()
    if not selected:
        st.warning("請至少選擇一個追蹤標的。")
        return

    with st.spinner("正在更新 Yahoo Finance 資料..."):
        tickers = tuple(selected)
        prices = load_history(tickers, period="20y")
        quotes = load_quotes(tickers)
        dividends = {ticker: load_dividends(ticker) for ticker in selected}
        fx_rate = load_fx_rate()
        news = load_news(tickers)

    weights = portfolio_weights(holdings, quotes, fx_rate)
    render_ticker_cards(selected, quotes)

    st.title("投資儀表板")
    st.caption("即時價格與買入價保留原幣別；市值、成本、損益與配置比例統一換算為台幣。新聞取自 Yahoo奇摩股市，快取時間：5 分鐘。")

    if prices.empty:
        st.error("目前無法取得價格歷史資料，請稍後重新整理。")
        return

    metrics = metrics_table(prices)
    growth = growth_curve(prices)
    annual_dividends = yearly_dividends(dividends)
    holdings_summary = render_holdings_summary(holdings, quotes, fx_rate)
    held_tickers = holdings.loc[holdings["quantity"] > 0, "ticker"].tolist()
    held_metrics = metrics.reindex(held_tickers).dropna(how="all") if held_tickers else pd.DataFrame()
    held_summary = holdings_summary.reindex(held_tickers).dropna(how="all") if held_tickers else pd.DataFrame()
    observation_metrics = metrics

    render_position_metrics(held_summary, fx_rate)

    tab_holdings, tab_observation, tab_security, tab_dividends, tab_news = st.tabs(
        ["持有資產", "觀察指標", "個股分析", "配息資訊", "新聞摘要"]
    )

    with tab_holdings:
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

        if held_metrics.empty:
            st.info("目前沒有數量大於 0 的持有標的歷史指標。")
        else:
            st.subheader("持有標的歷史指標")
            display_held_metrics = held_metrics.copy()
            st.dataframe(
                add_display_name_column(percent_dataframe(display_held_metrics), quotes),
                width="stretch",
                height=240,
            )

    with tab_observation:
        growth_window = st.radio(
            "成長曲線期間",
            options=["1年", "5年", "20年"],
            index=1,
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

    with tab_security:
        render_security_analysis(selected, quotes, holdings_summary)

    with tab_dividends:
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

    with tab_news:
        render_news(news)


if __name__ == "__main__":
    main()
