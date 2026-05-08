from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics import (
    growth_curve,
    metrics_table,
    yearly_dividends,
)
from src.charts import allocation_pie, comparison_chart, dividend_chart, growth_chart
from src.config import DEFAULT_TICKERS, TICKER_DISPLAY_NAMES
from src.data import (
    load_dividends,
    load_fx_rate,
    load_history,
    load_news,
    load_quotes,
)
from src.formatting import fmt_currency, fmt_percent, fmt_signed


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
        name = TICKER_DISPLAY_NAMES.get(ticker) or (quote.long_name if quote else ticker)
        price = fmt_currency(quote.price, quote.currency) if quote else "N/A"
        change = fmt_signed(quote.day_change) if quote else "N/A"
        change_pct = fmt_percent(quote.day_change_pct) if quote else "N/A"
        color = "#22c55e" if quote and quote.day_change and quote.day_change >= 0 else "#fb7185"
        st.sidebar.markdown(
            f"""
            <div class="ticker-card">
                <strong>{ticker}</strong><br>
                <span>{name}</span><br>
                <div style="margin-top:6px;">{price}</div>
                <div style="color:{color}; font-size:0.86rem;">{change} ({change_pct})</div>
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
        prices = load_history(tickers)
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

    tab_holdings, tab_observation, tab_dividends, tab_news = st.tabs(["持有資產", "觀察指標", "配息資訊", "新聞摘要"])

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
                percent_dataframe(display_held_metrics),
                width="stretch",
                height=240,
            )

    with tab_observation:
        observation_growth = growth
        st.plotly_chart(growth_chart(observation_growth, observation_metrics), width="stretch")
        st.plotly_chart(comparison_chart(observation_metrics), width="stretch")
        st.subheader("追蹤標的歷史指標")
        display_metrics = observation_metrics.copy()
        st.dataframe(
            percent_dataframe(display_metrics),
            width="stretch",
            height=300,
        )

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
