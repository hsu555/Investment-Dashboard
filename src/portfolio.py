from __future__ import annotations

import pandas as pd
import streamlit as st

from src.formatting import fmt_currency, fmt_percent
from src.names import ticker_display_name
from src.ui import quote_currency


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
    cols[0].metric("持有市值(TWD)", fmt_currency(float(total_market_value)) if total_market_value else "N/A")
    cols[1].metric("投入成本(TWD)", fmt_currency(float(total_cost)) if total_cost else "N/A")
    cols[2].metric("未實現損益(TWD)", fmt_currency(float(total_gain)) if total_gain is not None else "N/A")
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
