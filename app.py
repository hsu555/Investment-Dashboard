from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import time

import pandas as pd
import streamlit as st

from src.analytics import (
    correlation_matrix,
    growth_curve,
    metrics_table,
    portfolio_growth_curve,
    portfolio_risk_metrics,
    technical_indicators,
    technical_signal_table,
    yearly_dividends,
)
from src.charts import (
    allocation_pie,
    comparison_chart,
    correlation_heatmap,
    dividend_chart,
    growth_chart,
    momentum_chart,
    technical_price_chart,
)
from src.auth import require_password
from src.config import CACHE_TTL_NEWS
from src.data import (
    clear_timing_log,
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
from src.portfolio import (
    portfolio_weights,
    render_dividend_summary,
    render_holdings_summary,
    render_position_metrics,
)
from src.retirement_ui import render_retirement_view
from src.sidebar import render_sidebar
from src.holdings import clear_sidebar_editor_state, save_holdings
from src.targets import load_target_allocations, rebalance_table, save_target_allocations
from src.transactions import (
    TRANSACTION_TYPE_LABELS,
    holdings_from_transactions,
    initial_transactions_from_holdings,
    load_transactions,
    save_transactions,
    ticker_realized_summary,
    transaction_summary,
    transactions_for_display,
    transactions_from_display,
)
from src.ui import (
    add_display_name_column,
    configure_page,
    fmt_compact,
    fmt_number,
    percent_dataframe,
    sidebar_market_summary_html,
    value_or_fallback,
)


@st.cache_resource
def background_executor() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(max_workers=4)


configure_page()


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


def update_sidebar_quote_slots(quote_slots: list[tuple[str, object]], quotes) -> None:
    for ticker, slot in quote_slots:
        slot.markdown(sidebar_market_summary_html(ticker, quotes.get(ticker)), unsafe_allow_html=True)


def quote_has_price(quote) -> bool:
    return quote is not None and quote.price is not None and quote.price == quote.price


def merge_quote_updates(previous_quotes: dict, updated_quotes: dict) -> dict:
    merged = dict(previous_quotes)
    for ticker, quote in updated_quotes.items():
        previous = previous_quotes.get(ticker)
        if quote_has_price(quote) or not quote_has_price(previous):
            merged[ticker] = quote
    return merged


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


def current_user_id() -> str | None:
    current_user = st.session_state.get("current_user", {})
    return current_user.get("id")


def render_rebalance_view(held_summary: pd.DataFrame) -> None:
    st.markdown("##### 目標配置與再平衡")
    if held_summary.empty:
        st.info("目前沒有持有部位，尚無法計算再平衡建議。")
        return

    tickers = held_summary.index.astype(str).tolist()
    user_id = current_user_id()
    targets_key = f"target_allocations_{'|'.join(tickers)}"
    if targets_key not in st.session_state:
        st.session_state[targets_key] = load_target_allocations(user_id, tickers)

    targets = st.session_state[targets_key].copy()
    editor = targets.assign(target_weight=targets["target_weight"] * 100).rename(columns={"ticker": "標的", "target_weight": "目標比例(%)"})
    edited = st.data_editor(
        editor,
        width="stretch",
        hide_index=True,
        disabled=["標的"],
        column_config={
            "目標比例(%)": st.column_config.NumberColumn("目標比例(%)", min_value=0.0, max_value=100.0, step=1.0, format="%.1f"),
        },
        key="target_allocation_editor",
    )
    clean_targets = edited.rename(columns={"標的": "ticker", "目標比例(%)": "target_weight"})
    clean_targets["target_weight"] = pd.to_numeric(clean_targets["target_weight"], errors="coerce").fillna(0.0) / 100
    target_total = float(clean_targets["target_weight"].sum()) if "target_weight" in clean_targets else 0.0
    cols = st.columns([0.34, 0.33, 0.33])
    new_cash = cols[0].number_input("新增投入現金(TWD)", min_value=0.0, step=1000.0, value=0.0)
    cols[1].metric("目標比例合計", fmt_percent(target_total))
    if cols[2].button("儲存目標配置", type="primary", width="stretch"):
        try:
            save_target_allocations(clean_targets, user_id)
            st.session_state[targets_key] = clean_targets
            st.success("目標配置已儲存。")
        except Exception as exc:
            st.error(f"目標配置儲存失敗：{exc}")

    if abs(target_total - 1.0) > 0.001:
        st.warning("目標比例合計建議等於 100%，目前仍會依你輸入的比例計算。")

    rebalanced = rebalance_table(held_summary, clean_targets, new_cash)
    if rebalanced.empty:
        return
    display = rebalanced.copy()
    for column in ["目前比例", "目標比例", "偏離比例"]:
        display[column] = display[column].map(fmt_percent)
    for column in ["市值(TWD)", "目標市值(TWD)", "需調整金額(TWD)"]:
        display[column] = display[column].map(lambda value: fmt_currency(value, "TWD"))
    display["估計股數"] = display["估計股數"].map(lambda value: fmt_number(value, 2))
    st.dataframe(
        display[["名稱", "目前比例", "目標比例", "偏離比例", "市值(TWD)", "目標市值(TWD)", "建議動作", "需調整金額(TWD)", "估計股數"]],
        width="stretch",
        height=280,
    )


def render_transactions_view(holdings: pd.DataFrame, quotes, fx_rate: float | None) -> None:
    st.subheader("交易紀錄 / 現金流")
    st.caption(
        "目前持倉仍由側欄管理；這裡用來補交易流水帳，後續可支援 XIRR、已實現損益與投入本金曲線。"
        "現金類紀錄可用數量 1、成交價填現金金額。"
    )
    user_id = current_user_id()
    if "transactions" not in st.session_state:
        st.session_state.transactions = load_transactions(user_id)

    if st.session_state.transactions.empty:
        st.info("尚無交易紀錄。可以先從目前持倉建立一批初始買入紀錄，再逐筆修正日期、匯率與手續費。")
        if st.button("從目前持倉建立初始紀錄", type="primary"):
            st.session_state.transactions = initial_transactions_from_holdings(holdings, quotes, fx_rate)
            st.rerun()

    edited = st.data_editor(
        transactions_for_display(st.session_state.transactions),
        width="stretch",
        height=360,
        num_rows="dynamic",
        column_config={
            "date": st.column_config.DateColumn("日期"),
            "type": st.column_config.SelectboxColumn("類型", options=list(TRANSACTION_TYPE_LABELS.values()), required=True),
            "ticker": st.column_config.TextColumn("標的"),
            "quantity": st.column_config.NumberColumn("數量", min_value=0.0, step=1.0),
            "price": st.column_config.NumberColumn("成交價", min_value=0.0, step=0.01),
            "currency": st.column_config.TextColumn("幣別"),
            "fx_rate": st.column_config.NumberColumn("匯率", min_value=0.0, step=0.0001, format="%.4f"),
            "fee_twd": st.column_config.NumberColumn("手續費(TWD)", min_value=0.0, step=1.0),
            "note": st.column_config.TextColumn("備註"),
        },
        key="transaction_editor",
    )
    edited_transactions = transactions_from_display(edited)
    if st.button("儲存交易紀錄", type="primary"):
        try:
            save_transactions(edited_transactions, user_id)
            st.session_state.transactions = edited_transactions
            st.success("交易紀錄已儲存。")
        except Exception as exc:
            st.error(f"交易紀錄儲存失敗：{exc}")

    sync_cols = st.columns([0.68, 0.32])
    sync_cols[0].caption("同步會以「買入 / 賣出」推算淨股數與買入均價，並保留沒有交易紀錄的既有觀察標的。")
    if sync_cols[1].button("同步到左側持倉", width="stretch"):
        try:
            synced_holdings = holdings_from_transactions(edited_transactions, holdings)
            save_holdings(synced_holdings, user_id)
            st.session_state.holdings = synced_holdings
            clear_sidebar_editor_state()
            st.success("已依交易紀錄更新左側持倉。")
            st.rerun()
        except Exception as exc:
            st.error(f"同步持倉失敗：{exc}")

    summary = transaction_summary(edited_transactions)
    realized = ticker_realized_summary(edited_transactions)
    if not summary.empty:
        st.markdown("##### 現金流摘要")
        display_summary = summary.copy()
        for column in ["現金流_TWD", "手續費_TWD"]:
            display_summary[column] = display_summary[column].map(lambda value: fmt_currency(value, "TWD"))
        st.dataframe(display_summary, width="stretch", height=220)
    if not realized.empty:
        st.markdown("##### 標的彙總")
        display_realized = realized.copy()
        for column in ["買入成本(TWD)", "賣出收入(TWD)", "估計已實現損益(TWD)"]:
            display_realized[column] = display_realized[column].map(lambda value: fmt_currency(value, "TWD"))
        st.dataframe(display_realized, width="stretch", height=240)


def render_portfolio_risk(prices: pd.DataFrame, weights: dict[str, float], held_tickers: list[str]) -> None:
    st.markdown("##### 投資組合風險")
    if not held_tickers:
        st.info("目前沒有持有部位，尚無法計算投組風險。")
        return

    held_prices = prices[[ticker for ticker in held_tickers if ticker in prices]] if not prices.empty else pd.DataFrame()
    if held_prices.empty:
        st.info("持有標的缺少歷史價格，暫時無法計算投組風險。")
        return

    risk = portfolio_risk_metrics(held_prices, weights)
    cols = st.columns(6)
    cols[0].metric("總報酬率", fmt_percent(risk["total_return"]))
    cols[1].metric("年化報酬率", fmt_percent(risk["cagr"]))
    cols[2].metric("年化波動", fmt_percent(risk["annualized_volatility"]))
    cols[3].metric("最大回撤", fmt_percent(risk["max_drawdown"]))
    cols[4].metric("Sharpe", fmt_number(risk["sharpe_ratio"]))
    cols[5].metric("最差單日", fmt_percent(risk["worst_day"]))

    portfolio_growth = portfolio_growth_curve(held_prices, weights)
    if not portfolio_growth.empty:
        st.plotly_chart(growth_chart(portfolio_growth, title="投資組合成長曲線"), width="stretch")
    st.plotly_chart(correlation_heatmap(correlation_matrix(held_prices, held_tickers)), width="stretch")


def build_observation_data(tickers: tuple[str, ...]) -> dict[str, object]:
    prices = load_history(tickers, period="20y")
    metrics = metrics_table(prices) if not prices.empty else pd.DataFrame()
    return {"prices": prices, "metrics": metrics}


def build_dividend_data(tickers: tuple[str, ...]) -> dict[str, object]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {ticker: executor.submit(load_dividends, ticker) for ticker in tickers}
        dividends = {ticker: fut.result() for ticker, fut in futures.items()}
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
    if "timing_log_cleared" not in st.session_state:
        clear_timing_log()
        st.session_state.timing_log_cleared = True

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
    previous_quotes = st.session_state.get("latest_quotes", {})
    quotes = merge_quote_updates(previous_quotes, load_quotes(tickers))
    st.session_state.latest_quotes = quotes
    update_sidebar_quote_slots(quote_slots, quotes)
    load_status.write("取得 USD/TWD 匯率，換算台幣市值與損益。")
    fx_rate = load_fx_rate()
    if fx_rate is None:
        fx_rate = st.session_state.get("latest_fx_rate")
    else:
        st.session_state.latest_fx_rate = fx_rate
    load_status.update(label="核心資料已載入", state="complete", expanded=False)

    weights = portfolio_weights(holdings, quotes, fx_rate)
    holdings_summary = render_holdings_summary(holdings, quotes, fx_rate)
    held_tickers = holdings.loc[holdings["quantity"] > 0, "ticker"].tolist()
    held_summary = holdings_summary.reindex(held_tickers).dropna(how="all") if held_tickers else pd.DataFrame()

    render_position_metrics(held_summary, fx_rate)

    active_view = st.radio(
        "檢視",
        ["持有資產", "交易紀錄", "觀察指標", "個股分析", "配息資訊", "新聞摘要", "退休試算"],
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
            render_rebalance_view(held_summary)
        st.caption(prefetch_status_caption(prefetch_jobs))

    elif active_view == "交易紀錄":
        render_transactions_view(holdings, quotes, fx_rate)

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
        render_portfolio_risk(prices, weights, held_tickers)

    elif active_view == "個股分析":
        security_data = wait_for_prefetch(prefetch_jobs, "security", "個股分析資料")
        render_security_analysis(selected, quotes, holdings_summary, security_data)

    elif active_view == "配息資訊":
        dividend_data = wait_for_prefetch(prefetch_jobs, "dividends", "配息資料")
        if dividend_data is None:
            with st.status("正在載入配息資料...", expanded=True) as dividend_status:
                dividend_status.write(f"平行讀取 {len(selected)} 支標的配息紀錄。")
                with ThreadPoolExecutor(max_workers=2) as executor:
                    futures = {ticker: executor.submit(load_dividends, ticker) for ticker in selected}
                    dividends = {ticker: fut.result() for ticker, fut in futures.items()}
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

    elif active_view == "退休試算":
        total_market_value_twd = float(held_summary["市值(TWD)"].dropna().sum()) if not held_summary.empty else None
        render_retirement_view(total_market_value_twd)


if __name__ == "__main__":
    main()
