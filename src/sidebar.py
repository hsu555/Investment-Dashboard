from __future__ import annotations

import pandas as pd
import streamlit as st

from src.auth import clear_auth_token
from src.data import read_timing_log
from src.holdings import clean_holdings, clear_sidebar_editor_state, load_holdings, save_holdings
from src.ui import sidebar_market_summary_html


def render_sidebar() -> tuple[pd.DataFrame, dict, list[tuple[str, object]]]:
    st.sidebar.title("追蹤清單")
    if st.sidebar.button("登出", width="stretch"):
        st.session_state.password_authenticated = False
        st.session_state.login_failed = False
        clear_auth_token()
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

    with st.sidebar.expander("API 計時紀錄", expanded=False):
        log = read_timing_log()
        if log.strip():
            st.code(log, language=None)
        else:
            st.caption("尚無紀錄（快取命中時不會重新呼叫 API）")

    return holdings, quotes, quote_slots


