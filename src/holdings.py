from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_HOLDINGS, DEFAULT_TICKERS
from src.supabase_store import (
    SupabaseConfigError,
    SupabaseRequestError,
    default_username,
    load_user_holdings,
    save_user_holdings,
    seed_user_holdings,
    supabase_configured,
)


PORTFOLIO_FILE = Path(__file__).resolve().parents[1] / "portfolio.json"


def default_holdings() -> pd.DataFrame:
    if DEFAULT_HOLDINGS:
        records = [
            {
                "order": index,
                "ticker": holding["ticker"],
                "quantity": holding.get("quantity", 0.0),
                "purchase_price": holding.get("purchase_price", 0.0),
            }
            for index, holding in enumerate(DEFAULT_HOLDINGS, start=1)
        ]
        return clean_holdings(pd.DataFrame(records))

    return clean_holdings(
        pd.DataFrame(
            {
                "order": range(1, len(DEFAULT_TICKERS) + 1),
                "ticker": DEFAULT_TICKERS,
                "quantity": [0.0] * len(DEFAULT_TICKERS),
                "purchase_price": [0.0] * len(DEFAULT_TICKERS),
            }
        )
    )


def default_watchlist() -> pd.DataFrame:
    tickers = [holding["ticker"] for holding in DEFAULT_HOLDINGS] if DEFAULT_HOLDINGS else DEFAULT_TICKERS
    return clean_holdings(
        pd.DataFrame(
            {
                "order": range(1, len(tickers) + 1),
                "ticker": tickers,
                "quantity": [0.0] * len(tickers),
                "purchase_price": [0.0] * len(tickers),
            }
        )
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


def load_holdings(user_id: str | None = None, username: str | None = None) -> pd.DataFrame:
    if user_id and supabase_configured():
        try:
            if username == default_username():
                seed_user_holdings(user_id)
            loaded = clean_holdings(load_user_holdings(user_id))
            return loaded if not loaded.empty else default_watchlist()
        except (SupabaseConfigError, SupabaseRequestError, OSError, ValueError) as exc:
            st.error(f"無法從 Supabase 載入持倉資料：{exc}")
            st.stop()

    try:
        payload = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_holdings()

    if not isinstance(payload, list):
        return default_holdings()
    return clean_holdings(pd.DataFrame(payload))


def save_holdings(holdings: pd.DataFrame, user_id: str | None = None) -> None:
    records = clean_holdings(holdings).to_dict(orient="records")
    if user_id and supabase_configured():
        save_user_holdings(user_id, pd.DataFrame(records))
        return

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
