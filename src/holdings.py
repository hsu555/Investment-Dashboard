from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.config import DEFAULT_HOLDINGS, DEFAULT_TICKERS


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
