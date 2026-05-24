from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.portfolio import twd_fx_rate
from src.supabase_store import (
    SupabaseConfigError,
    SupabaseRequestError,
    load_user_transactions,
    save_user_transactions,
    supabase_configured,
)
from src.ui import quote_currency


TRANSACTIONS_FILE = Path(__file__).resolve().parents[1] / "transactions.json"
TRANSACTION_TYPES = ("BUY", "SELL", "DIVIDEND", "DEPOSIT", "WITHDRAW")
TRANSACTION_TYPE_LABELS = {
    "BUY": "買入",
    "SELL": "賣出",
    "DIVIDEND": "配息",
    "DEPOSIT": "入金",
    "WITHDRAW": "出金",
}
TRANSACTION_LABEL_TO_TYPE = {label: key for key, label in TRANSACTION_TYPE_LABELS.items()}


def clean_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    columns = ["date", "type", "ticker", "quantity", "price", "currency", "fx_rate", "fee_twd", "note"]
    if frame.empty:
        return pd.DataFrame(columns=columns)

    cleaned = frame.copy()
    for column in columns:
        if column not in cleaned:
            cleaned[column] = "" if column in {"date", "type", "ticker", "currency", "note"} else 0.0

    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce").dt.date
    cleaned["date"] = cleaned["date"].fillna(dt.date.today())
    cleaned["type"] = cleaned["type"].fillna("BUY").astype(str).str.strip()
    cleaned["type"] = cleaned["type"].map(lambda value: TRANSACTION_LABEL_TO_TYPE.get(value, value.upper()))
    cleaned["type"] = cleaned["type"].where(cleaned["type"].isin(TRANSACTION_TYPES), "BUY")
    cleaned["ticker"] = cleaned["ticker"].fillna("").astype(str).str.strip().str.upper()
    cleaned["currency"] = cleaned["currency"].fillna("").astype(str).str.strip().str.upper()
    for column in ["quantity", "price", "fx_rate", "fee_twd"]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(0.0).clip(lower=0)
    cleaned["note"] = cleaned["note"].fillna("").astype(str)
    cleaned = cleaned.sort_values(["date", "ticker", "type"]).reset_index(drop=True)
    return cleaned[columns]


def transactions_for_display(transactions: pd.DataFrame) -> pd.DataFrame:
    display = clean_transactions(transactions)
    if display.empty:
        return display
    display = display.copy()
    display["type"] = display["type"].map(TRANSACTION_TYPE_LABELS).fillna(display["type"])
    return display


def transactions_from_display(transactions: pd.DataFrame) -> pd.DataFrame:
    return clean_transactions(transactions)


def load_transactions(user_id: str | None) -> pd.DataFrame:
    if user_id and supabase_configured():
        try:
            return clean_transactions(load_user_transactions(user_id))
        except (SupabaseConfigError, SupabaseRequestError, OSError, ValueError) as exc:
            st.warning(f"無法從 Supabase 載入交易紀錄，暫以空白紀錄顯示：{exc}")
            return clean_transactions(pd.DataFrame())

    try:
        payload = json.loads(TRANSACTIONS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return clean_transactions(pd.DataFrame())

    if not isinstance(payload, list):
        return clean_transactions(pd.DataFrame())
    return clean_transactions(pd.DataFrame(payload))


def save_transactions(transactions: pd.DataFrame, user_id: str | None) -> None:
    cleaned = clean_transactions(transactions)
    records = cleaned.assign(date=cleaned["date"].map(lambda value: value.isoformat())).to_dict(orient="records")
    if user_id and supabase_configured():
        save_user_transactions(user_id, cleaned)
        return

    TRANSACTIONS_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def initial_transactions_from_holdings(holdings: pd.DataFrame, quotes, fx_rate: float | None) -> pd.DataFrame:
    rows = []
    today = dt.date.today().isoformat()
    for row in holdings.itertuples(index=False):
        quantity = float(row.quantity)
        purchase_price = float(row.purchase_price)
        if quantity <= 0 or purchase_price <= 0:
            continue
        quote = quotes.get(row.ticker)
        currency = quote_currency(row.ticker, quote)
        conversion = twd_fx_rate(row.ticker, quote, fx_rate) or 1.0
        rows.append(
            {
                "date": today,
                "type": "BUY",
                "ticker": row.ticker,
                "quantity": quantity,
                "price": purchase_price,
                "currency": currency,
                "fx_rate": conversion,
                "fee_twd": 0.0,
                "note": "由目前持倉建立的初始紀錄",
            }
        )
    return clean_transactions(pd.DataFrame(rows))


def transaction_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    clean = clean_transactions(transactions)
    if clean.empty:
        return pd.DataFrame()

    signed = clean.copy()
    signed["cash_flow_twd"] = signed.apply(_cash_flow_twd, axis=1)
    by_type = signed.groupby("type", as_index=False).agg(
        筆數=("type", "size"),
        現金流_TWD=("cash_flow_twd", "sum"),
        手續費_TWD=("fee_twd", "sum"),
    )
    by_type["type"] = by_type["type"].map(TRANSACTION_TYPE_LABELS).fillna(by_type["type"])
    return by_type.set_index("type")


def ticker_realized_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    clean = clean_transactions(transactions)
    security_trades = clean[clean["type"].isin(["BUY", "SELL"]) & (clean["ticker"] != "")].copy()
    if security_trades.empty:
        return pd.DataFrame()

    rows = []
    for ticker, group in security_trades.groupby("ticker"):
        buy = group[group["type"] == "BUY"]
        sell = group[group["type"] == "SELL"]
        buy_qty = float(buy["quantity"].sum())
        sell_qty = float(sell["quantity"].sum())
        buy_cost = float((buy["quantity"] * buy["price"] * buy["fx_rate"] + buy["fee_twd"]).sum())
        sell_proceeds = float((sell["quantity"] * sell["price"] * sell["fx_rate"] - sell["fee_twd"]).sum())
        avg_cost = buy_cost / buy_qty if buy_qty > 0 else None
        realized_cost = avg_cost * sell_qty if avg_cost is not None else None
        rows.append(
            {
                "標的": ticker,
                "買入股數": buy_qty,
                "賣出股數": sell_qty,
                "淨股數": buy_qty - sell_qty,
                "買入成本(TWD)": buy_cost,
                "賣出收入(TWD)": sell_proceeds,
                "估計已實現損益(TWD)": sell_proceeds - realized_cost if realized_cost is not None else None,
            }
        )
    return pd.DataFrame(rows).set_index("標的")


def holdings_from_transactions(transactions: pd.DataFrame, existing_holdings: pd.DataFrame) -> pd.DataFrame:
    clean = clean_transactions(transactions)
    trades = clean[clean["type"].isin(["BUY", "SELL"]) & (clean["ticker"] != "")].copy()

    existing = existing_holdings.copy()
    if existing.empty:
        existing = pd.DataFrame(columns=["order", "ticker", "quantity", "purchase_price"])

    rows_by_ticker: dict[str, dict[str, object]] = {}
    for row in existing.itertuples(index=False):
        rows_by_ticker[str(row.ticker)] = {
            "order": int(row.order),
            "ticker": str(row.ticker),
            "quantity": float(row.quantity),
            "purchase_price": float(row.purchase_price),
        }

    next_order = int(existing["order"].max()) + 1 if "order" in existing and not existing.empty else 1
    for ticker, group in trades.groupby("ticker"):
        buy = group[group["type"] == "BUY"]
        sell = group[group["type"] == "SELL"]
        buy_qty = float(buy["quantity"].sum())
        sell_qty = float(sell["quantity"].sum())
        net_qty = max(buy_qty - sell_qty, 0.0)
        avg_purchase_price = float((buy["quantity"] * buy["price"]).sum() / buy_qty) if buy_qty > 0 else 0.0

        if ticker in rows_by_ticker:
            rows_by_ticker[ticker]["quantity"] = net_qty
            rows_by_ticker[ticker]["purchase_price"] = avg_purchase_price if net_qty > 0 else 0.0
        else:
            rows_by_ticker[ticker] = {
                "order": next_order,
                "ticker": ticker,
                "quantity": net_qty,
                "purchase_price": avg_purchase_price if net_qty > 0 else 0.0,
            }
            next_order += 1

    result = pd.DataFrame(rows_by_ticker.values())
    if result.empty:
        return existing[["order", "ticker", "quantity", "purchase_price"]]
    result = result.sort_values(["order", "ticker"]).reset_index(drop=True)
    result["order"] = range(1, len(result) + 1)
    return result[["order", "ticker", "quantity", "purchase_price"]]


def _cash_flow_twd(row: pd.Series) -> float:
    amount = float(row["quantity"]) * float(row["price"]) * float(row["fx_rate"])
    fee = float(row["fee_twd"])
    if row["type"] == "BUY":
        return -(amount + fee)
    if row["type"] == "SELL":
        return amount - fee
    if row["type"] == "DIVIDEND":
        return amount - fee
    if row["type"] == "DEPOSIT":
        return amount
    if row["type"] == "WITHDRAW":
        return -amount
    return 0.0
