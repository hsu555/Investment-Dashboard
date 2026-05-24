from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.supabase_store import (
    SupabaseConfigError,
    SupabaseRequestError,
    load_user_target_allocations,
    save_user_target_allocations,
    supabase_configured,
)


TARGETS_FILE = Path(__file__).resolve().parents[1] / "target_allocations.json"


def clean_target_allocations(frame: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    if frame.empty:
        cleaned = pd.DataFrame(columns=["ticker", "target_weight"])
    else:
        cleaned = frame.copy()
        for column in ["ticker", "target_weight"]:
            if column not in cleaned:
                cleaned[column] = "" if column == "ticker" else 0.0
        cleaned["ticker"] = cleaned["ticker"].fillna("").astype(str).map(lambda value: value.strip().upper())
        cleaned = cleaned[cleaned["ticker"] != ""]
        cleaned["target_weight"] = pd.to_numeric(cleaned["target_weight"], errors="coerce").fillna(0.0).clip(0, 1)
        cleaned = cleaned.drop_duplicates(subset="ticker", keep="last")
        cleaned = cleaned[["ticker", "target_weight"]]

    if tickers is not None:
        known = pd.DataFrame({"ticker": [ticker.strip().upper() for ticker in tickers]})
        cleaned = known.merge(cleaned, on="ticker", how="left")
        cleaned["target_weight"] = cleaned["target_weight"].fillna(0.0).clip(0, 1)

    return cleaned.sort_values("ticker").reset_index(drop=True)


def load_target_allocations(user_id: str | None, tickers: list[str]) -> pd.DataFrame:
    if user_id and supabase_configured():
        try:
            return clean_target_allocations(load_user_target_allocations(user_id), tickers)
        except (SupabaseConfigError, SupabaseRequestError, OSError, ValueError) as exc:
            st.warning(f"無法從 Supabase 載入目標配置，暫以空白配置顯示：{exc}")
            return clean_target_allocations(pd.DataFrame(), tickers)

    try:
        payload = json.loads(TARGETS_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return clean_target_allocations(pd.DataFrame(), tickers)

    if not isinstance(payload, list):
        return clean_target_allocations(pd.DataFrame(), tickers)
    return clean_target_allocations(pd.DataFrame(payload), tickers)


def save_target_allocations(targets: pd.DataFrame, user_id: str | None) -> None:
    records = clean_target_allocations(targets).to_dict(orient="records")
    if user_id and supabase_configured():
        save_user_target_allocations(user_id, pd.DataFrame(records))
        return

    TARGETS_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def rebalance_table(
    holdings_summary: pd.DataFrame,
    targets: pd.DataFrame,
    new_cash_twd: float = 0.0,
) -> pd.DataFrame:
    if holdings_summary.empty:
        return pd.DataFrame()

    current = holdings_summary.reset_index()[["標的", "名稱", "市值(TWD)", "現價", "匯率"]].rename(columns={"標的": "ticker"})
    current["市值(TWD)"] = pd.to_numeric(current["市值(TWD)"], errors="coerce").fillna(0.0)
    target_map = clean_target_allocations(targets).set_index("ticker")["target_weight"].to_dict()
    current["目標比例"] = current["ticker"].map(target_map).fillna(0.0)

    current_total = float(current["市值(TWD)"].sum())
    target_total = current_total + float(new_cash_twd or 0.0)
    current["目前比例"] = current["市值(TWD)"] / current_total if current_total > 0 else 0.0
    current["目標市值(TWD)"] = current["目標比例"] * target_total
    current["需調整金額(TWD)"] = current["目標市值(TWD)"] - current["市值(TWD)"]
    current["偏離比例"] = current["目前比例"] - current["目標比例"]
    current["估計股數"] = current.apply(
        lambda row: row["需調整金額(TWD)"] / (row["現價"] * row["匯率"])
        if row["現價"] not in (None, 0) and row["匯率"] not in (None, 0) and pd.notna(row["現價"]) and pd.notna(row["匯率"])
        else None,
        axis=1,
    )
    current["建議動作"] = current["需調整金額(TWD)"].map(lambda value: "買入" if value > 1 else "賣出" if value < -1 else "維持")
    return current.set_index("ticker")
