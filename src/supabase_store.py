from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import secrets
import time
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


SUPABASE_URL_KEY = "supabase_url"
SUPABASE_SERVICE_KEY = "supabase_service_role_key"
DEFAULT_USERNAME_KEY = "default_username"
PASSWORD_SECRET_KEY = "dashboard_password"
DEFAULT_USERNAME = "hsu555"
PORTFOLIO_FILE = Path(__file__).resolve().parents[1] / "portfolio.json"


class SupabaseConfigError(RuntimeError):
    pass


class SupabaseRequestError(RuntimeError):
    pass


def _secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except (StreamlitSecretNotFoundError, FileNotFoundError):
        value = os.environ.get(name.upper(), default)
    return str(value or "").strip()


def supabase_configured() -> bool:
    return bool(_secret(SUPABASE_URL_KEY) and _secret(SUPABASE_SERVICE_KEY))


def default_username() -> str:
    return _secret(DEFAULT_USERNAME_KEY, DEFAULT_USERNAME) or DEFAULT_USERNAME


def configured_password() -> str:
    return _secret(PASSWORD_SECRET_KEY)


def _base_url() -> str:
    url = _secret(SUPABASE_URL_KEY).rstrip("/")
    if not url:
        raise SupabaseConfigError(f"尚未設定 `{SUPABASE_URL_KEY}`。")
    return url


def _service_key() -> str:
    key = _secret(SUPABASE_SERVICE_KEY)
    if not key:
        raise SupabaseConfigError(f"尚未設定 `{SUPABASE_SERVICE_KEY}`。")
    return key


def _headers(prefer: str | None = None) -> dict[str, str]:
    key = _service_key()
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    if prefer:
        headers["Prefer"] = prefer
    return headers


def _request(method: str, path: str, **kwargs) -> Any:
    prefer = kwargs.pop("prefer", None)
    timeout = kwargs.pop("timeout", 30)
    url = f"{_base_url()}/rest/v1/{path}"
    last_error: requests.RequestException | None = None
    for attempt in range(3):
        try:
            response = requests.request(
                method,
                url,
                headers=_headers(prefer),
                timeout=timeout,
                **kwargs,
            )
            break
        except requests.RequestException as exc:
            last_error = exc
            if attempt == 2:
                raise SupabaseRequestError(f"Supabase {method} {path} failed: {exc}") from exc
            time.sleep(0.5 * (attempt + 1))
    else:
        raise SupabaseRequestError(f"Supabase {method} {path} failed: {last_error}")

    if response.status_code >= 400:
        raise SupabaseRequestError(f"Supabase {method} {path} failed: {response.status_code} {response.text}")
    if not response.content:
        return None
    return response.json()


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    rounds = 260_000
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return f"pbkdf2_sha256${rounds}${base64.b64encode(salt).decode('ascii')}${base64.b64encode(digest).decode('ascii')}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algorithm, rounds_text, salt_text, digest_text = password_hash.split("$", 3)
        if algorithm != "pbkdf2_sha256":
            return False
        rounds = int(rounds_text)
        salt = base64.b64decode(salt_text)
        expected = base64.b64decode(digest_text)
    except (ValueError, TypeError):
        return False

    supplied = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, rounds)
    return hmac.compare_digest(supplied, expected)


def get_user_by_username(username: str) -> dict[str, Any] | None:
    encoded = quote(username.strip(), safe="")
    rows = _request("GET", f"dashboard_users?username=eq.{encoded}&select=id,username,password_hash")
    return rows[0] if rows else None


def create_user(username: str, password: str) -> dict[str, Any]:
    payload = {"username": username.strip(), "password_hash": hash_password(password)}
    rows = _request(
        "POST",
        "dashboard_users?select=id,username,password_hash",
        data=json.dumps(payload),
        prefer="return=representation",
    )
    if not rows:
        raise SupabaseRequestError("Supabase did not return the created user.")
    return rows[0]


def ensure_default_user() -> dict[str, Any] | None:
    if not supabase_configured():
        return None

    username = default_username()
    password = configured_password()
    user = get_user_by_username(username)
    if user is not None:
        return user
    if not password:
        return None

    user = create_user(username, password)
    seed_user_holdings(user["id"])
    return user


def authenticate_user(username: str, password: str) -> dict[str, Any] | None:
    user = get_user_by_username(username)
    if not user:
        return None
    return user if verify_password(password, str(user.get("password_hash", ""))) else None


def holdings_count(user_id: str) -> int:
    encoded = quote(user_id, safe="")
    rows = _request("GET", f"holdings?user_id=eq.{encoded}&select=id")
    return len(rows or [])


def local_portfolio_records() -> list[dict[str, Any]]:
    try:
        payload = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def load_user_holdings(user_id: str) -> pd.DataFrame:
    encoded = quote(user_id, safe="")
    rows = _request(
        "GET",
        "holdings"
        f"?user_id=eq.{encoded}"
        "&select=order_index,ticker,quantity,purchase_price"
        "&order=order_index.asc",
    )
    records = [
        {
            "order": row.get("order_index"),
            "ticker": row.get("ticker"),
            "quantity": row.get("quantity"),
            "purchase_price": row.get("purchase_price"),
        }
        for row in (rows or [])
    ]
    return pd.DataFrame(records)


def save_user_holdings(user_id: str, holdings: pd.DataFrame) -> None:
    encoded = quote(user_id, safe="")
    _request("DELETE", f"holdings?user_id=eq.{encoded}")
    if holdings.empty:
        return

    records = []
    for row in holdings.itertuples(index=False):
        records.append(
            {
                "user_id": user_id,
                "order_index": int(row.order),
                "ticker": str(row.ticker),
                "quantity": float(row.quantity),
                "purchase_price": float(row.purchase_price),
            }
        )
    _request("POST", "holdings", data=json.dumps(records), prefer="return=minimal")


def seed_user_holdings(user_id: str) -> None:
    if holdings_count(user_id) > 0:
        return
    records = local_portfolio_records()
    if records:
        frame = pd.DataFrame(records)
        for column in ["order", "ticker", "quantity", "purchase_price"]:
            if column not in frame:
                frame[column] = "" if column == "ticker" else 0
        save_user_holdings(user_id, frame[["order", "ticker", "quantity", "purchase_price"]])


def load_retirement_settings(user_id: str) -> dict[str, Any] | None:
    encoded = quote(user_id, safe="")
    rows = _request("GET", f"retirement_settings?user_id=eq.{encoded}&select=*")
    return rows[0] if rows else None


def save_retirement_settings(user_id: str, inputs: Any) -> None:
    payload = {
        "user_id": user_id,
        "current_age": int(inputs.current_age),
        "retirement_age": int(inputs.retirement_age),
        "life_expectancy": int(inputs.life_expectancy),
        "current_assets_wan": float(inputs.current_assets_wan),
        "monthly_contribution_wan": float(inputs.monthly_contribution_wan),
        "monthly_expense_wan": float(inputs.monthly_expense_wan),
        "mean_annual_return": float(inputs.mean_annual_return),
        "annual_return_std": float(inputs.annual_return_std),
        "inflation_rate": float(inputs.inflation_rate),
        "n_simulations": int(inputs.n_simulations),
    }
    _request(
        "POST",
        "retirement_settings",
        data=json.dumps(payload),
        prefer="resolution=merge-duplicates,return=minimal",
    )
