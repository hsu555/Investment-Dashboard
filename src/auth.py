from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


PASSWORD_SECRET_KEY = "dashboard_password"
AUTH_TOKEN_QUERY_PARAM = "auth_token"
AUTH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300


@st.cache_resource
def _auth_store() -> dict:
    """Server-side lockout state shared across refreshes and sessions."""
    return {"lockout_until": 0.0, "failed_attempts": 0}


def get_configured_password() -> str:
    try:
        return str(st.secrets.get(PASSWORD_SECRET_KEY, ""))
    except StreamlitSecretNotFoundError:
        return ""


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def create_auth_token(configured_password: str) -> str:
    payload = {"exp": int(time.time() + AUTH_TOKEN_TTL_SECONDS)}
    payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        configured_password.encode("utf-8"),
        payload_part.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_part}.{_b64url_encode(signature)}"


def is_valid_auth_token(token: str, configured_password: str) -> bool:
    if not token or not configured_password or "." not in token:
        return False

    payload_part, signature_part = token.split(".", 1)
    expected_signature = hmac.new(
        configured_password.encode("utf-8"),
        payload_part.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        supplied_signature = _b64url_decode(signature_part)
        payload = json.loads(_b64url_decode(payload_part))
    except (ValueError, json.JSONDecodeError):
        return False

    if not hmac.compare_digest(supplied_signature, expected_signature):
        return False
    return float(payload.get("exp", 0)) > time.time()


def set_auth_token(configured_password: str) -> None:
    st.query_params[AUTH_TOKEN_QUERY_PARAM] = create_auth_token(configured_password)


def clear_auth_token() -> None:
    if AUTH_TOKEN_QUERY_PARAM in st.query_params:
        del st.query_params[AUTH_TOKEN_QUERY_PARAM]


def check_password() -> bool:
    if st.session_state.get("password_authenticated", False):
        return True

    configured_password = get_configured_password()
    auth_token = st.query_params.get(AUTH_TOKEN_QUERY_PARAM, "")
    if is_valid_auth_token(auth_token, configured_password):
        st.session_state.password_authenticated = True
        st.session_state.login_failed = False
        return True
    if auth_token:
        clear_auth_token()

    st.title("投資儀表板")
    st.caption("請先輸入密碼，通過後才會載入持倉與投資資料。")

    store = _auth_store()
    if store["lockout_until"] > time.time():
        remaining = int(store["lockout_until"] - time.time())
        st.error(f"登入嘗試次數過多，請等待 {remaining} 秒後再試。")
        return False

    with st.form("login_form"):
        password = st.text_input("密碼", type="password")
        submitted = st.form_submit_button("登入", type="primary")

    if submitted:
        if not configured_password:
            st.error(f"尚未設定登入密碼。請在 Streamlit Secrets 新增 `{PASSWORD_SECRET_KEY}`。")
            return False
        if hmac.compare_digest(password, configured_password):
            st.session_state.password_authenticated = True
            st.session_state.login_failed = False
            store["failed_attempts"] = 0
            set_auth_token(configured_password)
            st.rerun()

        store["failed_attempts"] += 1
        st.session_state.login_failed = True
        if store["failed_attempts"] >= _MAX_FAILED_ATTEMPTS:
            store["lockout_until"] = time.time() + _LOCKOUT_SECONDS
            store["failed_attempts"] = 0
            st.rerun()

    if st.session_state.get("login_failed", False):
        remaining_attempts = _MAX_FAILED_ATTEMPTS - store["failed_attempts"]
        st.error(f"密碼錯誤，無法存取儀表板。（還剩 {remaining_attempts} 次機會）")
    return False


def require_password() -> None:
    if not check_password():
        st.stop()
