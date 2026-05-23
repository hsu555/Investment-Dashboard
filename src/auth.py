from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import streamlit as st

from src.supabase_store import (
    DEFAULT_USERNAME,
    PASSWORD_SECRET_KEY,
    SupabaseConfigError,
    SupabaseRequestError,
    authenticate_user,
    configured_password,
    create_user,
    default_username,
    ensure_default_user,
    supabase_configured,
)


AUTH_TOKEN_QUERY_PARAM = "auth_token"
AUTH_TOKEN_TTL_SECONDS = 30 * 24 * 60 * 60
_MAX_FAILED_ATTEMPTS = 5
_LOCKOUT_SECONDS = 300


@st.cache_resource
def _auth_store() -> dict:
    """Server-side lockout state shared across refreshes and sessions."""
    return {"lockout_until": 0.0, "failed_attempts": 0}


def _signing_secret() -> str:
    return configured_password() or "investment-dashboard-local-dev"


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: str) -> bytes:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding)


def create_auth_token(username: str) -> str:
    payload = {"username": username, "exp": int(time.time() + AUTH_TOKEN_TTL_SECONDS)}
    payload_part = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    signature = hmac.new(
        _signing_secret().encode("utf-8"),
        payload_part.encode("ascii"),
        hashlib.sha256,
    ).digest()
    return f"{payload_part}.{_b64url_encode(signature)}"


def parse_valid_auth_token(token: str) -> dict | None:
    if not token or "." not in token:
        return None

    payload_part, signature_part = token.split(".", 1)
    expected_signature = hmac.new(
        _signing_secret().encode("utf-8"),
        payload_part.encode("ascii"),
        hashlib.sha256,
    ).digest()
    try:
        supplied_signature = _b64url_decode(signature_part)
        payload = json.loads(_b64url_decode(payload_part))
    except (ValueError, json.JSONDecodeError):
        return None

    if not hmac.compare_digest(supplied_signature, expected_signature):
        return None
    if float(payload.get("exp", 0)) <= time.time():
        return None
    return payload


def set_auth_token(username: str) -> None:
    st.query_params[AUTH_TOKEN_QUERY_PARAM] = create_auth_token(username)


def clear_auth_token() -> None:
    if AUTH_TOKEN_QUERY_PARAM in st.query_params:
        del st.query_params[AUTH_TOKEN_QUERY_PARAM]


def _set_authenticated_user(user: dict) -> None:
    st.session_state.password_authenticated = True
    st.session_state.login_failed = False
    st.session_state.current_user = {
        "id": user.get("id"),
        "username": user.get("username"),
    }


def _restore_from_token() -> bool:
    auth_token = st.query_params.get(AUTH_TOKEN_QUERY_PARAM, "")
    payload = parse_valid_auth_token(auth_token)
    if not payload:
        if auth_token:
            clear_auth_token()
        return False

    username = str(payload.get("username", "")).strip()
    if not username:
        clear_auth_token()
        return False

    if supabase_configured():
        from src.supabase_store import get_user_by_username

        user = get_user_by_username(username)
        if not user:
            clear_auth_token()
            return False
        _set_authenticated_user(user)
        return True

    _set_authenticated_user({"id": "local", "username": username})
    return True


def _login_with_supabase(username: str, password: str) -> dict | None:
    if not supabase_configured():
        configured = configured_password()
        if username == default_username() and configured and hmac.compare_digest(password, configured):
            return {"id": "local", "username": username}
        return None

    ensure_default_user()
    return authenticate_user(username, password)


def _render_create_user() -> None:
    with st.expander("新增使用者", expanded=False):
        with st.form("create_user_form"):
            username = st.text_input("帳號", key="create_username")
            password = st.text_input("密碼", type="password", key="create_password")
            password_confirm = st.text_input("確認密碼", type="password", key="create_password_confirm")
            submitted = st.form_submit_button("建立使用者", type="secondary")

        if submitted:
            username = username.strip()
            if not supabase_configured():
                st.error("新增使用者需要先設定 Supabase。")
                return
            if not username or not password:
                st.error("請輸入帳號與密碼。")
                return
            if password != password_confirm:
                st.error("兩次輸入的密碼不一致。")
                return
            try:
                create_user(username, password)
                st.success("使用者已建立，可以用新帳號登入。")
            except SupabaseRequestError as exc:
                if "duplicate key" in str(exc).lower() or "23505" in str(exc):
                    st.error("這個帳號已存在。")
                else:
                    st.error(f"建立使用者失敗：{exc}")


def check_password() -> bool:
    if st.session_state.get("password_authenticated", False) and st.session_state.get("current_user"):
        return True

    try:
        if _restore_from_token():
            return True
        if supabase_configured():
            ensure_default_user()
    except (SupabaseConfigError, SupabaseRequestError) as exc:
        st.title("投資儀表板")
        st.error(f"Supabase 初始化失敗：{exc}")
        st.stop()

    st.title("投資儀表板")
    st.caption("請先輸入帳號與密碼，通過後才會載入你的持倉與投資資料。")

    if not supabase_configured():
        st.warning(
            "尚未設定 Supabase，目前只允許用本機 secrets 的密碼登入。"
            f"預設帳號為 `{DEFAULT_USERNAME}`，密碼來源為 `{PASSWORD_SECRET_KEY}`。"
        )

    store = _auth_store()
    if store["lockout_until"] > time.time():
        remaining = int(store["lockout_until"] - time.time())
        st.error(f"登入嘗試次數過多，請等待 {remaining} 秒後再試。")
        return False

    with st.form("login_form"):
        username = st.text_input("帳號", value=default_username())
        password = st.text_input("密碼", type="password")
        submitted = st.form_submit_button("登入", type="primary")

    if submitted:
        try:
            user = _login_with_supabase(username.strip(), password)
        except (SupabaseConfigError, SupabaseRequestError) as exc:
            st.error(f"登入失敗：{exc}")
            return False

        if user:
            _set_authenticated_user(user)
            store["failed_attempts"] = 0
            set_auth_token(str(user["username"]))
            st.rerun()

        store["failed_attempts"] += 1
        st.session_state.login_failed = True
        if store["failed_attempts"] >= _MAX_FAILED_ATTEMPTS:
            store["lockout_until"] = time.time() + _LOCKOUT_SECONDS
            store["failed_attempts"] = 0
            st.rerun()

    if st.session_state.get("login_failed", False):
        remaining_attempts = _MAX_FAILED_ATTEMPTS - store["failed_attempts"]
        st.error(f"帳號或密碼錯誤，無法存取儀表板。（還剩 {remaining_attempts} 次機會）")

    _render_create_user()
    return False


def require_password() -> None:
    if not check_password():
        st.stop()
