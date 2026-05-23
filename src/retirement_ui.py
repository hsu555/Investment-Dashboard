from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.retirement import MonteCarloResult, RetirementInputs, run_monte_carlo
from src.supabase_store import (
    SupabaseConfigError,
    SupabaseRequestError,
    load_retirement_settings,
    save_retirement_settings,
    supabase_configured,
)


def _fmt_wan(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f} 萬"


def _default_inputs(current_assets_wan: float) -> RetirementInputs:
    return RetirementInputs(
        current_age=49,
        retirement_age=55,
        life_expectancy=90,
        current_assets_wan=float(current_assets_wan),
        monthly_contribution_wan=2.0,
        monthly_expense_wan=5.0,
        mean_annual_return=0.07,
        annual_return_std=0.15,
        inflation_rate=0.03,
        n_simulations=1000,
    )


def _inputs_from_settings(settings: dict | None, current_assets_wan: float) -> RetirementInputs:
    defaults = _default_inputs(current_assets_wan)
    if not settings:
        return defaults
    return RetirementInputs(
        current_age=int(settings.get("current_age") or defaults.current_age),
        retirement_age=int(settings.get("retirement_age") or defaults.retirement_age),
        life_expectancy=int(settings.get("life_expectancy") or defaults.life_expectancy),
        current_assets_wan=float(settings.get("current_assets_wan") or defaults.current_assets_wan),
        monthly_contribution_wan=float(settings.get("monthly_contribution_wan") or defaults.monthly_contribution_wan),
        monthly_expense_wan=float(settings.get("monthly_expense_wan") or defaults.monthly_expense_wan),
        mean_annual_return=float(settings.get("mean_annual_return") or defaults.mean_annual_return),
        annual_return_std=float(settings.get("annual_return_std") or defaults.annual_return_std),
        inflation_rate=float(settings.get("inflation_rate") or defaults.inflation_rate),
        n_simulations=int(settings.get("n_simulations") or defaults.n_simulations),
    )


def _load_saved_inputs(user_id: str | None, current_assets_wan: float) -> RetirementInputs:
    if user_id and supabase_configured():
        try:
            return _inputs_from_settings(load_retirement_settings(user_id), current_assets_wan)
        except (SupabaseConfigError, SupabaseRequestError) as exc:
            st.error(f"無法從 Supabase 載入退休試算參數：{exc}")
            st.stop()
    return _default_inputs(current_assets_wan)


def _ensure_retirement_form_state(defaults: RetirementInputs) -> None:
    initial_values = {
        "retirement_current_age": defaults.current_age,
        "retirement_retirement_age": defaults.retirement_age,
        "retirement_life_expectancy": defaults.life_expectancy,
        "retirement_current_assets_wan": defaults.current_assets_wan,
        "retirement_monthly_contribution_wan": defaults.monthly_contribution_wan,
        "retirement_monthly_expense_wan": defaults.monthly_expense_wan,
        "retirement_mean_return_pct": defaults.mean_annual_return * 100,
        "retirement_return_std_pct": defaults.annual_return_std * 100,
        "retirement_inflation_pct": defaults.inflation_rate * 100,
        "retirement_n_simulations": defaults.n_simulations,
    }
    for key, value in initial_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _validate_inputs(inputs: RetirementInputs) -> str | None:
    if inputs.retirement_age <= inputs.current_age:
        return "預計退休年齡必須大於目前年齡。"
    if inputs.life_expectancy <= inputs.retirement_age:
        return "預計試算年齡必須大於預計退休年齡。"
    return None


def _render_inputs(defaults: RetirementInputs) -> RetirementInputs:
    _ensure_retirement_form_state(defaults)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**個人設定**")
        current_age = int(st.number_input("目前年齡", min_value=18, max_value=80, step=1, key="retirement_current_age"))
        retirement_age = int(st.number_input("預計退休年齡", min_value=19, max_value=90, step=1, key="retirement_retirement_age"))
        life_expectancy = int(st.number_input("預計試算年齡", min_value=20, max_value=120, step=1, key="retirement_life_expectancy"))

    with col2:
        st.markdown("**財務現況（萬元）**")
        assets = st.number_input("現有投資資產（萬元）", min_value=0.0, step=10.0, key="retirement_current_assets_wan")
        monthly_contribution = st.number_input("每月定期投入（萬元）", min_value=0.0, step=0.5, key="retirement_monthly_contribution_wan")

    with col3:
        st.markdown("**退休後需求（萬元）**")
        monthly_expense = st.number_input("退休後每月支出（萬元）", min_value=0.0, step=0.5, key="retirement_monthly_expense_wan")

    st.markdown("**模擬參數**")
    pcol1, pcol2, pcol3, pcol4 = st.columns(4)
    with pcol1:
        mean_return = st.number_input(
            "年化報酬率(%)", min_value=0.0, max_value=30.0, step=0.5, format="%.1f", key="retirement_mean_return_pct"
        )
    with pcol2:
        return_std = st.number_input(
            "報酬率波動幅度 (%)", min_value=1.0, max_value=50.0, step=1.0, format="%.1f",
            help="報酬率的標準差（Std Dev）。數值越大代表好年份漲更多、壞年份跌更深。全球股市歷史約 15–20%，債券約 5–8%。",
            key="retirement_return_std_pct",
        )
    with pcol3:
        inflation = st.number_input(
            "通膨率 (%)", min_value=0.0, max_value=10.0, step=0.5, format="%.1f", key="retirement_inflation_pct"
        )
    with pcol4:
        n_sim = int(st.number_input("模擬次數", min_value=100, max_value=5000, step=100, key="retirement_n_simulations"))

    return RetirementInputs(
        current_age=current_age,
        retirement_age=retirement_age,
        life_expectancy=life_expectancy,
        current_assets_wan=assets,
        monthly_contribution_wan=monthly_contribution,
        monthly_expense_wan=monthly_expense,
        mean_annual_return=mean_return / 100,
        annual_return_std=return_std / 100,
        inflation_rate=inflation / 100,
        n_simulations=n_sim,
    )


def _render_metrics(result: MonteCarloResult, inputs: RetirementInputs) -> None:
    sr_pct = result.success_rate * 100

    if sr_pct >= 90:
        delta_str, delta_color = "達標", "normal"
    elif sr_pct >= 75:
        delta_str, delta_color = "偏低", "off"
    else:
        delta_str, delta_color = "風險偏高", "inverse"

    cols = st.columns(4)
    cols[0].metric("成功率", f"{sr_pct:.1f}%", delta=delta_str, delta_color=delta_color)
    cols[1].metric(f"退休資產中位數（{inputs.retirement_age}歲）", _fmt_wan(result.median_at_retirement))
    cols[2].metric("悲觀情境 P10", _fmt_wan(result.p10_at_retirement))
    cols[3].metric("樂觀情境 P90", _fmt_wan(result.p90_at_retirement))

    if sr_pct >= 90:
        st.success(
            f"✅ 資產可支撐至 {inputs.life_expectancy} 歲的機率為 {sr_pct:.1f}%"
            f"（{result.n_simulations:,} 次模擬，{result.n_simulations - result.depleted_count:,} 次達標）"
        )
    elif sr_pct >= 75:
        st.warning(
            f"⚠️ 成功率 {sr_pct:.1f}%，建議提高每月投入、延後退休或降低支出。"
            f"（{result.depleted_count:,}/{result.n_simulations:,} 次模擬資產耗盡）"
        )
    else:
        st.error(
            f"❌ 成功率僅 {sr_pct:.1f}%，退休計劃需大幅調整。"
            f"（{result.depleted_count:,}/{result.n_simulations:,} 次模擬資產耗盡）"
        )


def _render_fan_chart(result: MonteCarloResult, inputs: RetirementInputs) -> None:
    paths = result.percentile_paths
    ages = result.ages

    fig = go.Figure()

    # Outer band P10–P90 (light fill)
    fig.add_trace(go.Scatter(
        x=ages, y=paths[90],
        line=dict(width=0, color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=paths[10], name="P10–P90 區間",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(34,197,94,0.10)",
        hoverinfo="skip",
    ))

    # Inner band P25–P75 (medium fill)
    fig.add_trace(go.Scatter(
        x=ages, y=paths[75],
        line=dict(width=0, color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=paths[25], name="P25–P75 區間",
        line=dict(width=0, color="rgba(0,0,0,0)"),
        fill="tonexty", fillcolor="rgba(96,165,250,0.18)",
        hoverinfo="skip",
    ))

    # Median line
    fig.add_trace(go.Scatter(
        x=ages, y=paths[50], name="中位數 P50",
        line=dict(color="#60a5fa", width=2.5),
    ))

    # Boundary reference lines
    fig.add_trace(go.Scatter(
        x=ages, y=paths[10], name="P10 悲觀",
        line=dict(color="#fb7185", width=1, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=ages, y=paths[90], name="P90 樂觀",
        line=dict(color="#22c55e", width=1, dash="dot"),
    ))

    fig.add_vline(
        x=inputs.retirement_age,
        line_dash="dot", line_color="#94a3b8",
        annotation_text=f"退休 {inputs.retirement_age} 歲",
        annotation_position="top right",
    )
    fig.update_layout(
        title=f"Monte Carlo 資產路徑扇形圖（{result.n_simulations:,} 次模擬）",
        xaxis_title="年齡",
        yaxis_title="資產（萬元）",
        plot_bgcolor="#0b1020",
        paper_bgcolor="#0b1020",
        font_color="#e5e7eb",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _render_percentile_table(result: MonteCarloResult, inputs: RetirementInputs) -> None:
    paths = result.percentile_paths
    retirement_idx = inputs.retirement_age - inputs.current_age

    rows = []
    for p, label in [(10, "P10 悲觀"), (25, "P25"), (50, "P50 中位"), (75, "P75"), (90, "P90 樂觀")]:
        rows.append({
            "百分位": label,
            f"退休時（{inputs.retirement_age}歲）": round(paths[p][retirement_idx]),
            f"試算終點（{inputs.life_expectancy}歲）": round(paths[p][-1]),
        })

    df = pd.DataFrame(rows).set_index("百分位")
    st.dataframe(df.style.format("{:,.0f}"), use_container_width=True)


def _render_yearly_table(result: MonteCarloResult, inputs: RetirementInputs) -> None:
    paths = result.percentile_paths
    ages = result.ages
    accum_years = inputs.retirement_age - inputs.current_age
    adj_expense = inputs.monthly_expense_wan * (1 + inputs.inflation_rate) ** accum_years
    g = inputs.inflation_rate
    annual_contrib = inputs.monthly_contribution_wan * 12
    base_withdrawal = adj_expense * 12

    rows = []
    for i, age in enumerate(ages):
        phase = "累積期" if age < inputs.retirement_age else "提領期"

        if i == 0:
            cash_flow = float("nan")
        elif age <= inputs.retirement_age:
            cash_flow = annual_contrib
        else:
            retire_yr = age - inputs.retirement_age - 1
            cash_flow = -(base_withdrawal * (1 + g) ** retire_yr)

        rows.append({
            "年齡": age,
            "階段": phase,
            "年度現金流（萬）": cash_flow,
            "月均（萬）": float("nan") if i == 0 else cash_flow / 12,
            "存活率": result.survival_rates[i] * 100,
            "P10 悲觀": round(paths[10][i]),
            "P25": round(paths[25][i]),
            "P50 中位": round(paths[50][i]),
            "P75": round(paths[75][i]),
            "P90 樂觀": round(paths[90][i]),
        })

    df = pd.DataFrame(rows).set_index("年齡")

    def style_cashflow(val):
        try:
            v = float(val)
            if pd.isna(v):
                return ""
            return "color: #22c55e" if v > 0 else "color: #fb7185"
        except (TypeError, ValueError):
            return ""

    def style_survival(val):
        try:
            v = float(val)
            if v >= 90:
                return "color: #22c55e"
            if v >= 75:
                return "color: #fbbf24"
            return "color: #fb7185"
        except (TypeError, ValueError):
            return ""

    def style_zero(val):
        try:
            if float(val) == 0:
                return "color: #fb7185; font-weight: bold"
        except (TypeError, ValueError):
            pass
        return ""

    nan_dash = lambda v: "—" if pd.isna(v) else f"{v:+,.1f}"

    p50_style = "background-color: rgba(96,165,250,0.15); font-weight: bold"

    try:
        formatted = (
            df.style
            .format({
                "年度現金流（萬）": nan_dash,
                "月均（萬）": nan_dash,
                "存活率": "{:.1f}%",
                "P10 悲觀": "{:,.0f}",
                "P25": "{:,.0f}",
                "P50 中位": "{:,.0f}",
                "P75": "{:,.0f}",
                "P90 樂觀": "{:,.0f}",
            })
            .map(style_cashflow, subset=["年度現金流（萬）", "月均（萬）"])
            .map(style_survival, subset=["存活率"])
            .map(style_zero, subset=["P10 悲觀", "P25", "P50 中位"])
            .map(lambda _: p50_style, subset=["P50 中位"])
        )
    except AttributeError:
        formatted = (
            df.style
            .format({
                "年度現金流（萬）": nan_dash,
                "月均（萬）": nan_dash,
                "存活率": "{:.1f}%",
                "P10 悲觀": "{:,.0f}",
                "P25": "{:,.0f}",
                "P50 中位": "{:,.0f}",
                "P75": "{:,.0f}",
                "P90 樂觀": "{:,.0f}",
            })
            .applymap(style_cashflow, subset=["年度現金流（萬）", "月均（萬）"])
            .applymap(style_survival, subset=["存活率"])
            .applymap(style_zero, subset=["P10 悲觀", "P25", "P50 中位"])
            .applymap(lambda _: p50_style, subset=["P50 中位"])
        )

    st.dataframe(formatted, use_container_width=True, height=600)


def render_retirement_view(total_market_value_twd: float | None) -> None:
    st.subheader("退休金試算（Monte Carlo）")
    st.caption(
        "以 Monte Carlo 隨機模擬，每年報酬率從常態分佈抽樣（年化報酬率 ± 波動幅度），"
        "統計多個情境下資產能否支撐至試算年齡，評估退休計劃的成功機率。"
        "每次點擊「開始試算」會產生新的隨機模擬結果。"
    )

    current_assets_wan = round((total_market_value_twd or 0) / 10000)
    current_user = st.session_state.get("current_user", {})
    user_id = current_user.get("id")
    if "retirement_inputs" not in st.session_state:
        st.session_state.retirement_inputs = _load_saved_inputs(user_id, current_assets_wan)

    with st.expander("試算參數設定", expanded=True):
        with st.form("retirement_form"):
            inputs = _render_inputs(st.session_state.retirement_inputs)
            submitted = st.form_submit_button("開始試算", type="primary", use_container_width=True)

    if submitted:
        validation_error = _validate_inputs(inputs)
        if validation_error:
            st.error(validation_error)
            return
        st.session_state.retirement_inputs = inputs
        if user_id and supabase_configured():
            try:
                save_retirement_settings(user_id, inputs)
            except (SupabaseConfigError, SupabaseRequestError) as exc:
                st.warning(f"試算完成，但退休參數未能存入 Supabase：{exc}")

    display_inputs = st.session_state.retirement_inputs

    if submitted or "retirement_result" not in st.session_state:
        with st.spinner("Monte Carlo 模擬中..."):
            st.session_state.retirement_result = run_monte_carlo(display_inputs)

    result: MonteCarloResult = st.session_state.retirement_result

    st.divider()
    _render_metrics(result, display_inputs)

    st.divider()
    _render_fan_chart(result, display_inputs)

    st.divider()
    st.markdown("##### 各百分位資產試算（萬元）")
    st.caption("P10/P90 代表悲觀/樂觀極端情境；P50 為中位數情境。金額單位：萬元。")
    _render_percentile_table(result, display_inputs)

    st.divider()
    st.markdown("##### 逐年詳細試算")
    st.caption(
        "年度現金流：累積期為定期投入（正值），提領期為計畫提領額（負值，每年隨通膨遞增）。"
        "存活率：該年仍有正資產的模擬比例，跌至 75% 以下顯示橘色警示，跌至 0 顯示紅色。"
        "P值欄為資產水位（萬元），歸零以紅色標示。"
    )
    _render_yearly_table(result, display_inputs)
