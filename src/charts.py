"""Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


TEMPLATE = "plotly_dark"
COLOR_SEQUENCE = [
    "#5eead4",
    "#60a5fa",
    "#f59e0b",
    "#f472b6",
    "#a3e635",
    "#fb7185",
    "#c084fc",
]


def _fmt_percent(value: object) -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{float(value) * 100:,.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def apply_layout(fig: go.Figure, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=TEMPLATE,
        height=height,
        margin=dict(l=30, r=24, t=56, b=36),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="rgba(148, 163, 184, 0.35)",
            font=dict(color="#e5e7eb", size=15),
        ),
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(148, 163, 184, 0.14)")
    return fig


def growth_chart(growth: pd.DataFrame, metrics: pd.DataFrame | None = None, title: str = "成長曲線") -> go.Figure:
    fig = go.Figure()
    if growth.empty:
        fig.add_annotation(
            text="目前沒有可用的歷史價格資料",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#cbd5e1"),
        )
        fig.update_layout(title=title)
        return apply_layout(fig)

    display = growth.loc[:, ~growth.columns.duplicated()].copy()
    for index, column in enumerate(display.columns):
        series = display[column].dropna()
        if series.empty:
            continue
        row = metrics.loc[column] if metrics is not None and column in metrics.index else {}
        hover_values = [
            _fmt_percent(row.get("Total Return") if hasattr(row, "get") else None),
            _fmt_percent(row.get("CAGR 5Y") if hasattr(row, "get") else None),
            _fmt_percent(row.get("Max Drawdown") if hasattr(row, "get") else None),
        ]
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                customdata=[hover_values] * len(series),
                mode="lines",
                name=str(column),
                connectgaps=False,
                line=dict(color=COLOR_SEQUENCE[index % len(COLOR_SEQUENCE)], width=2),
                hovertemplate=(
                    f"<b>{column}</b><br>"
                    "%{x|%Y/%m/%d}, %{y:.0%}<br>"
                    "總報酬率：%{customdata[0]}<br>"
                    "5Y CAGR：%{customdata[1]}<br>"
                    "Max Drawdown：%{customdata[2]}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(title=title)
    fig.update_yaxes(tickformat=".0%")
    return apply_layout(fig)


def comparison_chart(metrics: pd.DataFrame) -> go.Figure:
    display = metrics[["CAGR 1Y", "CAGR 3Y", "CAGR 5Y", "Annualized Volatility"]].copy()
    display = display.reset_index().melt(id_vars="標的", var_name="Metric", value_name="Value")
    fig = px.bar(
        display,
        x="標的",
        y="Value",
        color="Metric",
        barmode="group",
        title="標的比較圖",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_yaxes(tickformat=".1%")
    return apply_layout(fig)


def dividend_chart(annual_dividends: pd.DataFrame) -> go.Figure:
    fig = px.bar(
        annual_dividends,
        x=annual_dividends.index,
        y=annual_dividends.columns,
        title="年度配息圖",
        barmode="group",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_xaxes(type="category")
    return apply_layout(fig)


def allocation_pie(weights: dict[str, float]) -> go.Figure:
    labels = [ticker for ticker, weight in weights.items() if weight > 0]
    values = [weight * 100 for weight in weights.values() if weight > 0]
    if not values:
        fig = go.Figure()
        fig.add_annotation(
            text="目前沒有持有部位",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#cbd5e1"),
        )
        fig.update_layout(title="資產配置比例")
        return apply_layout(fig, height=380)

    fig = px.pie(
        names=labels,
        values=values,
        title="資產配置比例",
        hole=0.48,
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig.update_traces(textposition="inside", textinfo="label+percent")
    return apply_layout(fig, height=380)
