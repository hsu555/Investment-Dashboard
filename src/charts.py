"""Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .config import CAGR_WINDOWS


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
DARK_HOVERLABEL = dict(
    bgcolor="#111827",
    bordercolor="rgba(148, 163, 184, 0.35)",
    font=dict(color="#e5e7eb", size=15),
)


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
        hoverlabel=DARK_HOVERLABEL,
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
    metric_columns = [f"CAGR {label}" for label in CAGR_WINDOWS if f"CAGR {label}" in metrics]
    metric_columns.append("Annualized Volatility")
    display = metrics[metric_columns].copy()
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


def technical_price_chart(indicators: pd.DataFrame, ticker: str, title: str) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.72, 0.28],
    )
    if indicators.empty:
        fig.add_annotation(
            text="目前沒有可用的價格資料",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#cbd5e1"),
        )
        fig.update_layout(title=title)
        return apply_layout(fig, height=620)

    data = indicators.dropna(subset=["Close"]).copy()
    fig.add_trace(
        go.Candlestick(
            x=data.index,
            open=data["Open"] if "Open" in data else data["Close"],
            high=data["High"] if "High" in data else data["Close"],
            low=data["Low"] if "Low" in data else data["Close"],
            close=data["Close"],
            name=ticker,
            increasing_line_color="#22c55e",
            decreasing_line_color="#fb7185",
            hoverlabel=DARK_HOVERLABEL,
            hovertemplate=(
                "<b>%{x|%Y/%m/%d}</b><br>"
                "開：%{open:,.2f}<br>"
                "高：%{high:,.2f}<br>"
                "低：%{low:,.2f}<br>"
                "收：%{close:,.2f}"
                "<extra></extra>"
            ),
        ),
        row=1,
        col=1,
    )

    for column, color in [
        ("MA20", "#60a5fa"),
        ("MA60", "#f59e0b"),
        ("MA240", "#f472b6"),
        ("BB Upper", "rgba(148, 163, 184, 0.55)"),
        ("BB Lower", "rgba(148, 163, 184, 0.55)"),
    ]:
        if column in data and data[column].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=data[column],
                    mode="lines",
                    name=column,
                    line=dict(color=color, width=1.5, dash="dot" if column.startswith("BB") else "solid"),
                    hoverlabel=DARK_HOVERLABEL,
                    hovertemplate=f"<b>{column}</b><br>%{{x|%Y/%m/%d}}<br>%{{y:,.2f}}<extra></extra>",
                ),
                row=1,
                col=1,
            )

    if "Volume" in data:
        colors = ["#22c55e" if row.Close >= row.Open else "#fb7185" for row in data.itertuples()]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["Volume"],
                name="Volume",
                marker_color=colors,
                opacity=0.42,
                hoverlabel=DARK_HOVERLABEL,
                hovertemplate="<b>成交量</b><br>%{x|%Y/%m/%d}<br>%{y:,.0f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    fig.update_yaxes(title_text="價格", row=1, col=1)
    fig.update_yaxes(title_text="成交量", row=2, col=1)
    return apply_layout(fig, height=620)


def momentum_chart(indicators: pd.DataFrame, title: str = "動能指標") -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("RSI14", "MACD"),
    )
    if indicators.empty:
        fig.add_annotation(
            text="目前沒有可用的技術指標",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color="#cbd5e1"),
        )
        fig.update_layout(title=title)
        return apply_layout(fig, height=460)

    data = indicators.copy()
    if "RSI14" in data:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["RSI14"],
                mode="lines",
                name="RSI14",
                line=dict(color="#5eead4"),
                hoverlabel=DARK_HOVERLABEL,
                hovertemplate="<b>RSI14</b><br>%{x|%Y/%m/%d}<br>%{y:,.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dot", line_color="#fb7185", row=1, col=1)
        fig.add_hline(y=30, line_dash="dot", line_color="#22c55e", row=1, col=1)

    if "MACD" in data:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD"],
                mode="lines",
                name="MACD",
                line=dict(color="#60a5fa"),
                hoverlabel=DARK_HOVERLABEL,
                hovertemplate="<b>MACD</b><br>%{x|%Y/%m/%d}<br>%{y:,.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if "MACD Signal" in data:
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data["MACD Signal"],
                mode="lines",
                name="Signal",
                line=dict(color="#f59e0b"),
                hoverlabel=DARK_HOVERLABEL,
                hovertemplate="<b>Signal</b><br>%{x|%Y/%m/%d}<br>%{y:,.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
    if "MACD Hist" in data:
        colors = ["#22c55e" if value >= 0 else "#fb7185" for value in data["MACD Hist"].fillna(0)]
        fig.add_trace(
            go.Bar(
                x=data.index,
                y=data["MACD Hist"],
                name="Hist",
                marker_color=colors,
                opacity=0.46,
                hoverlabel=DARK_HOVERLABEL,
                hovertemplate="<b>Hist</b><br>%{x|%Y/%m/%d}<br>%{y:,.2f}<extra></extra>",
            ),
            row=2,
            col=1,
        )

    fig.update_layout(title=title)
    return apply_layout(fig, height=460)
