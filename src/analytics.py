"""Investment analytics and return metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .config import CAGR_WINDOWS, TRADING_DAYS


def growth_curve(prices: pd.DataFrame) -> pd.DataFrame:
    clean = prices.dropna(how="all").ffill()
    if clean.empty:
        return pd.DataFrame()
    first_valid = clean.apply(lambda col: col.loc[col.first_valid_index()] if col.first_valid_index() else np.nan)
    return clean.divide(first_valid).dropna(how="all")


def total_return(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    return float(clean.iloc[-1] / clean.iloc[0] - 1)


def cagr(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    years = (clean.index[-1] - clean.index[0]).days / 365.25
    if years <= 0 or clean.iloc[0] <= 0:
        return None
    return float((clean.iloc[-1] / clean.iloc[0]) ** (1 / years) - 1)


def cagr_windows(series: pd.Series) -> dict[str, float | None]:
    clean = series.dropna()
    if clean.empty:
        return {label: None for label in CAGR_WINDOWS}

    end = clean.index[-1]
    results: dict[str, float | None] = {}
    for label, years in CAGR_WINDOWS.items():
        start = end - pd.DateOffset(years=years)
        window = clean.loc[clean.index >= start]
        results[label] = cagr(window) if len(window) > 1 else None
    return results


def max_drawdown(series: pd.Series) -> float | None:
    clean = series.dropna()
    if len(clean) < 2:
        return None
    running_max = clean.cummax()
    drawdown = clean / running_max - 1
    return float(drawdown.min())


def annualized_volatility(series: pd.Series) -> float | None:
    returns = series.pct_change(fill_method=None).dropna()
    if returns.empty:
        return None
    value = returns.std() * math.sqrt(TRADING_DAYS)
    return float(value) if pd.notna(value) else None


def metrics_table(prices: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in prices.columns:
        series = prices[ticker]
        cagr_values = cagr_windows(series)
        rows.append(
            {
                "標的": ticker,
                "Total Return": total_return(series),
                "CAGR 1Y": cagr_values["1Y"],
                "CAGR 3Y": cagr_values["3Y"],
                "CAGR 5Y": cagr_values["5Y"],
                "Max Drawdown": max_drawdown(series),
                "Annualized Volatility": annualized_volatility(series),
            }
        )
    return pd.DataFrame(rows).set_index("標的")


def yearly_dividends(dividends: dict[str, pd.Series]) -> pd.DataFrame:
    frames = []
    for ticker, series in dividends.items():
        if series.empty:
            continue
        annual = series.groupby(series.index.year).sum().rename(ticker)
        frames.append(annual)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, axis=1, sort=False).fillna(0).sort_index()
