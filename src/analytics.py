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


def technical_indicators(history: pd.DataFrame) -> pd.DataFrame:
    """Append common technical indicators to an OHLCV frame."""
    if history.empty or "Close" not in history:
        return pd.DataFrame()

    result = history.copy()
    close = result["Close"]
    high = result["High"] if "High" in result else close
    low = result["Low"] if "Low" in result else close

    for window in (5, 20, 60, 120, 240):
        result[f"MA{window}"] = close.rolling(window).mean()

    ma20 = result["MA20"]
    std20 = close.rolling(20).std()
    result["BB Upper"] = ma20 + 2 * std20
    result["BB Lower"] = ma20 - 2 * std20

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    result["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    result["MACD"] = ema12 - ema26
    result["MACD Signal"] = result["MACD"].ewm(span=9, adjust=False).mean()
    result["MACD Hist"] = result["MACD"] - result["MACD Signal"]

    low9 = low.rolling(9).min()
    high9 = high.rolling(9).max()
    rsv = (close - low9) / (high9 - low9).replace(0, np.nan) * 100
    result["K"] = rsv.ewm(alpha=1 / 3, adjust=False).mean()
    result["D"] = result["K"].ewm(alpha=1 / 3, adjust=False).mean()

    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["ATR14"] = true_range.rolling(14).mean()
    result["Volume MA20"] = result["Volume"].rolling(20).mean() if "Volume" in result else np.nan
    return result


def resample_ohlcv(history: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Resample daily OHLCV data to weekly/monthly/yearly bars."""
    if history.empty:
        return history

    columns = {
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }
    available = {column: method for column, method in columns.items() if column in history}
    return history.resample(frequency).agg(available).dropna(subset=["Close"])


def technical_signal_table(indicators: pd.DataFrame) -> pd.DataFrame:
    """Build a compact signal table from the latest technical values."""
    clean = indicators.dropna(subset=["Close"])
    if clean.empty:
        return pd.DataFrame()

    latest = clean.iloc[-1]
    previous = clean.iloc[-2] if len(clean) > 1 else latest
    rows = []

    def add(group: str, item: str, value: object, signal: str, note: str) -> None:
        rows.append({"類別": group, "指標": item, "數值": value, "訊號": signal, "說明": note})

    close = latest.get("Close")
    ma20 = latest.get("MA20")
    ma60 = latest.get("MA60")
    ma240 = latest.get("MA240")
    trend_signal = (
        "多頭"
        if pd.notna(close) and pd.notna(ma20) and pd.notna(ma60) and close > ma20 > ma60
        else "空頭"
        if pd.notna(close) and pd.notna(ma20) and pd.notna(ma60) and close < ma20 < ma60
        else "盤整"
    )
    add("趨勢", "價格 / MA20 / MA60", close, trend_signal, "收盤價位於主要均線的相對位置")
    ma240_signal = "長線資料不足"
    if pd.notna(close) and pd.notna(ma240):
        ma240_signal = "長線偏多" if close > ma240 else "長線偏弱"
    add("趨勢", "MA240", ma240, ma240_signal, "價格與年線位置比較")

    rsi = latest.get("RSI14")
    if pd.notna(rsi):
        rsi_signal = "超買" if rsi >= 70 else "超賣" if rsi <= 30 else "中性"
        add("動能", "RSI14", rsi, rsi_signal, "70 以上偏熱，30 以下偏弱")

    macd = latest.get("MACD")
    macd_signal = latest.get("MACD Signal")
    prev_macd = previous.get("MACD")
    prev_signal = previous.get("MACD Signal")
    if pd.notna(macd) and pd.notna(macd_signal):
        cross = "黃金交叉" if prev_macd <= prev_signal and macd > macd_signal else "死亡交叉" if prev_macd >= prev_signal and macd < macd_signal else "多方" if macd > macd_signal else "空方"
        add("動能", "MACD", macd - macd_signal, cross, "MACD 與 Signal 線差距")

    k_value = latest.get("K")
    d_value = latest.get("D")
    if pd.notna(k_value) and pd.notna(d_value):
        add("動能", "KD", k_value - d_value, "偏多" if k_value > d_value else "偏空", "K 值高於 D 值代表短線動能較強")

    upper = latest.get("BB Upper")
    lower = latest.get("BB Lower")
    if pd.notna(upper) and pd.notna(lower) and upper != lower:
        band_position = (close - lower) / (upper - lower)
        band_signal = "貼近上緣" if band_position >= 0.8 else "貼近下緣" if band_position <= 0.2 else "區間內"
        add("波動", "布林位置", band_position, band_signal, "價格在布林通道中的位置")

    atr = latest.get("ATR14")
    if pd.notna(atr) and close:
        add("波動", "ATR14 / 收盤價", atr / close, "波動偏高" if atr / close >= 0.04 else "波動正常", "近 14 期平均真實波幅")

    volume = latest.get("Volume")
    volume_ma = latest.get("Volume MA20")
    if pd.notna(volume) and pd.notna(volume_ma) and volume_ma:
        add("量能", "成交量 / 20MA", volume / volume_ma, "放量" if volume > volume_ma * 1.3 else "量縮" if volume < volume_ma * 0.7 else "正常", "成交量相對 20 期平均")

    return pd.DataFrame(rows)
