"""Application configuration for tracked investments and default assumptions."""

from __future__ import annotations

DEFAULT_TICKERS = [
    "VT",
    "VOO",
    "VXUS",
    "QQQ",
    "BND",
    "0050.TW",
    "0056.TW",
]

TICKER_DISPLAY_NAMES = {
    "VT": "Vanguard Total World Stock",
    "VOO": "Vanguard S&P 500",
    "VXUS": "Vanguard Total International Stock",
    "QQQ": "Invesco QQQ Trust",
    "BND": "Vanguard Total Bond Market",
    "0050.TW": "元大台灣卓越50",
    "0056.TW": "元大高股息",
}

CAGR_WINDOWS = {
    "1Y": 1,
    "3Y": 3,
    "5Y": 5,
    "10Y": 10,
    "15Y": 15,
    "20Y": 20,
}

RISK_FREE_RATE = 0.0
CACHE_TTL_QUOTES   = 300     # 即時報價：5 分鐘
CACHE_TTL_HISTORY  = 3600   # 歷史價格／股息：1 小時
CACHE_TTL_PROFILE  = 1800   # 基本面資料：30 分鐘
CACHE_TTL_NEWS     = 1800   # 新聞：30 分鐘
CACHE_TTL_SECONDS  = 1800   # 向後相容保留
TRADING_DAYS = 252
FX_TICKER = "TWD=X"

YAHOO_TW_RSS_FEEDS = {
    "國際財經": "https://tw.stock.yahoo.com/rss?category=intl-markets",
    "基金動態": "https://tw.stock.yahoo.com/rss?category=funds-news",
    "研究報導": "https://tw.stock.yahoo.com/rss?category=research",
    "台股動態": "https://tw.stock.yahoo.com/rss?category=tw-market",
    "最新新聞": "https://tw.stock.yahoo.com/rss?category=news",
}
