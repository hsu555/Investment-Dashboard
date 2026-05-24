# 投資儀表板

使用 Python、Streamlit、yfinance 與 Plotly 建立的投資儀表板。可追蹤 Yahoo Finance 支援的股票、基金與其他商品代號。預設追蹤：

- VT
- VOO
- VXUS
- QQQ
- BND
- 0050.TW
- 0056.TW

## 功能

- 自訂追蹤代號
- 持倉數量與買入價儲存
- 美元標的依 USD/TWD 匯率換算，市值、成本、損益與配置比例統一以台幣計算
- 依目前台幣市值自動計算配置比例
- 未購入標的可用數量 0 加入觀察
- 即時價格與日漲跌
- 總報酬率
- 1 年、3 年、5 年 CAGR
- 最大回撤（Max Drawdown）
- 年化波動率
- 配息資訊與年度配息圖
- 資產配置比例與圓餅圖
- 目標配置與再平衡建議，可估算需調整金額與股數
- 交易紀錄 / 現金流流水帳，可從目前持倉建立初始買入紀錄
- 投資組合層級風險分析，包含年化報酬、波動、最大回撤、Sharpe 與相關係數
- 美元 / 台幣匯率（Yahoo Finance `TWD=X`）
- Yahoo奇摩股市繁體中文財經新聞摘要

資料每次開啟頁面會更新，並以 Streamlit 快取 30 分鐘，避免頻繁打 API。新聞來源使用 Yahoo奇摩股市 RSS。

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 啟動

```bash
streamlit run app.py
```

開啟瀏覽器中的本機網址後即可使用。

## Supabase 與登入

儀表板啟動後會先要求輸入帳號與密碼，通過後才會載入該使用者的持倉、觀察清單與退休試算參數。

先到 Supabase SQL Editor 執行 `supabase_schema.sql` 建立資料表。接著在 Streamlit Secrets 設定：

本機開發時可建立 `.streamlit/secrets.toml`：

```toml
supabase_url = "https://你的專案.supabase.co"
supabase_service_role_key = "你的 service_role key"
dashboard_password = "原本的儀表板密碼"
default_username = "hsu555"
```

第一次連上 Supabase 時，系統會自動建立預設使用者 `hsu555`，密碼使用 `dashboard_password`，並把現有 `portfolio.json` 的持股與觀察清單匯入這個帳號。後續新增使用者可在登入畫面的「新增使用者」建立。

若是從舊版升級，請重新執行 `supabase_schema.sql`，新增 `target_allocations` 與 `transactions` 兩張表，才能在 Supabase 儲存目標配置與交易紀錄。本機開發未設定 Supabase 時，這兩項會分別寫入 `target_allocations.json` 與 `transactions.json`。

`.streamlit/secrets.toml` 已在 `.gitignore` 中，不會提交到 GitHub。部署到 Streamlit Community Cloud 時，請到 app settings 的 Secrets 貼上同樣內容。請使用 Supabase Project Settings > API 的 `service_role` key，不需要提供 Supabase 帳號密碼，也不需要直接提供 Postgres 密碼。

## 專案結構

```text
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
└── src
    ├── __init__.py
    ├── analytics.py
    ├── charts.py
    ├── config.py
    ├── data.py
    └── formatting.py
```

## 擴充方式

### 新增追蹤標的

到 `src/config.py` 修改：

- `DEFAULT_TICKERS`
- `TICKER_DISPLAY_NAMES`

也可以在側欄的持倉表格直接新增 Yahoo Finance 代號，例如 `AAPL`、`TSLA`、`2330.TW`。數量填 `0` 時會作為觀察標的，不列入資產配置比例。

### 儲存持倉

側欄的「儲存持倉」會將標的、數量與買入價寫入目前登入使用者的 Supabase 資料。未設定 Supabase 時，開發環境仍會暫時寫入 `portfolio.json`。買入價請輸入該標的原幣別價格；例如美股輸入美元價格，台股輸入台幣價格。

### 新增策略回測

建議新增 `src/backtesting.py`，將策略訊號、再平衡規則、交易成本與績效統計拆成獨立函式，再由 `app.py` 新增 Streamlit 分頁呼叫。

### 新增退休模擬

建議新增 `src/retirement.py`，放入提領率、通膨、投資年限、蒙地卡羅模擬等邏輯，再由 UI 提供輸入參數與結果圖表。

## 注意事項

- yfinance/Yahoo Finance 資料可能有延遲或短暫缺漏。
- 部分標的的配息、幣別與 Yahoo Finance 欄位有時不完整，儀表板會以 `N/A` 顯示缺漏資料。
- 本工具僅供投資分析與視覺化參考，不構成投資建議。
