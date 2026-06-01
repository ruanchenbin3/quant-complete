---
name: quant-trading
description: "📊 完整量化交易技能包 — 数据获取、技术分析、ML模型、回测、风险管理、投资组合优化、自动化交易"
version: 2.0.0
author: agent
tags: [quant, trading, finance, ml, backtesting, risk-management, portfolio, automation]
platforms: [windows, linux, macos]
---

# 📊 量化交易完整技能包

一个技能覆盖量化交易全流程。安装了它，你不需要其他量化技能。

---

## 快速开始

```python
# 1. 安装依赖
pip install yfinance pandas numpy matplotlib scikit-learn ta

# 2. 获取数据并分析
from quant_agent.data.fetcher import StockDataFetcher
fetcher = StockDataFetcher()
df = fetcher.fetch("BTC-USD", period="1y")

# 3. 跑技术指标
from quant_agent.analysis.indicators import add_all_indicators
df = add_all_indicators(df)
print(df[["close","rsi_14","macd"]].tail())

# 4. 回测一个策略
from quant_agent.analysis.backtest import BacktestEngine
from quant_agent.strategies.examples import sma_cross
engine = BacktestEngine(df, sma_cross)
result = engine.run()
print(f"收益率: {result.total_return}%  夏普: {result.sharpe_ratio}")

# 5. ML 预测
from quant_agent.ml.pipeline import MLPipeline
pipeline = MLPipeline()
pipeline.run("BTC-USD", model_type="random_forest", period="2y")
pipeline.report()

# 6. 投资组合优化
from quant_agent.ml.portfolio import PortfolioOptimizer
opt = PortfolioOptimizer()
weights = opt.risk_parity(returns_df)
```

---

## 一、市场数据获取

### 美股/全球股票 (yfinance)
```python
import yfinance as yf

# 单只
df = yf.download("AAPL", period="1y", interval="1d")
# period: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,max
# interval: 1m,2m,5m,15m,30m,60m,1d,1wk,1mo

# 多只
df = yf.download(["AAPL", "MSFT", "GOOGL"], period="6mo")

# A股
df_sh = yf.download("600519.SS")  # 上证: .SS
df_sz = yf.download("000858.SZ")  # 深证: .SZ
```

### 加密货币 (CCXT)
```python
import ccxt, pandas as pd
exchange = ccxt.binance()
ohlcv = exchange.fetch_ohlcv("BTC/USDT", "1d", limit=500)
df = pd.DataFrame(ohlcv, columns=["timestamp","open","high","low","close","volume"])
df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
df.set_index("timestamp", inplace=True)
```

### 数据清洗
```python
df = df.fillna(method="ffill")         # 前向填充缺失
df = df[~df.index.duplicated()]        # 去重
df = df.sort_index()                   # 排序
returns = df["Close"].pct_change()
df = df[abs(returns) < 0.2]            # 过滤异常值
```

---

## 二、技术指标（纯 Python 实现）

### 趋势指标
| 指标 | 参数 | 用法 |
|------|------|------|
| SMA | window=20 | 简单移动平均 |
| EMA | span=20 | 指数移动平均 |
| MACD | 12/26/9 | MACD>信号线=多头 |

```python
def sma(s, w): return s.rolling(w).mean()
def ema(s, sp): return s.ewm(span=sp, adjust=False).mean()

# MACD
ema_fast = ema(df["Close"], 12)
ema_slow = ema(df["Close"], 26)
df["macd"] = ema_fast - ema_slow
df["macd_signal"] = ema(df["macd"], 9)
df["macd_hist"] = df["macd"] - df["macd_signal"]
```

### 震荡指标
```python
# RSI
delta = df["Close"].diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = -delta.clip(upper=0).rolling(14).mean()
df["rsi14"] = 100 - 100 / (1 + gain / loss.replace(0, 1e-10))
# >70 超买, <30 超卖

# KDJ
low_min = df["Low"].rolling(9).min()
high_max = df["High"].rolling(9).max()
rsv = (df["Close"] - low_min) / (high_max - low_min) * 100
df["k"] = rsv.ewm(com=2).mean()
df["d"] = df["k"].ewm(com=2).mean()
df["j"] = 3 * df["k"] - 2 * df["d"]
```

### 波动率指标
```python
# 布林带
df["bb_mid"] = df["Close"].rolling(20).mean()
df["bb_std"] = df["Close"].rolling(20).std()
df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]

# ATR
tr = pd.concat([
    df["High"] - df["Low"],
    (df["High"] - df["Close"].shift(1)).abs(),
    (df["Low"] - df["Close"].shift(1)).abs(),
], axis=1).max(axis=1)
df["atr14"] = tr.rolling(14).mean()
```

---

## 三、策略类型

### 趋势跟踪 (适合牛市)
```python
def sma_cross(df, fast=20, slow=50):
    s = pd.Series(0, index=df.index)
    s[(df["Close"].rolling(fast).mean() > df["Close"].rolling(slow).mean())] = 1
    return s
```

### 均值回归 (适合震荡市)
```python
def rsi_reversal(df, oversold=30, overbought=70):
    s = pd.Series(0, index=df.index)
    s[df["rsi14"] < oversold] = 1
    s[df["rsi14"] > overbought] = -1
    return s
```

### 带趋势过滤 (保守)
```python
def safe_trend(df):
    s20 = df["Close"].rolling(20).mean()
    s50 = df["Close"].rolling(50).mean()
    s200 = df["Close"].rolling(200).mean()
    s = pd.Series(0, index=df.index)
    s[(s20 > s50) & (s200 > s200.shift(20))] = 1
    return s
```

---

## 四、回测

```python
def backtest(df, strategy_fn, initial=100000, commission=0.001):
    df = df.copy()
    df["signal"] = strategy_fn(df)
    df["position"] = df["signal"].shift(1)  # 防未来函数
    df["returns"] = df["Close"].pct_change()
    df["strategy"] = df["position"] * df["returns"]
    df["strategy"] -= commission * df["position"].diff().abs()
    equity = initial * (1 + df["strategy"]).cumprod()

    total = (equity.iloc[-1] / initial - 1) * 100
    years = len(df) / 252
    annual = (1 + total/100) ** (1/years) - 1 if years > 0 else 0
    rolling_max = equity.cummax()
    mdd = ((equity - rolling_max) / rolling_max * 100).min()
    sr = df["strategy"].dropna()
    sharpe = (252 ** 0.5) * sr.mean() / sr.std() if sr.std() > 0 else 0

    return {
        "total_return": round(total, 2),
        "annual_return": round(annual * 100, 2),
        "max_drawdown": round(mdd, 2),
        "sharpe_ratio": round(sharpe, 2),
        "final_equity": round(equity.iloc[-1], 2),
    }
```

### 指标解读
| 指标 | 好 | 一般 | 差 |
|------|----|------|----|
| 夏普 | >2.0 | 1-2 | <1 |
| 回撤 | <10% | 10-25% | >25% |
| 胜率 | >55% | 40-55% | <40% |

### 五大陷阱
1. **未来函数**: 信号必须 `.shift(1)` 
2. **过拟合**: 参数≤3个，样本外验证
3. **存活偏差**: 退市股票不参与回测
4. **交易成本**: 手续费+滑点≈0.2%/笔
5. **心理偏差**: 回测能扛30%，实盘10%就慌了

---

## 五、机器学习量化

### 模型

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

# 特征
features = ["sma20","sma50","rsi14","volatility","ret5","ret20"]
# 标签: 未来5天涨=1, 跌=0
df["target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)

# Walk-forward 验证 (防过拟合)
tscv = TimeSeriesSplit(n_splits=5)
accuracies = []
for train_idx, test_idx in tscv.split(X):
    model = RandomForestClassifier(n_estimators=100, max_depth=5)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    accuracies.append((pred == y[test_idx]).mean())

print(f"平均准确率: {np.mean(accuracies)*100:.1f}%")
```

### 真实预期
```
金融数据信噪比极低:
  瞎猜:  50%
  ML:    52-58% (优秀)
  85%+: 数据泄露，实盘必崩
```

---

## 六、风险管理

### 仓位计算
```python
# 凯利公式
def kelly(win_rate, avg_win, avg_loss):
    b = avg_win / abs(avg_loss)
    k = (win_rate * b - (1 - win_rate)) / b
    return max(0, k * 0.5)  # 半凯利

# 固定比例
def position_size(capital, risk=0.02, stop=0.05):
    return (capital * risk) / (capital * stop)
```

### 止损
```python
# 固定止损
stop_loss = entry * 0.95   # -5%

# ATR 止损
stop_loss = entry - 2 * atr

# 移动止损
trailing = prices.cummax() * 0.95
```

### 风险价值
```python
var = df["returns"].quantile(0.05) * 100
print(f"95% VaR: {var:.2f}%")  # 95%可能一天亏不超过这个数
```

---

## 七、投资组合优化

```python
# 等权重
weights = {t: 1/n for n, t in enumerate(tickers)}

# 风险平价 (每个资产风险贡献相同)
cov = returns.cov()
inv_vol = 1 / (np.diag(cov) ** 0.5)
weights = {t: w for t, w in zip(tickers, inv_vol / inv_vol.sum())}

# 最小方差
inv_cov = np.linalg.inv(cov.values)
ones = np.ones(len(tickers))
weights = inv_cov @ ones / (ones.T @ inv_cov @ ones)
```

---

## 八、自动化交易

### 每日信号检查脚本
```python
# signal_check.py — 每天早上9点自动跑
import yfinance as yf, pandas as pd
df = yf.download("BTC-USD", period="6mo")
c = df["Close"]
s20 = c.rolling(20).mean().iloc[-1]
s50 = c.rolling(50).mean().iloc[-1]
s200 = c.rolling(200).mean().iloc[-1]
buy = (s20 > s50) and (s200 > s200.shift(20).iloc[-1])
print(f"Signal: {'BUY' if buy else 'WAIT'} | ${float(c.iloc[-1]):,.0f}")
```

### 定时任务设置
```bash
# Linux crontab
0 9 * * * cd /path && python signal_check.py

# Hermes cron (本技能已内置)
# /cron create '0 9 * * *' 'run signal check'
```

---

## 九、Hermes Agent 对话示例

```
你: "分析BTC当前技术面"

Agent:
├ 拉取 yfinance 数据
├ 计算 SMA20/50/200、RSI、MACD
├ 画K线图
├ 判断多空
└ 输出: "BTC $72k, SMA20 < SMA50, 空头排列, 建议等待"

你: "回测SMA交叉策略"

Agent:
├ 拉取2年数据
├ 运行 BacktestEngine
├ 输出: 收益+5.19% 回撤-14.66% 夏普0.20
└ 对比买入持有

你: "用ML跑一遍"

Agent:
├ 特征工程 → 训练 → Walk-forward验证
├ 输出: 准确率46%, 夏普1.61
└ "ML提升有限，建议结合传统策略"
```

---

## ⚠️ 免责声明

本技能仅供学习和研究。所有代码和分析不构成投资建议。量化交易有风险，实盘需谨慎。
