# 📊 Quant Trading Complete

完整量化交易技能包——一个仓库包含所有量化交易所需代码。

## 目录

```
quant-complete/
├── skills/SKILL.md          ← 完整的 Hermes 量化技能（一个文件覆盖所有）
├── src/quant_agent/         ← Python 量化核心库
│   ├── data/fetcher.py      ← 数据获取（yfinance/CCXT）
│   ├── analysis/             ← 技术指标 + 回测引擎
│   │   ├── indicators.py
│   │   └── backtest.py
│   ├── strategies/           ← 示例策略
│   │   └── examples.py
│   ├── agent/                ← AI Agent 辅助分析
│   │   └── quant_agent.py
│   └── ml/                   ← ML 量化（Qlib/FreqTrade/PPO 最佳实践）
│       ├── models.py
│       ├── pipeline.py
│       └── portfolio.py
└── scripts/                  ← 日常使用脚本
    ├── signal_check.py       ← 每日信号检查
    └── quant_trader.py       ← 自动交易模块
```

## 安装

```bash
pip install yfinance pandas numpy matplotlib scikit-learn ta ccxt
```

## 使用

### Hermes Agent 加载技能

把 `skills/SKILL.md` 放到 Hermes skills 目录后，对话中:
```
/skill quant-trading
```

### Python 直接使用

```python
from quant_agent.data.fetcher import StockDataFetcher
from quant_agent.analysis.indicators import add_all_indicators
from quant_agent.analysis.backtest import BacktestEngine
from quant_agent.strategies.examples import sma_cross
from quant_agent.ml.pipeline import MLPipeline

# 全流程示例
df = StockDataFetcher().fetch("BTC-USD", period="1y")
df = add_all_indicators(df)
result = BacktestEngine(df, sma_cross).run()
print(f"收益: {result.total_return}%  夏普: {result.sharpe_ratio}")

pipeline = MLPipeline()
pipeline.run("BTC-USD", period="2y")
pipeline.report()
```

### 每日自动信号

```bash
python scripts/signal_check.py
```

## 技能内容

一个技能覆盖量化交易全流程:
1. 数据获取 — yfinance/CCXT
2. 技术指标 — SMA/EMA/RSI/MACD/布林带/KDJ/ATR
3. 策略类型 — 趋势跟踪/均值回归/带过滤
4. 回测 — 完整引擎 + 陷阱解析
5. ML 量化 — 随机森林/GBDT + Walk-forward
6. 风险管理 — 凯利/VaR/止损
7. 投资组合 — 风险平价/最小方差
8. 自动化 — 每日信号/定时任务

## License

MIT
