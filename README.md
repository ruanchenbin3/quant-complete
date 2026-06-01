# 📊 Quant Complete

完整量化交易技能包 — 选股 + 择时 + 自动化

---

## 项目结构

```
quant-complete/
├── scripts/
│   ├── deep_value_screener.py   每天扫 200+ 只股票, 找跌深+有价值的标的
│   └── quant_trader.py          自动交易模块
├── src/quant_agent/             Python 量化核心库
│   ├── data/fetcher.py          数据获取
│   ├── analysis/                技术指标 + 回测
│   ├── strategies/              示例策略
│   ├── agent/                   AI Agent 分析
│   └── ml/                      ML 量化模块
├── reports/                     每日分析报告 (按日期存放)
│   ├── 2026-06-01/report.json
│   ├── 2026-06-02/report.json
│   └── tracking.csv             跟踪列表
├── skills/SKILL.md              Hermes 量化技能 (一个文件覆盖全流程)
└── README.md
```

---

## 每日选股策略

```
深度价值选股 = 选好公司 + 等好价格

选好公司:
  1. 从 200+ 只知名股票中筛选
  2. 基本面评分: PE合理、PB低、负债率低
  3. 市值 > 100 亿, 流动性好

等好价格:
  1. 从高点跌超 15% 才关注
  2. RSI < 30 极度超卖时重点考虑
  3. 分 3 批买入, 每跌 5% 加一次
  4. 止损 -10%, 目标 +20%

这不是价值投资, 也不是纯量化
是: 价值选股 + 量化择时 = 完整的交易系统
```

### 量化的本质

```
价格 = 所有人信心和恐惧的总和
K线 = 这种情绪的量化记录
技术指标 = 情绪的数学摘要
模型 = 在情绪数据中找重复模式
```

### 量化 vs 价值投资

```
             量化交易                   价值投资
──────────────────────────────────────────────────────────
信仰:     价格波动可预测             价格终归会回归价值
依据:     历史数据 + 统计规律        公司基本面 + 护城河
持仓:     几分钟到几个月             几年到几十年
决策:     指标信号触发买卖           估值高估/低估时操作
风险:     模型失效、过拟合、黑天鹅    看错公司、行业消亡
收益来源:  赚波动钱                   赚企业成长钱
本质:     投机                       投资
```

---

## 快速开始

```bash
# 安装依赖
pip install yfinance pandas numpy

# 每日选股
python scripts/deep_value_screener.py
```

### 输出示例

```
深度价值选股 | 2026-06-02
扫描 163 只, 找到 29 只, 耗时 93s

==============================================================
  UHS — Universal Health Services, Inc.
  评分: 12/15
  价格: $144.27  从高点跌了 -39.7%
  RSI: 11.1  PE: 6.0  PB: 1.17
  极度超卖！PE极低，估值有安全边际。负债率低，财务稳健。
  简介: Universal Health Services provides hospital and healthcare services...
```

### 用 Hermes Agent 加载技能

```
/skill quant-trading
```

---

## 回测验证

用历史数据回测过去 30 天:
```
7 只被选中 → 3 只涨 4 只跌 (准确率 43%)
但赚的: MDB +49%, F +44%, MSFT +11%
亏的: PYPL -9.5%, SNAP -6%, UBER -2.8%
净收益: +12.2% / 30 天

关键: 胜率不重要, 盈亏比才重要
```

---

## 定时任务

每天早上 9 点自动运行:
- BTC 信号检查
- 深度价值选股 (200+ 只)

报告保存到 `reports/YYYY-MM-DD/report.json`

---

## 知识笔记

量化交易的基础知识笔记放在:
[https://github.com/ruanchenbin3/quant-notes](https://github.com/ruanchenbin3/quant-notes)

---

## License

MIT
