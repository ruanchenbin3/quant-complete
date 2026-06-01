"""
QuantAgent — AI 辅助量化分析智能体

这个 Agent 的核心功能:
1. 快速分析: 输入一支股票，自动计算技术指标 + 给出解读
2. 回测运行: 输入策略和标的，运行回测并给出评价
3. 知识问答: 解释量化概念
4. 报告生成: 生成可读性强的分析报告

TODO: 可以接入 LLM API (OpenAI/DeepSeek 等) 来实现自然语言交互
"""
from datetime import datetime
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from quant_agent.data.fetcher import StockDataFetcher
from quant_agent.analysis.indicators import add_all_indicators, rsi
from quant_agent.analysis.backtest import BacktestEngine
from quant_agent.strategies.examples import sma_cross, rsi_strategy, bollinger_reversal, macd_signal

console = Console()

# 策略注册表 — 新增策略只需在这里注册
STRATEGY_REGISTRY = {
    "sma_cross": sma_cross,
    "rsi": rsi_strategy,
    "bollinger": bollinger_reversal,
    "macd": macd_signal,
}


class QuantAgent:
    """量化分析 AI Agent"""

    def __init__(self):
        self.fetcher = StockDataFetcher()

    # ──────────── 快速分析 ────────────

    def quick_analysis(self, symbol: str = "AAPL", period: str = "6mo") -> str:
        """对一只股票做快速技术分析"""
        try:
            df = self.fetcher.fetch(symbol, period=period, interval="1d")
        except Exception as e:
            return f"[red]✗ 数据获取失败: {e}[/red]"

        df = add_all_indicators(df)
        latest = df.iloc[-1]

        # 构建分析报告
        report_parts = [
            Panel(f"[bold cyan]{symbol}[/bold cyan] 快速技术分析 | {datetime.now().strftime('%Y-%m-%d %H:%M')}"),
            "",
        ]

        # 价格信息
        price_table = Table(title="价格概览")
        price_table.add_column("指标", style="cyan")
        price_table.add_column("数值", style="yellow")
        price_table.add_row("最新收盘价", f"{latest['close']:.2f}")
        price_table.add_row("开盘价", f"{latest['open']:.2f}")
        price_table.add_row("最高 / 最低", f"{latest['high']:.2f} / {latest['low']:.2f}")
        price_table.add_row("成交量", f"{latest['volume']:,.0f}")
        period_return = (latest["close"] / df.iloc[0]["close"] - 1) * 100
        price_table.add_row(f"期间涨跌幅 ({period})", f"{period_return:+.2f}%")
        report_parts.append(price_table)

        # 趋势
        trend_table = Table(title="趋势指标")
        trend_table.add_column("指标", style="cyan")
        trend_table.add_column("数值", style="yellow")
        trend_table.add_column("信号", style="green")
        trend_table.add_row("SMA(10)", f"{latest.get('sma_10', 0):.2f}", "↑ 短期均线" if latest.get('sma_10', 0) > latest.get('sma_30', 0) else "↓ 短期均线")
        trend_table.add_row("SMA(30)", f"{latest.get('sma_30', 0):.2f}", "-")
        trend_table.add_row("RSI(14)", f"{latest.get('rsi_14', 0):.1f}", self._rsi_signal(latest.get('rsi_14', 50)))
        trend_table.add_row("MACD", f"{latest.get('macd', 0):.4f}", "看多" if latest.get('macd', 0) > latest.get('signal', 0) else "看空")

        bb_signal = ""
        close = latest["close"]
        bb_lower = latest.get("lower", 0)
        bb_upper = latest.get("upper", 0)
        if close < bb_lower:
            bb_signal = "触及下轨 (超卖)"
        elif close > bb_upper:
            bb_signal = "触及上轨 (超买)"
        else:
            bb_signal = "布林带内"
        trend_table.add_row("布林带位置", f"{close:.2f}", bb_signal)
        report_parts.append(trend_table)

        # 综合建议
        advice = self._generate_advice(latest)
        report_parts.append(Panel(advice, title="💡 综合分析建议", border_style="yellow"))

        # 用 console 捕获渲染结果
        from io import StringIO
import pandas as pd
        buf = StringIO()
        _console = Console(file=buf, width=100)
        for part in report_parts:
            if isinstance(part, str):
                _console.print(part)
            else:
                _console.print(part)
        return buf.getvalue()

    def _rsi_signal(self, val: float) -> str:
        if val < 30:
            return "🟢 超卖 (可能反弹)"
        elif val > 70:
            return "🔴 超买 (可能回调)"
        elif val < 50:
            return "📉 偏弱"
        else:
            return "📈 偏强"

    def _generate_advice(self, latest) -> str:
        """根据指标生成综合建议"""
        lines = ["基于技术指标的综合判断:\n"]

        rsi_val = latest.get("rsi_14", 50)
        if rsi_val < 30:
            lines.append("• RSI 处于超卖区 (< 30)，短期反弹概率较大，可关注买入机会")
        elif rsi_val > 70:
            lines.append("• RSI 处于超买区 (> 70)，短期回调风险较高，注意止盈")
        else:
            lines.append(f"• RSI 处于 {rsi_val:.0f}，无明显超买超卖信号")

        macd_val = latest.get("macd", 0)
        signal_val = latest.get("signal", 0)
        if macd_val > signal_val:
            lines.append("• MACD 位于信号线上方，短期趋势偏多")
        else:
            lines.append("• MACD 位于信号线下方，短期趋势偏空")

        close = latest["close"]
        bb_lower = latest.get("lower", 0)
        bb_upper = latest.get("upper", 0)
        if bb_lower and close < bb_lower:
            lines.append("• 价格触及布林带下轨，处于相对低位")
        elif bb_upper and close > bb_upper:
            lines.append("• 价格触及布林带上轨，处于相对高位")

        lines.append("\n⚠️ 以上仅为技术面分析，不构成投资建议。请结合基本面、市场情绪综合判断。")
        return "\n".join(lines)

    # ──────────── 回测 ────────────

    def run_backtest(
        self,
        symbol: str,
        strategy_name: str = "sma_cross",
        start: str = "2023-01-01",
        end: str = "2024-12-31",
    ) -> str:
        """运行策略回测"""
        if strategy_name not in STRATEGY_REGISTRY:
            return f"[red]✗ 未知策略: {strategy_name}. 可选: {list(STRATEGY_REGISTRY.keys())}[/red]"

        try:
            df = self.fetcher.fetch(symbol, period="2y", interval="1d")
            df = df[start:end]
        except Exception as e:
            return f"[red]✗ 数据获取失败: {e}[/red]"

        if df.empty:
            return "[red]✗ 所选时间段内没有数据[/red]"

        strategy_fn = STRATEGY_REGISTRY[strategy_name]
        engine = BacktestEngine(df, strategy_fn)
        result = engine.run()

        # 渲染结果
        eq = result.equity_curve
        ret_table = Table(title=f"[bold]{symbol} — {strategy_name} 回测结果[/bold]")
        ret_table.add_column("指标", style="cyan")
        ret_table.add_column("数值", style="yellow")

        ret_table.add_row("初始资金", f"${engine.initial_capital:,.0f}")
        ret_table.add_row("最终净值", f"${eq.iloc[-1]:,.2f}")
        ret_table.add_row("总收益率", f"{result.total_return:+.2f}%")
        ret_table.add_row("年化收益率", f"{result.annual_return:+.2f}%")
        ret_table.add_row("最大回撤", f"{result.max_drawdown:.2f}%")
        ret_table.add_row("夏普比率", f"{result.sharpe_ratio:.2f}")
        ret_table.add_row("交易次数", f"{result.total_trades}")
        ret_table.add_row("胜率", f"{result.win_rate:.1f}%")

        buf = StringIO()
        _console = Console(file=buf, width=100)
        _console.print(ret_table)
        return buf.getvalue()

    # ──────────── 知识问答 ────────────

    def explain_concept(self, concept: str) -> str:
        """解释量化概念"""
        explanations = {
            "均线": "移动平均线 (MA) 是最基础的趋势指标。\
                    快线上穿慢线 = 金叉 (看多)；\
                    快线下穿慢线 = 死叉 (看空)。\
                    常用组合: SMA(5,10,20) 短线, SMA(30,60) 中线, SMA(120,250) 长线。",
            "rsi": "相对强弱指标 (RSI) 衡量价格涨跌速度。\
                   RSI > 70 = 超买 (可能回调)；\
                   RSI < 30 = 超卖 (可能反弹)；\
                   常用参数: 14 天周期。",
            "macd": "指数平滑异同移动平均线 (MACD)。\
                    MACD = 快线(12日EMA) - 慢线(26日EMA)。\
                    MACD上穿信号线 = 金叉 (买入信号)。\
                    MACD下穿信号线 = 死叉 (卖出信号)。\
                    柱状图 = MACD - 信号线，反映动能强弱。",
            "回测": "回测 (Backtesting) 是用历史数据验证策略表现的过程。\
                    核心指标: 总收益率、年化收益率、最大回撤、夏普比率、胜率。\
                    注意事项: 过拟合、生存偏差、未来函数、交易成本。",
            "夏普": "夏普比率 (Sharpe Ratio) 衡量风险调整后收益。\
                    公式: (策略收益率 - 无风险利率) / 收益率标准差。\
                    一般 > 1 算可以，> 2 算优秀。",
            "最大回撤": "最大回撤 (Max Drawdown) 指从最高点到最低点的最大跌幅。\
                        公式: (波谷值 - 波峰值) / 波峰值 × 100%。\
                        反映策略在最差情况下的亏损程度。",
            "因子": "因子 (Factor) 是能解释股票收益的特征。\
                    常见因子: 市值因子、价值因子、动量因子、质量因子、低波因子。\
                    多因子模型 = 组合多个因子来选股。",
        }
        concept_lower = concept.lower()
        for key, exp in explanations.items():
            if key in concept_lower:
                return f"📖 **{concept}**\n\n{exp}"
        return f"📖 关于「{concept}」我还需要学习。试试: 均线, RSI, MACD, 回测, 夏普, 最大回撤, 因子"
