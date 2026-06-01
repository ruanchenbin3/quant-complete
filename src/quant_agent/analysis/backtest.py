"""
回测引擎 — 支持自定义策略的回测框架
"""
from typing import Callable, Dict
import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class Trade:
    """单笔交易记录"""
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    side: str  # "long" or "short"
    size: float = 1.0
    pnl: float = 0.0
    pnl_pct: float = 0.0


@dataclass
class BacktestResult:
    """回测结果"""
    total_return: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    total_trades: int = 0
    trades: list = field(default_factory=list)
    equity_curve: pd.Series = field(default_factory=pd.Series)


class BacktestEngine:
    """回测引擎

    用法:
        engine = BacktestEngine(data, strategy_func)
        result = engine.run()
    """

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_fn: Callable,
        initial_capital: float = 100_000,
        commission: float = 0.001,
    ):
        """
        Parameters
        ----------
        data : pd.DataFrame
            包含 'open','high','low','close','volume' 的行情数据
        strategy_fn : Callable
            策略函数，接受 data 返回持仓信号 (-1~1)
        initial_capital : float
            初始资金
        commission : float
            手续费比例 (如 0.001 = 千分之一)
        """
        self.data = data.copy()
        self.strategy_fn = strategy_fn
        self.initial_capital = initial_capital
        self.commission = commission
        self.position = 0  # 当前持仓量
        self.cash = initial_capital
        self.trades: list[Trade] = []
        self.equity = []

    def run(self) -> BacktestResult:
        """运行回测"""
        signals = self.strategy_fn(self.data)
        self.data = self.data.copy()
        self.data["signal"] = signals

        for i in range(len(self.data)):
            row = self.data.iloc[i]
            price = row["close"]
            signal = row["signal"]

            # 信号为 -1~1 的连续值 -> 仓位比例
            target_pos = signal * self.initial_capital / price
            delta = target_pos - self.position

            if abs(delta) > 0:
                # 执行交易
                trade_value = abs(delta) * price
                cost = trade_value * self.commission
                self.cash -= cost

                if delta > 0:
                    self.position += delta
                    self.cash -= trade_value
                else:
                    self.position += delta
                    self.cash += trade_value

                self.trades.append(
                    Trade(
                        entry_date=row.name,
                        exit_date=row.name,
                        entry_price=price,
                        exit_price=price,
                        side="long" if delta > 0 else "short",
                        size=abs(delta),
                    )
                )

            # 记录净值
            equity_value = self.cash + self.position * price
            self.equity.append(equity_value)

        return self._compute_results()

    def _compute_results(self) -> BacktestResult:
        """计算回测指标"""
        equity_curve = pd.Series(self.equity, index=self.data.index)
        returns = equity_curve.pct_change().dropna()

        total_return = (equity_curve.iloc[-1] / self.initial_capital - 1) * 100
        n_years = len(returns) / 252
        annual_return = (1 + total_return / 100) ** (1 / n_years) - 1 if n_years > 0 else 0

        # 最大回撤
        rolling_max = equity_curve.expanding().max()
        drawdown = (equity_curve - rolling_max) / rolling_max * 100
        max_drawdown = drawdown.min()

        # 夏普比
        sharpe = (
            np.sqrt(252) * returns.mean() / returns.std()
            if returns.std() > 0
            else 0.0
        )

        # 胜率
        trades_pnl = [t.pnl_pct for t in self.trades if t.pnl_pct != 0]
        win_rate = sum(1 for p in trades_pnl if p > 0) / len(trades_pnl) * 100 if trades_pnl else 0.0

        return BacktestResult(
            total_return=round(total_return, 2),
            annual_return=round(annual_return * 100, 2),
            max_drawdown=round(max_drawdown, 2),
            sharpe_ratio=round(sharpe, 2),
            win_rate=round(win_rate, 2),
            total_trades=len(self.trades),
            trades=self.trades,
            equity_curve=equity_curve,
        )
