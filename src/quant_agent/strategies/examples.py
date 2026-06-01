"""
示例策略 — 从简单到复杂，方便入门学习
"""
import pandas as pd
import numpy as np

from quant_agent.analysis.indicators import sma, rsi, macd, bollinger_bands


def sma_cross(df: pd.DataFrame, fast: int = 10, slow: int = 30) -> pd.Series:
    """双均线交叉策略 — 最经典的入门策略

    金叉(快线上穿慢线) -> 买入
    死叉(快线下穿慢线) -> 卖出

    信号: 1=做多, -1=做空, 0=空仓
    """
    fast_sma = sma(df["close"], fast)
    slow_sma = sma(df["close"], slow)

    signal = pd.Series(0, index=df.index)
    signal[fast_sma > slow_sma] = 1
    signal[fast_sma < slow_sma] = -1
    return signal


def rsi_strategy(df: pd.DataFrame, window: int = 14, oversold: float = 30, overbought: float = 70) -> pd.Series:
    """RSI 超买超卖策略

    RSI < 30 (超卖) -> 买入
    RSI > 70 (超买) -> 卖出
    """
    rsi_val = rsi(df["close"], window)

    signal = pd.Series(0, index=df.index)
    signal[rsi_val < oversold] = 1
    signal[rsi_val > overbought] = -1
    return signal


def bollinger_reversal(df: pd.DataFrame, window: int = 20, num_std: float = 2.0) -> pd.Series:
    """布林带均值回归策略

    价格触及下轨 -> 买入 (预期回归)
    价格触及上轨 -> 卖出 (预期回归)
    """
    bb = bollinger_bands(df["close"], window, num_std)

    signal = pd.Series(0, index=df.index)
    signal[df["close"] < bb["lower"]] = 1   # 触及下轨，买入
    signal[df["close"] > bb["upper"]] = -1  # 触及上轨，卖出
    return signal


def macd_signal(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal_period: int = 9) -> pd.Series:
    """MACD 金叉死叉策略

    MACD 线上穿信号线 -> 买入
    MACD 线下穿信号线 -> 卖出
    """
    macd_data = macd(df["close"], fast, slow, signal_period)

    signal = pd.Series(0, index=df.index)
    signal[macd_data["macd"] > macd_data["signal"]] = 1
    signal[macd_data["macd"] < macd_data["signal"]] = -1
    return signal
