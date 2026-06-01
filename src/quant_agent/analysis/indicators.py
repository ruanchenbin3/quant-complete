"""
技术指标计算模块
包含常见技术指标的纯 NumPy/Pandas 实现，不依赖第三方 TA 库
"""
import pandas as pd
import numpy as np


def sma(series: pd.Series, window: int = 20) -> pd.Series:
    """简单移动平均线"""
    return series.rolling(window=window).mean()


def ema(series: pd.Series, span: int = 20) -> pd.Series:
    """指数移动平均线"""
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """相对强弱指标 (RSI)"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """MACD 指标"""
    ema_fast = ema(series, span=fast)
    ema_slow = ema(series, span=slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, span=signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "histogram": histogram,
    })


def bollinger_bands(
    series: pd.Series, window: int = 20, num_std: float = 2.0
) -> pd.DataFrame:
    """布林带"""
    middle = sma(series, window)
    std = series.rolling(window=window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return pd.DataFrame({
        "upper": upper,
        "middle": middle,
        "lower": lower,
    })


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """平均真实波幅 (ATR)"""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=window).mean()


def kdj(
    df: pd.DataFrame, window: int = 9, k_smooth: int = 3, d_smooth: int = 3
) -> pd.DataFrame:
    """KDJ 随机指标（国内常用）"""
    low_min = df["low"].rolling(window=window).min()
    high_max = df["high"].rolling(window=window).max()
    rsv = (df["close"] - low_min) / (high_max - low_min) * 100
    k = rsv.ewm(com=k_smooth - 1, adjust=False).mean()
    d = k.ewm(com=d_smooth - 1, adjust=False).mean()
    j = 3 * k - 2 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """一键添加所有常用指标到 DataFrame"""
    result = df.copy()
    close = result["close"]
    result["sma_10"] = sma(close, 10)
    result["sma_30"] = sma(close, 30)
    result["ema_12"] = ema(close, 12)
    result["ema_26"] = ema(close, 26)
    result["rsi_14"] = rsi(close, 14)
    macd_df = macd(close)
    result = result.join(macd_df)
    bb = bollinger_bands(close)
    result = result.join(bb)
    result["atr_14"] = atr(result)
    kdj_df = kdj(result)
    result = result.join(kdj_df)
    return result
