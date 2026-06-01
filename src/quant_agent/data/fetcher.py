"""
数据获取模块 — 支持股票(A股/美股)和加密货币数据
"""
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
import yfinance as yf
import ccxt


class StockDataFetcher:
    """股票数据获取 (yfinance 驱动，支持美股 + A股)"""

    def __init__(self):
        self._cache = {}

    def fetch(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        获取历史行情数据

        Parameters
        ----------
        symbol : str
            股票代码，如 "AAPL", "600519.SS"
        period : str
            数据周期: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, max
        interval : str
            数据粒度: 1m, 2m, 5m, 15m, 30m, 60m, 1d, 1wk, 1mo

        Returns
        -------
        pd.DataFrame with columns: Open, High, Low, Close, Volume
        """
        cache_key = f"{symbol}_{period}_{interval}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            raise ValueError(f"无法获取 {symbol} 的数据，请检查代码是否正确")

        df.columns = [c.lower() for c in df.columns]
        self._cache[cache_key] = df.copy()
        return df

    def info(self, symbol: str) -> dict:
        """获取股票基本面信息"""
        ticker = yf.Ticker(symbol)
        return ticker.info

    def list_available(self, market: str = "us") -> list:
        """列出常见股票代码（示例）"""
        if market == "us":
            return ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "NVDA", "META", "SPY", "QQQ"]
        elif market == "cn":
            return ["600519.SS", "000858.SZ", "601318.SS", "600036.SS"]
        return []


class CryptoDataFetcher:
    """加密货币数据获取 (CCXT 驱动)"""

    def __init__(self, exchange_id: str = "binance"):
        exchange_class = getattr(ccxt, exchange_id)
        self.exchange = exchange_class({"enableRateLimit": True})
        self._cache = {}

    def fetch(
        self,
        symbol: str = "BTC/USDT",
        timeframe: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        获取加密货币 kline 数据

        Parameters
        ----------
        symbol : str
            交易对，如 "BTC/USDT"
        timeframe : str
            周期: 1m, 5m, 15m, 1h, 4h, 1d, 1w
        limit : int
            获取条数

        Returns
        -------
        pd.DataFrame
        """
        cache_key = f"{symbol}_{timeframe}_{limit}"
        if cache_key in self._cache:
            return self._cache[cache_key].copy()

        ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        self._cache[cache_key] = df.copy()
        return df

    def list_symbols(self) -> list:
        """列出可用的交易对"""
        markets = self.exchange.load_markets()
        return list(markets.keys())[:50]
