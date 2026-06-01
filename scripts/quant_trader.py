"""
quant_trader.py — Agent 自动交易模块

用法:
    from quant_trader import QuantTrader
    
    trader = QuantTrader()                    # dry_run=True 模拟交易
    trader = QuantTrader(dry_run=False)       # 实盘 (需配置 API Key)
    
    trader.check_signal()                     # 当前信号
    trader.buy()                              # 开仓
    trader.sell()                             # 平仓
    trader.status()                           # 持仓状态

安全限制:
    - 单笔最大 100 RMB (硬编码)
    - 不上杠杆 (现货)
    - dry_run 默认开启
    - 所有交易记录到 trades.csv
"""

import os, json, time
from pathlib import Path
from datetime import datetime

class QuantTrader:
    """100 RMB 量化交易员"""
    
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.max_investment = 14.0  # ~100 RMB in USDT
        self.trade_log = Path(__file__).parent / "trades.csv"
        self._init_log()
        
        # CCXT exchange (延迟初始化)
        self._exchange = None
    
    def _get_exchange(self):
        """获取交易所连接"""
        if self._exchange:
            return self._exchange
        
        import ccxt
        
        # 读取 API Key
        config_path = Path(__file__).parent / "config" / "exchange.json"
        if not config_path.exists():
            print("⚠️  未配置交易所 API Key")
            print(f"   创建 {config_path}:")
            print('    {"api_key": "xxx", "api_secret": "xxx", "exchange": "binance"}')
            return None
        
        with open(config_path) as f:
            conf = json.load(f)
        
        exchange_class = getattr(ccxt, conf.get("exchange", "binance"))
        self._exchange = exchange_class({
            "apiKey": conf["api_key"],
            "secret": conf["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        return self._exchange
    
    def _init_log(self):
        """初始化交易日志"""
        if not self.trade_log.exists():
            with open(self.trade_log, "w") as f:
                f.write("time,action,price,amount,value,note\n")
    
    def _log_trade(self, action, price, amount, value, note=""):
        """记录交易"""
        with open(self.trade_log, "a") as f:
            f.write(f"{datetime.now().isoformat()},{action},{price},{amount},{value},{note}\n")
        print(f"  📝 已记录: {action} ${price} × {amount}")
    
    # ──────────── 信号 ────────────
    
    def check_signal(self):
        """检查当前交易信号"""
        import yfinance as yf, pandas as pd
        
        df = yf.download("BTC-USD", period="6mo", interval="1d")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        
        c = df["Close"]
        sma20 = c.rolling(20).mean()
        sma50 = c.rolling(50).mean()
        sma200 = c.rolling(200).mean()
        
        l = df.iloc[-1]
        price = float(l["Close"])
        s20 = float(sma20.iloc[-1])
        s50 = float(sma50.iloc[-1])
        s200 = float(sma200.iloc[-1])
        s200_prev = float(sma200.shift(20).iloc[-1])
        
        buy = (s20 > s50) and (s200 > s200_prev)
        sell = (s20 < s50)  # 死叉平仓
        
        return {
            "signal": "buy" if buy else ("sell" if sell else "hold"),
            "price": price,
            "sma20": s20,
            "sma50": s50,
            "sma200": s200,
            "date": str(l.name.date()),
        }
    
    # ──────────── 交易 ────────────
    
    def buy(self):
        """开仓买入"""
        signal = self.check_signal()
        
        if signal["signal"] != "buy":
            print(f"❌ 信号不是买入: {signal['signal']}")
            return False
        
        amount_usdt = self.max_investment
        price = signal["price"]
        btc_amount = amount_usdt / price
        
        if self.dry_run:
            print(f"🟢 [模拟] 买入 ${price:,.0f} × {btc_amount:.6f} BTC = ${amount_usdt:.2f}")
            self._log_trade("buy_sim", price, btc_amount, amount_usdt, "dry_run")
            return True
        
        # 实盘
        ex = self._get_exchange()
        if not ex:
            return False
        
        try:
            # 市价买单
            order = ex.create_market_buy_order("BTC/USDT", btc_amount)
            print(f"🟢 [实盘] 买入成功! ID: {order.get('id', 'N/A')}")
            filled_price = float(order.get("price", price))
            filled_amount = float(order.get("filled", btc_amount))
            cost = float(order.get("cost", filled_price * filled_amount))
            self._log_trade("buy", filled_price, filled_amount, cost, order.get("id", ""))
            return True
        except Exception as e:
            print(f"❌ 买入失败: {e}")
            return False
    
    def sell(self):
        """平仓卖出"""
        position = self.get_position()
        if position["btc"] <= 0:
            print("❌ 没有持仓可卖")
            return False
        
        btc_amount = position["btc"]
        price = position["current_price"]
        value = btc_amount * price
        
        if self.dry_run:
            print(f"🔴 [模拟] 卖出 {btc_amount:.6f} BTC @ ${price:,.0f} = ${value:.2f}")
            profit = value - self.max_investment
            print(f"   盈亏: ${profit:+.2f} ({profit/self.max_investment*100:+.2f}%)")
            self._log_trade("sell_sim", price, btc_amount, value, "dry_run")
            return True
        
        ex = self._get_exchange()
        if not ex:
            return False
        
        try:
            order = ex.create_market_sell_order("BTC/USDT", btc_amount)
            print(f"🔴 [实盘] 卖出成功! ID: {order.get('id', 'N/A')}")
            filled_price = float(order.get("price", price))
            filled_amount = float(order.get("filled", btc_amount))
            cost = float(order.get("cost", filled_price * filled_amount))
            profit = cost - self.max_investment
            print(f"   盈亏: ${profit:+.2f} ({profit/self.max_investment*100:+.2f}%)")
            self._log_trade("sell", filled_price, filled_amount, cost, order.get("id", ""))
            return True
        except Exception as e:
            print(f"❌ 卖出失败: {e}")
            return False
    
    def get_position(self):
        """查询当前持仓"""
        import yfinance as yf
        ticker = yf.Ticker("BTC-USD")
        info = ticker.history(period="1d")
        price = float(info["Close"].iloc[-1])
        
        if self.dry_run:
            # 从日志读取最近未平仓的买入
            btc = 0.0
            if self.trade_log.exists():
                with open(self.trade_log) as f:
                    lines = f.readlines()[1:]  # skip header
                for line in reversed(lines):
                    parts = line.strip().split(",")
                    if len(parts) >= 4:
                        action = parts[1]
                        if action == "buy_sim":
                            btc += float(parts[3])
                        elif action in ("sell_sim", "sell"):
                            btc -= float(parts[3])
            
            return {
                "btc": max(0, btc),
                "current_price": price,
                "value_usdt": max(0, btc) * price,
                "mode": "dry_run",
            }
        
        ex = self._get_exchange()
        if not ex:
            return {"btc": 0, "current_price": price, "value_usdt": 0, "mode": "no_exchange"}
        
        try:
            balance = ex.fetch_balance()
            btc = float(balance["BTC"]["free"]) if "BTC" in balance else 0
            return {
                "btc": btc,
                "current_price": price,
                "value_usdt": btc * price,
                "mode": "live",
            }
        except Exception as e:
            print(f"❌ 查询持仓失败: {e}")
            return {"btc": 0, "current_price": price, "value_usdt": 0, "mode": "error"}
    
    def status(self):
        """完整状态报告"""
        signal = self.check_signal()
        pos = self.get_position()
        
        print(f"{'='*50}")
        print(f"  🤖 QuantTrader 状态")
        print(f"{'='*50}")
        print(f"  模式:     {'🟡 模拟' if self.dry_run else '🔴 实盘'}")
        print(f"  信号:     {'🟢 买入' if signal['signal']=='buy' else '🔴 卖出' if signal['signal']=='sell' else '⚪ 持有'}")
        print(f"  BTC:      ${signal['price']:>8,.0f}")
        print(f"  SMA20:    ${signal['sma20']:>8,.0f}")
        print(f"  SMA50:    ${signal['sma50']:>8,.0f}")
        print(f"  SMA200:   ${signal['sma200']:>8,.0f}")
        print(f"  持仓:     {pos['btc']:.6f} BTC (${pos['value_usdt']:.2f})")
        print(f"{'='*50}")
        
        return {"signal": signal, "position": pos}
