import yfinance as yf, pandas as pd
import matplotlib.pyplot as plt

df = yf.download("BTC-USD", period="3mo", interval="1d")
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.droplevel(1)

c = df["Close"]
s20 = c.rolling(20).mean().iloc[-1]
s50 = c.rolling(50).mean().iloc[-1]
s200 = c.rolling(200).mean().iloc[-1]
s200p = c.rolling(200).mean().shift(20).iloc[-1]
p = float(c.iloc[-1])
buy = (s20 > s50) and (s200 > s200p) if not pd.isna(s200) else False
rsi = 14
d_ = c.diff(); g_ = d_.clip(lower=0); l_ = -d_.clip(upper=0)
ag_ = g_.rolling(14, min_periods=1).mean(); al_ = l_.rolling(14, min_periods=1).mean()
rsi_val = 100 - 100/(1 + ag_/al_.replace(0, 1e-10))

# 画图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

ax1.plot(df.index, df["Close"], label="Close", color="black", linewidth=1.5)
ax1.plot(df.index, c.rolling(20).mean(), label="SMA20", linestyle="--", alpha=0.7)
ax1.plot(df.index, c.rolling(50).mean(), label="SMA50", linestyle="--", alpha=0.7)
if not pd.isna(s200):
    ax1.plot(df.index, c.rolling(200).mean(), label="SMA200", linestyle=":", alpha=0.5)
ax1.set_title(f"BTC/USD - {'BUY' if buy else 'WAIT'} | ${p:,.0f}")
ax1.legend()
ax1.grid(alpha=0.3)

ax2.plot(df.index, rsi_val, color="purple", linewidth=1)
ax2.axhline(70, color="r", linestyle=":", alpha=0.5)
ax2.axhline(30, color="g", linestyle=":", alpha=0.5)
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"C:\\Users\\test\\quant-mira\\workspace\\btc_chart.png", dpi=150)
print(f"Chart saved")
print(f"=== BTC SIGNAL | {df.index[-1].strftime('%Y-%m-%d')} ===")
print(f"Price:  ${p:>8,.0f}")
print(f"SMA20:  ${float(s20):>8,.0f}")
print(f"SMA50:  ${float(s50):>8,.0f}")
print(f"SMA200: ${float(s200) if not pd.isna(s200) else 'N/A':>8}")
print(f"RSI:    {float(rsi_val.iloc[-1]):>7.1f}")
print(f"Signal: {'BUY' if buy else 'WAIT'}")
print(f"Action: {'买入' if buy else '空仓等待'}")

import os
size = os.path.getsize(r"C:\\Users\\test\\quant-mira\\workspace\\btc_chart.png")
print(f"Chart file: btc_chart.png ({size/1024:.0f} KB)")
