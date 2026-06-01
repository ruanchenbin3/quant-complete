"""
深度价值选股 — 每天扫一遍全市场, 找跌得多+有价值的股票
"""

import yfinance as yf, pandas as pd, numpy as np
from datetime import datetime
from pathlib import Path
import json

REPORTS_DIR = Path(__file__).parent.parent / "reports"
TRACK_FILE = REPORTS_DIR / "tracking.csv"

TICKERS = [
    "INTC","PYPL","SNAP","PFE","MRNA","BA","WBA","DIS",
    "F","GM","NIO","UBER","MDB","T","VZ",
    "AAPL","MSFT","GOOGL","NVDA","AMD",
]

def screen():
    results = []
    today = datetime.now().strftime("%Y-%m-%d")
    today_dir = REPORTS_DIR / today
    today_dir.mkdir(exist_ok=True)
    
    print(f"深度价值选股 | {today}")
    print("=" * 55)
    
    for t in TICKERS:
        try:
            df = yf.download(t, period="6mo", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            if len(df) < 60:
                continue
            
            c = df["Close"]
            price = float(c.iloc[-1])
            high = float(c.max())
            drop = (price / high - 1) * 100
            
            # 只关注跌超15%的
            if drop > -15:
                continue
            
            info = yf.Ticker(t).info
            pe = info.get("trailingPE", 0) or 0
            pb = info.get("priceToBook", 0) or 0
            
            delta = c.diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = -delta.clip(upper=0).rolling(14).mean()
            rsi = float((100 - 100 / (1 + gain / loss.replace(0, 1e-10))).iloc[-1])
            
            score = 0
            if drop < -30: score += 3
            elif drop < -20: score += 2
            if 5 < pe < 25: score += 2
            if 0 < pb < 3: score += 1
            if rsi < 35: score += 2
            elif rsi < 45: score += 1
            
            results.append({
                "ticker": t, "score": score, "price": round(price, 2),
                "drop_pct": round(drop, 1), "rsi": round(rsi, 1),
                "pe": round(pe, 1), "pb": round(pb, 2), "date": today
            })
        except:
            continue
    
    results.sort(key=lambda x: -x["score"])
    
    # Print results
    print(f"\n{'股票':<8} {'评分':<6} {'价格':<10} {'跌幅%':<8} {'RSI':<6} {'PE':<8}")
    print("-" * 50)
    for r in results[:10]:
        print(f"{r['ticker']:<8} {r['score']:<6} ${r['price']:<7,.2f} {r['drop_pct']:<+7.1f}% {r['rsi']:<5.1f} {r['pe']:<8.1f}")
    
    # Save report
    report_file = today_dir / "report.json"
    with open(report_file, "w") as f:
        json.dump({"date": today, "results": results[:10], "total_screened": len(TICKERS), "total_found": len(results)}, f, indent=2)
    
    # Update tracking
    track = []
    if TRACK_FILE.exists():
        track = pd.read_csv(TRACK_FILE).to_dict("records")
    
    for r in results[:5]:
        existing = [x for x in track if x["ticker"] == r["ticker"]]
        if not existing:
            r["entry_price"] = r["price"]
            r["status"] = "watching"
            track.append(r)
    
    pd.DataFrame(track).to_csv(TRACK_FILE, index=False)
    
    print(f"\n报告已保存: reports/{today}/")
    print(f"跟踪列表: {len(track)} 只股票")
    
    return results

if __name__ == "__main__":
    screen()
