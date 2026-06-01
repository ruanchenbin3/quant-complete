"""
深度价值选股 — 每日扫 200+ 只股票, 找跌深+有价值的标的
"""
import yfinance as yf, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json
from pathlib import Path

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","BRK-B","JPM","V",
    "JNJ","WMT","PG","MA","UNH","HD","DIS","NFLX","ADBE","CRM",
    "INTC","AMD","IBM","CSCO","QCOM","TXN","AVGO","ORCL","SAP","SPOT",
    "PYPL","SQ","SNAP","UBER","ABNB","DASH","MDB","CRWD","ZS","DDOG",
    "PFE","MRNA","BNTX","GILD","AMGN","BIIB","REGN","VRTX","ILMN","ABBV",
    "BA","CAT","GE","MMM","HON","LMT","RTX","NOC","GD","TXT",
    "XOM","CVX","COP","SLB","OXY","EOG","HAL","DVN","MPC","VLO",
    "F","GM","RIVN","LCID","NIO","LI","STLA",
    "WBA","T","VZ","CMCSA","CHTR","C",
    "WFC","BAC","GS","MS","SCHW","BLK","BX","KKR","APO","ARES",
    "ABT","TMO","MDT","SYK","BSX","EW","STE",
    "COST","TGT","LOW","ROST","TJX","SBUX",
    "NKE","MCD","DEO","PEP","KO","MDLZ","CL","KMB",
    "SPGI","MCO","FI","FIS","JKHY","GPN",
    "NOW","WDAY","TEAM","ZS","PANW","FTNT","CHKP","OKTA",
    "DE","LIN","SHW","ECL","APD","PPG",
    "USB","PNC","TFC","KEY","HBAN","CFG","RF","ZION","MTB",
    "ED","DUK","SO","D","NEE","AEP","EXC","PEG","PCG","ES",
    "KRE","KBE","LULU","DECK","CROX","SKX","ONON","BABA",
    "RACE","TSM","ASML","DE","LIN","SHW","ECL","APD",
    "HCA","UHS","DVA","THC","ACM","TTEK",
    "LULU","DECK","CROX","SKX","ONON","BABA",
    "USB","PNC","TFC","KEY","HBAN","CFG","RF","ZION",
    "ED","DUK","SO","D","NEE","AEP","EXC","PEG","ES",
]

REPORTS_DIR = Path(__file__).parent.parent / "reports"

def analyze_stock(t):
    try:
        df = yf.download(t, period="6mo", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
        if len(df) < 60:
            return None
        c = df["Close"]; price = float(c.iloc[-1])
        high = float(c.max()); drop = (price / high - 1) * 100
        if drop > -15: return None
        info = yf.Ticker(t).info
        pe = info.get("trailingPE", 0) or 0
        pb = info.get("priceToBook", 0) or 0
        de = info.get("debtToEquity", 0) or 0
        mc = info.get("marketCap", 0) or 0
        name = info.get("shortName", t)[:30]
        delta = c.diff(); g = delta.clip(lower=0).rolling(14).mean(); l = -delta.clip(upper=0).rolling(14).mean()
        rsi = float((100 - 100/(1+g/l.replace(0,1e-10))).iloc[-1])
        score = 0
        if drop < -40: score += 4; score += 2
        elif drop < -30: score += 3
        elif drop < -20: score += 2
        if 5 < pe < 20: score += 3
        if 0 < pb < 2: score += 2
        elif pb < 4: score += 1
        if de < 50: score += 2; score += 1
        elif de < 100: score += 1
        if rsi < 30: score += 2
        elif rsi < 40: score += 1
        if mc > 10e9: score += 1
        return {"ticker":t,"name":name,"score":score,"price":round(price,2),
                "drop":round(drop,1),"rsi":round(rsi,1),"pe":round(pe,1),"pb":round(pb,2),"de":round(de,1),"mc_b":round(mc/1e9,1)}
    except: return None

def main():
    start = datetime.now()
    results = []
    with ThreadPoolExecutor(max_workers=10) as exe:
        futures = {exe.submit(analyze_stock, t): t for t in TICKERS if t}
        for f in as_completed(futures):
            r = f.result()
            if r: results.append(r)
    results.sort(key=lambda x: -x["score"])
    elapsed = (datetime.now() - start).seconds
    today = datetime.now().strftime("%Y-%m-%d")

    # Print
    print(f"深度价值选股 | {today}")
    print(f"扫描 {len(TICKERS)} 只, 找到 {len(results)} 只, 耗时 {elapsed}s")
    print(f"{'评分':<4} {'股票':<8} {'名称':<22} {'价格':<10} {'跌幅':<8} {'RSI':<6} {'PE':<8}")
    print("-"*70)
    for r in results[:10]:
        print(f"{r['score']:<4} {r['ticker']:<8} {r['name']:<22} ${r['price']:<7,.2f} {r['drop']:<+7.1f}% {r['rsi']:<5.1f} {r['pe']:<8.1f}")

    # Save report
    today_dir = REPORTS_DIR / today
    today_dir.mkdir(parents=True, exist_ok=True)
    with open(today_dir / "report.json", "w") as f:
        json.dump({"date":today,"total":len(TICKERS),"found":len(results),"elapsed":elapsed,"results":results[:15]}, f, indent=2)
    print(f"\\n报告: reports/{today}/report.json")
    print(f"推荐: {results[0]['ticker']} (评分{results[0]['score']})" if results else "无符合条件的股票")

if __name__ == "__main__":
    main()
