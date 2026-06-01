"""
深度价值选股 — 每日扫 200+ 只股票, 找跌深+有价值的标的
"""
import yfinance as yf
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
    "LULU","DECK","CROX","SKX","ONON","BABA",
    "HCA","UHS","DVA","THC","ACM","TTEK",
]

REPORTS_DIR = Path(__file__).parent.parent / "reports"

def build_note(r):
    note = ""
    if r["rsi"] < 25: note += "极度超卖！历史上这种位置反弹概率较高。"
    elif r["rsi"] < 35: note += "超卖区，关注是否止跌。"
    if r["pe"] and r["pe"] > 0 and r["pe"] < 10: note += "PE极低，估值有安全边际。"
    if r["de"] < 30: note += "负债率低，财务稳健。"
    if r["mc_b"] > 100: note += "大盘股，流动性好。"
    return note

def analyze(t):
    try:
        df = yf.download(t, period="6mo", progress=False)
        if isinstance(df.columns, type(pd.Index([]))):
            df.columns = df.columns.droplevel(1)
        if len(df) < 60: return None
        c = df["Close"]; p = float(c.iloc[-1]); hi = float(c.max())
        drop = (p / hi - 1) * 100
        if drop > -15: return None
        info = yf.Ticker(t).info
        pe = info.get("trailingPE", 0) or 0
        pb = info.get("priceToBook", 0) or 0
        de = info.get("debtToEquity", 0) or 0
        mc = info.get("marketCap", 0) or 0
        nm = (info.get("shortName", t) or t)[:45]
        sec = info.get("sector", "") or ""
        ind = info.get("industry", "") or ""
        summ = (info.get("longBusinessSummary", "") or "")[:300]
        d = c.diff(); g = d.clip(lower=0).rolling(14).mean(); lo = -d.clip(upper=0).rolling(14).mean()
        rsi = float((100 - 100/(1+g/lo.replace(0,1e-10))).iloc[-1])
        score = 0
        if drop < -40: score += 4
        elif drop < -30: score += 3
        elif drop < -20: score += 2
        if 5 < pe < 20: score += 3
        if 0 < pb < 2: score += 2
        elif pb < 4: score += 1
        if de < 50: score += 2
        elif de < 100: score += 1
        if rsi < 30: score += 2
        elif rsi < 40: score += 1
        if mc > 10e9: score += 1
        return {"ticker":t,"name":nm,"score":score,"price":round(p,2),"drop":round(drop,1),
                "rsi":round(rsi,1),"pe":round(pe,1),"pb":round(pb,2),"de":round(de,1),"mc_b":round(mc/1e9,1),
                "sector":sec,"industry":ind,"summary":summ,"note":""}
    except: return None

def main():
    import pandas as pd
    start = datetime.now(); results = []
    with ThreadPoolExecutor(max_workers=10) as exe:
        for f in as_completed({exe.submit(analyze, t): t for t in TICKERS if t}):
            r = f.result()
            if r: r["note"] = build_note(r); results.append(r)
    results.sort(key=lambda x: -x["score"])
    el = (datetime.now()-start).seconds
    today = datetime.now().strftime("%Y-%m-%d")

    print(f"深度价值选股 | {today}")
    print(f"扫描 {len(TICKERS)} 只, 找到 {len(results)} 只, 耗时 {el}s")
    print()
    for r in results[:3]:
        print("="*62)
        print(f"  {r['ticker']} — {r['name']}")
        print(f"  评分: {r['score']}/15 | 行业: {r['sector']} > {r['industry']}")
        print(f"  价格: ${r['price']:.2f}  从高点跌了 {r['drop']:+.1f}%")
        print(f"  RSI: {r['rsi']:.1f}  PE: {r['pe']:.1f}  PB: {r['pb']:.2f}  负债率: {r['de']:.0f}%")
        if r['note']: print(f"  {r['note']}")
        if r['summary']: print(f"  简介: {r['summary'][:200]}...")
        print()

    print("="*62)
    print(f"完整报告: reports/{today}/report.json")
    if results:
        print(f"推荐关注: {results[0]['ticker']} (评分{results[0]['score']})")

    # Save
    td = REPORTS_DIR / today
    td.mkdir(parents=True, exist_ok=True)
    with open(td / "report.json", "w") as f:
        json.dump({"date":today,"total":len(TICKERS),"found":len(results),"elapsed":el,"results":results[:10]}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
