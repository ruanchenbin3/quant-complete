
import yfinance as yf, pandas as pd, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
import json, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

TICKERS = [
    "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V",
    "JNJ","WMT","PG","MA","UNH","HD","DIS","NFLX","ADBE","CRM",
    "INTC","AMD","IBM","CSCO","QCOM","TXN","AVGO","ORCL",
    "PYPL","UBER","ABNB","DASH","MDB","CRWD","ZS",
    "PFE","MRNA","BNTX","GILD","AMGN","BIIB","REGN","VRTX","ABBV",
    "BA","CAT","GE","MMM","HON","LMT","RTX",
    "XOM","CVX","COP","SLB","OXY","EOG","HAL","DVN",
    "F","GM","NIO","LI",
    "T","VZ","C","WFC","BAC","GS","MS","SCHW","BLK","BX",
    "ABT","TMO","MDT","SYK","BSX","EW",
    "COST","TGT","LOW","ROST","TJX","SBUX",
    "NKE","MCD","DEO","PEP","KO","MDLZ",
    "SPGI","MCO","FI","FIS",
    "NOW","WDAY","ZS","PANW","FTNT",
    "DE","LIN","SHW","ECL",
    "LULU","BABA","HCA","UHS","THC","ACM",
]

REPORTS_DIR = Path(__file__).parent.parent / "reports"

def analyze(t):
    try:
        df = yf.download(t, period="1y", progress=False)
        if isinstance(df.columns, type(pd.Index([]))): df.columns = df.columns.droplevel(1)
        if len(df) < 60: return None
        c = df["Close"]; p = float(c.iloc[-1]); hi = float(c.max())
        drop = (p / hi - 1) * 100
        if drop > -15: return None
        info = yf.Ticker(t).info
        pe = info.get("trailingPE", 0) or 0; pb = info.get("priceToBook", 0) or 0
        de = info.get("debtToEquity", 0) or 0; mc = info.get("marketCap", 0) or 0
        nm = (info.get("shortName", t) or t)[:40]
        sec = info.get("sector", "") or ""; ind = info.get("industry", "") or ""
        biz = (info.get("longBusinessSummary", "") or "")[:200]
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
                "rsi":round(rsi,1),"pe":round(pe,1),"pb":round(pb,2),"de":round(de,1),
                "mc_b":round(mc/1e9,1),"sector":sec,"industry":ind,"biz":biz}
    except: return None

start = datetime.now(); results = []
with ThreadPoolExecutor(max_workers=8) as exe:
    for f in as_completed({exe.submit(analyze, t): t for t in TICKERS if t}):
        r = f.result()
        if r: results.append(r)
results.sort(key=lambda x: -x["score"])
el = (datetime.now()-start).seconds
today = datetime.now().strftime("%Y-%m-%d")

for r in results[:3]:
    print("="*60)
    print(f"  {r['ticker']} - {r['name']} (Score {r['score']}/15)")
    print(f"  Price ${r['price']:.2f} | Drop {r['drop']:+.1f}% | RSI {r['rsi']:.1f} | PE {r['pe']:.1f}")

td = REPORTS_DIR / today; td.mkdir(parents=True, exist_ok=True)
with open(td / "report.json", "w") as f:
    json.dump({"date":today,"total":len(TICKERS),"found":len(results),"elapsed":el,"results":results[:10]}, f, indent=2, ensure_ascii=False)

lines_en = [f"# Deep Value Screener - {today}", "", f"Scanned {len(TICKERS)} stocks, found {len(results)} candidates.", ""]
lines_zh = [f"# 深度价值选股 - {today}", "", f"扫描 {len(TICKERS)} 只股票, 找到 {len(results)} 只。", ""]

for r in results[:3]:
    t = r["ticker"]
    df = yf.download(t, period="1y", progress=False)
    if isinstance(df.columns, type(pd.Index([]))): df.columns = df.columns.droplevel(1)
    c = df["Close"]; p = float(c.iloc[-1]); hi = float(c.max()); lo = float(c.min())
    d = c.diff(); g = d.clip(lower=0).rolling(14).mean(); ll = -d.clip(upper=0).rolling(14).mean()
    rsi_s = 100 - 100/(1+g/ll.replace(0,1e-10))
    fig, (a1, a2) = plt.subplots(2,1,figsize=(10,6),gridspec_kw={"height_ratios":[3,1]})
    a1.plot(df.index, c, "k-", lw=1.5, label=t)
    a1.axhline(hi, color="r", ls=":", alpha=0.4, label=f"52wH ${hi:.0f}")
    a1.axhline(lo, color="g", ls=":", alpha=0.4, label=f"52wL ${lo:.0f}")
    a1.axhline(p, color="b", ls="--", alpha=0.6, label=f"Now ${p:.0f}")
    a1.set_title(f"{t} | ${p:.0f} | Drop {(p/hi-1)*100:.1f}% | RSI {float(rsi_s.iloc[-1]):.0f}")
    a1.legend(fontsize=8); a1.grid(alpha=0.3)
    a2.plot(df.index, rsi_s, "purple", lw=1)
    a2.axhline(70, color="r", ls=":", alpha=0.5); a2.axhline(30, color="g", ls=":", alpha=0.5)
    a2.set_ylabel("RSI"); a2.set_ylim(0,100); a2.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(td / f"{t}_chart.png", dpi=130); plt.close()
    
    rv = r["rsi"]; dv = r["drop"]
    tech_en = "Extremely oversold." if rv < 20 else "Oversold." if rv < 30 else "Near oversold." if rv < 40 else "Neutral."
    tech_zh = "极度超卖。" if rv < 20 else "超卖区。" if rv < 30 else "接近超卖。" if rv < 40 else "中性。"
    lines_en += [f"## {t} - {r['name']}", "", f"Price ${r['price']:.2f} | Drop {dv:+.1f}% | RSI {rv:.1f} | PE {r['pe']:.1f}", "", f"Technical: {tech_en} | PE {r['pe']:.1f}" + (f" | Low debt {r['de']:.0f}%" if r['de'] < 30 else ""), "", f"![{t}]({t}_chart.png)", "", "---", ""]
    lines_zh += [f"## {t} - {r['name']} (评分{r['score']}/15)", "", f"价格 ${r['price']:.2f} | 跌幅 {dv:+.1f}% | RSI {rv:.1f} | PE {r['pe']:.1f}", "", f"技术面: {tech_zh} | PE {r['pe']:.1f}" + (f" | 负债率低 {r['de']:.0f}%" if r['de'] < 30 else ""), "", f"![{t} K线图]({t}_chart.png)", "", "---", ""]

lines_en.append("*For educational purposes only.*")
lines_zh.append("*仅供学习研究。*")

with open(td / "README.md", "w") as f: f.write("\n".join(lines_en))
with open(td / "README.zh.md", "w") as f: f.write("\n".join(lines_zh))
print(f"\nSaved to {td}/")
for f in os.listdir(td): print(f"  {f}")
