"""
深度价值选股 — 每日扫 200+ 只股票, 找跌深+有价值的标的
输出包含: 业务简介、估值分析、成长性、盈利能力、分析师评级
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
    "BA","CAT","GE","MMM","HON","LMT","RTX","NOC","GD",
    "XOM","CVX","COP","SLB","OXY","EOG","HAL","DVN",
    "F","GM","RIVN","LCID","NIO","LI",
    "T","VZ","CMCSA","CHTR","C",
    "WFC","BAC","GS","MS","SCHW","BLK","BX",
    "ABT","TMO","MDT","SYK","BSX","EW",
    "COST","TGT","LOW","ROST","TJX","SBUX",
    "NKE","MCD","DEO","PEP","KO","MDLZ",
    "SPGI","MCO","FI","FIS","JKHY",
    "NOW","WDAY","TEAM","ZS","PANW","FTNT",
    "DE","LIN","SHW","ECL",
    "LULU","DECK","CROX","SKX","BABA",
    "HCA","UHS","DVA","THC","ACM","TTEK",
]

REPORTS_DIR = Path(__file__).parent.parent / "reports"


def value_brief(info, price, r):
    """生成价值分析文字"""
    lines = []

    # 业务简介
    biz = info.get("longBusinessSummary", "")
    if biz:
        short_biz = biz[:180].split(".")[0] + "."
        lines.append(short_biz)

    # 股票位置
    h52 = info.get("fiftyTwoWeekHigh", 0)
    l52 = info.get("fiftyTwoWeekLow", 0)
    if h52 and l52:
        pct = (price / h52 - 1) * 100
        lines.append(f"52周范围: ${l52:.0f} ~ ${h52:.0f}, 当前在低位{pct:.0f}%分位")

    # 估值
    pe = info.get("trailingPE", 0)
    fpe = info.get("forwardPE", 0)
    if pe:
        val = f"PE={pe:.1f}"
        if fpe:
            val += f", 前瞻PE={fpe:.1f}"
            if fpe < pe: val += " (估值在改善)"
        lines.append(f"估值: {val}")

    # 成长
    rg = info.get("revenueGrowth", 0)
    eg = info.get("earningsGrowth", 0)
    if rg:
        lines.append(f"成长: 营收增{rg*100:.0f}%" + (f", 盈利增{eg*100:.0f}%" if eg else ""))

    # 盈利能力
    roe = info.get("returnOnEquity", 0)
    pm = info.get("profitMargins", 0)
    if pm:
        lines.append(f"盈利: 利润率{pm*100:.1f}%, ROE={roe*100:.1f}%" if roe else f"盈利: 利润率{pm*100:.1f}%")

    # 分析师评级
    rec = info.get("recommendationMean", 0)
    target = info.get("targetMeanPrice", 0)
    na = info.get("numberOfAnalystOpinions", 0)
    if rec and target and na:
        labels = {1:"强力买入",2:"买入",3:"持有",4:"卖出",5:"强力卖出"}
        upside = (target / price - 1) * 100
        lines.append(f"分析师: {na}家评级{labels.get(round(rec),'')}, 目标${target:.0f} (上行空间{upside:.0f}%)")

    # 风险
    sr = info.get("shortRatio", 0)
    if sr and sr > 5:
        lines.append(f"风险: 做空比例偏高({sr:.1f}天)")

    return "\n".join(lines)


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
        nm = (info.get("shortName", t) or t)[:40]
        sec = info.get("sector", "") or ""
        ind = info.get("industry", "") or ""

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

        r = {"ticker":t,"name":nm,"score":score,"price":round(p,2),"drop":round(drop,1),
             "rsi":round(rsi,1),"pe":round(pe,1),"pb":round(pb,2),"de":round(de,1),
             "mc_b":round(mc/1e9,1),"sector":sec,"industry":ind}
        r["value_brief"] = value_brief(info, p, r)
        return r
    except: return None


def main():
    import pandas as pd
    start = datetime.now(); results = []
    with ThreadPoolExecutor(max_workers=10) as exe:
        for f in as_completed({exe.submit(analyze, t): t for t in TICKERS if t}):
            r = f.result()
            if r: results.append(r)
    results.sort(key=lambda x: -x["score"])
    el = (datetime.now()-start).seconds
    today = datetime.now().strftime("%Y-%m-%d")

    for r in results[:3]:
        print("="*62)
        print(f"  {r['ticker']} — {r['name']}  (评分{r['score']}/15)")
        print(f"  行业: {r['sector']} > {r['industry']}")
        print(f"  价格: ${r['price']:.2f}  |  从高点跌了 {r['drop']:+.1f}%  |  RSI: {r['rsi']:.1f}")
        print(r["value_brief"])
        print()

    print(f"扫描 {len(TICKERS)} 只, 找到 {len(results)} 只, 耗时 {el}s")
    print(f"完整报告: reports/{today}/report.json")
    if results: print(f"推荐关注: {results[0]['ticker']} (评分{results[0]['score']})")

    # 保存
    td = REPORTS_DIR / today
    td.mkdir(parents=True, exist_ok=True)
    with open(td / "report.json", "w") as f:
        json.dump({"date":today,"total":len(TICKERS),"found":len(results),
                   "elapsed":el,"results":results[:10]}, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
