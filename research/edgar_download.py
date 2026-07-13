#!/usr/bin/env python3
"""edgar_download.py — SEC EDGAR XBRL fundamentals via the FRAMES API (one request = all companies for a
concept+period). Pulls book equity (quarterly instantaneous), shares outstanding (quarterly), and annual net
income, for our universe, 2010-2026. Maps cik->ticker. Output tidy facts (ticker, concept, end, val) ->
data/edgar/facts.parquet. Point-in-time is applied downstream in VALUE.py via a conservative reporting LAG
(frames has no filed date; standard practice lags accounting data ~1-2 quarters to guarantee availability)."""
import warnings; warnings.filterwarnings("ignore")
import os, time, requests
import numpy as np, pandas as pd

HDR = {"User-Agent": "XFUND quant research xfund-research@proton.me"}; os.makedirs("data/edgar", exist_ok=True)
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
mpx = px.reindex(me); mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill")
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (mpx > 5) & (cov > 0.9) & (mdv > 5e6); uni = set(px.columns[elig.any(axis=0).values])

cmap = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HDR, timeout=30).json()
c2t = {}                                                                    # cik -> ticker (first/primary listing)
for v in cmap.values(): c2t.setdefault(int(v["cik_str"]), v["ticker"].upper())
up = {u.upper(): u for u in px.columns}                                     # upper -> original casing
print(f"[edgar] universe {len(uni)}, cik map {len(c2t)}", flush=True)

def frame(ns, tag, unit, period):
    try:
        r = requests.get(f"https://data.sec.gov/api/xbrl/frames/{ns}/{tag}/{unit}/{period}.json", headers=HDR, timeout=30)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception: return []

INST = [("us-gaap","StockholdersEquity","USD","book"), ("dei","EntityCommonStockSharesOutstanding","shares","shares"),
        ("us-gaap","CommonStockSharesOutstanding","shares","shares"), ("us-gaap","Assets","USD","assets")]
ANN  = [("us-gaap","NetIncomeLoss","USD","ni"), ("us-gaap","Revenues","USD","rev"),
        ("us-gaap","RevenueFromContractWithCustomerExcludingAssessedTax","USD","rev"), ("us-gaap","SalesRevenueNet","USD","rev"),
        ("us-gaap","CostOfRevenue","USD","cogs"), ("us-gaap","CostOfGoodsAndServicesSold","USD","cogs"), ("us-gaap","CostOfGoodsSold","USD","cogs")]
rows = []; nreq = 0
for Y in range(2010, 2027):
    for q in (1,2,3,4):
        for ns, tag, unit, concept in INST:
            for e in frame(ns, tag, unit, f"CY{Y}Q{q}I"):
                tk = c2t.get(e["cik"])
                if tk and tk in up and up[tk] in uni: rows.append((up[tk], concept, e["end"], e["val"]))
            nreq += 1; time.sleep(0.11)
    for ns, tag, unit, concept in ANN:                                     # annual (10-K FY) duration concepts
        for e in frame(ns, tag, unit, f"CY{Y}"):
            tk = c2t.get(e["cik"])
            if tk and tk in up and up[tk] in uni: rows.append((up[tk], concept, e["end"], e["val"]))
        nreq += 1; time.sleep(0.11)
    print(f"[edgar] {Y} done ({nreq} requests, {len(rows)} facts)", flush=True)

df = pd.DataFrame(rows, columns=["ticker","concept","end","val"]).drop_duplicates()
df.to_parquet("data/edgar/facts.parquet")
print(f"[edgar] DONE {df['ticker'].nunique()} tickers, {len(df)} facts -> data/edgar/facts.parquet", flush=True)
print(df.groupby("concept")["ticker"].nunique().to_string())
