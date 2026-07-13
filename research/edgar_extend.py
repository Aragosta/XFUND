#!/usr/bin/env python3
"""edgar_extend.py — extend the EDGAR pull with (1) SIC industry codes (fixes value's sector-bet problem;
enables sector-neutral construction) and (2) more fundamentals for the ML value/quality model: operating cash
flow, debt, R&D, inventory, receivables, current assets/liabs, cash, capex, dividends, gross profit.
SIC via the submissions API (per-CIK). New concepts via FRAMES. Appends to facts.parquet, SIC -> sic.parquet."""
import warnings; warnings.filterwarnings("ignore")
import os, time, requests, numpy as np, pandas as pd

HDR = {"User-Agent": "XFUND quant research xfund-research@proton.me"}
px = pd.read_parquet("tiingo_daily_close.parquet")
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
mpx = px.reindex(me); mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill")
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (mpx > 5) & (cov > 0.9) & (mdv > 5e6); uni = set(px.columns[elig.any(axis=0).values])
cmap = requests.get("https://www.sec.gov/files/company_tickers.json", headers=HDR, timeout=30).json()
c2t, t2c = {}, {}
for v in cmap.values():
    cik = int(v["cik_str"]); tk = v["ticker"].upper(); c2t.setdefault(cik, tk); t2c.setdefault(tk, cik)
up = {u.upper(): u for u in px.columns}
def frame(ns, tag, unit, period):
    try:
        r = requests.get(f"https://data.sec.gov/api/xbrl/frames/{ns}/{tag}/{unit}/{period}.json", headers=HDR, timeout=30)
        return r.json().get("data", []) if r.status_code == 200 else []
    except Exception: return []

# ---- (1) SIC industry per ticker (submissions API) ----
print("[sic] fetching industry codes ...", flush=True); sicrows = []
unis = sorted(t for t in uni)
for i, tk in enumerate(unis):
    cik = t2c.get(tk.upper())
    if cik is None: continue
    try:
        r = requests.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json", headers=HDR, timeout=20)
        if r.status_code == 200:
            j = r.json(); sicrows.append((tk, j.get("sic",""), j.get("sicDescription","")))
    except Exception: pass
    if (i+1) % 500 == 0: print(f"[sic] {i+1}/{len(unis)}", flush=True)
    time.sleep(0.09)
sic = pd.DataFrame(sicrows, columns=["ticker","sic","sicDescription"])
sic["sector2"] = sic["sic"].astype(str).str[:2]                                # 2-digit SIC major group ~ sector
sic.to_parquet("data/edgar/sic.parquet"); print(f"[sic] {len(sic)} tickers -> data/edgar/sic.parquet", flush=True)

# ---- (2) extended fundamentals ----
INST = [("us-gaap","LongTermDebtNoncurrent","USD","debt_lt"), ("us-gaap","LongTermDebt","USD","debt_lt"),
        ("us-gaap","InventoryNet","USD","inventory"), ("us-gaap","AccountsReceivableNetCurrent","USD","receivables"),
        ("us-gaap","AssetsCurrent","USD","assets_cur"), ("us-gaap","LiabilitiesCurrent","USD","liab_cur"),
        ("us-gaap","CashAndCashEquivalentsAtCarryingValue","USD","cash")]
ANN  = [("us-gaap","NetCashProvidedByUsedInOperatingActivities","USD","ocf"),
        ("us-gaap","ResearchAndDevelopmentExpense","USD","rnd"), ("us-gaap","GrossProfit","USD","gross_profit"),
        ("us-gaap","PaymentsToAcquirePropertyPlantAndEquipment","USD","capex"),
        ("us-gaap","PaymentsOfDividendsCommonStock","USD","dividends"), ("us-gaap","PaymentsOfDividends","USD","dividends")]
rows = []; nreq = 0
for Y in range(2010, 2027):
    for q in (1,2,3,4):
        for ns, tag, unit, concept in INST:
            for e in frame(ns, tag, unit, f"CY{Y}Q{q}I"):
                tk = c2t.get(e["cik"])
                if tk and tk in up and up[tk] in uni: rows.append((up[tk], concept, e["end"], e["val"]))
            nreq += 1; time.sleep(0.11)
    for ns, tag, unit, concept in ANN:
        for e in frame(ns, tag, unit, f"CY{Y}"):
            tk = c2t.get(e["cik"])
            if tk and tk in up and up[tk] in uni: rows.append((up[tk], concept, e["end"], e["val"]))
        nreq += 1; time.sleep(0.11)
    print(f"[edgar+] {Y} done ({nreq} req, {len(rows)} new facts)", flush=True)
ext = pd.DataFrame(rows, columns=["ticker","concept","end","val"]).drop_duplicates()
old = pd.read_parquet("data/edgar/facts.parquet")
pd.concat([old, ext], ignore_index=True).drop_duplicates().to_parquet("data/edgar/facts.parquet")
print(f"[edgar+] appended {len(ext)} facts ({ext['concept'].nunique()} new concepts) -> facts.parquet", flush=True)
print(ext.groupby("concept")["ticker"].nunique().to_string())
