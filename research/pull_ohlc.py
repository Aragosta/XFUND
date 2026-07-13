#!/usr/bin/env python3
"""pull_ohlc.py — fetch daily adjOpen/adjHigh/adjLow (+ divCash, splitFactor) from Tiingo for the tradeable
universe. adjHigh/adjLow unlock RANGE-BASED volatility (Garman-Klass / Parkinson) — a far better realised-vol
estimator than close-to-close, feeding the risk model. divCash -> total return; splitFactor -> clean corp actions.
Stored as per-field wide panels data/tiingo_ohlc/{field}.parquet. Checkpointed + resumable. Long background job."""
import warnings; warnings.filterwarnings("ignore")
import os, time, requests, numpy as np, pandas as pd

KEY = "102cb09d2f83b832d38f00437fd18de26e025d95"; HDR = {"Authorization": f"Token {KEY}"}
os.makedirs("data/tiingo_ohlc", exist_ok=True)
FIELDS = ["close", "adjHigh", "adjLow", "adjOpen", "divCash", "splitFactor"]     # close = RAW price (daily), incl. free
OUT = {f: f"data/tiingo_ohlc/{f}.parquet" for f in FIELDS}

px = pd.read_parquet("tiingo_daily_close.parquet")
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
universe = sorted(px.columns[px.notna().sum() >= 250])                          # tradeable-ever names
print(f"[ohlc] target {len(universe)} tickers", flush=True)

panels = {f: {} for f in FIELDS}
if os.path.exists(OUT["adjHigh"]):                                             # resume
    for f in FIELDS:
        if os.path.exists(OUT[f]):
            prev = pd.read_parquet(OUT[f]); panels[f] = {c: prev[c] for c in prev.columns}
    print(f"[ohlc] resuming, {len(panels['adjHigh'])} done", flush=True)

todo = [t for t in universe if t not in panels["adjHigh"]]
for i, tk in enumerate(todo):
    try:
        u = (f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate=2000-01-01"
             f"&columns=close,adjOpen,adjHigh,adjLow,divCash,splitFactor&token={KEY}")
        r = requests.get(u, headers=HDR, timeout=25)
        if r.status_code == 200 and r.json():
            df = pd.DataFrame(r.json()); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None); df = df.set_index("date")
            for f in FIELDS:
                if f in df.columns: panels[f][tk] = df[f]
    except Exception: pass
    if (i+1) % 200 == 0:
        for f in FIELDS: pd.DataFrame(panels[f]).sort_index().to_parquet(OUT[f])
        print(f"[ohlc] {i+1}/{len(todo)} ({tk}) saved {len(panels['adjHigh'])}", flush=True)
    time.sleep(0.03)
for f in FIELDS: pd.DataFrame(panels[f]).sort_index().to_parquet(OUT[f])
print(f"[ohlc] DONE — {len(panels['adjHigh'])} tickers, fields {FIELDS} -> data/tiingo_ohlc/", flush=True)
