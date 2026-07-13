#!/usr/bin/env python3
"""repull_raw.py — fetch RAW (unadjusted) monthly close from Tiingo for the fundamentals universe, so mcap =
raw_close * EDGAR_shares is correct by construction (both as-reported basis) and VALUE becomes measurable.
Our stored tiingo_daily_close is split-ADJUSTED (right for returns, wrong for mcap). This adds the raw panel.
Checkpointed + resumable. Background job."""
import warnings; warnings.filterwarnings("ignore")
import os, time, requests, numpy as np, pandas as pd

KEY = "102cb09d2f83b832d38f00437fd18de26e025d95"
OUT = "data/tiingo_raw_monthly.parquet"
HDR = {"Authorization": f"Token {KEY}", "Content-Type": "application/json"}

px = pd.read_parquet("tiingo_daily_close.parquet")
edgar_tk = set(pd.read_parquet("data/edgar/facts.parquet")["ticker"].unique())
universe = sorted(set(px.columns) & edgar_tk)                                   # ~fundamentals-covered names
print(f"[repull] target {len(universe)} tickers (px ∩ EDGAR)", flush=True)

done = {}
if os.path.exists(OUT):
    prev = pd.read_parquet(OUT); done = {c: prev[c] for c in prev.columns}      # resume
    print(f"[repull] resuming, {len(done)} already fetched", flush=True)

cols = {**done}; todo = [t for t in universe if t not in cols]
for i, tk in enumerate(todo):
    try:
        u = f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate=2000-01-01&resampleFreq=monthly&columns=close&token={KEY}"
        r = requests.get(u, headers=HDR, timeout=20)
        if r.status_code == 200 and r.json():
            df = pd.DataFrame(r.json()); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            cols[tk] = df.set_index("date")["close"]                            # RAW close
    except Exception:
        pass
    if (i+1) % 200 == 0:
        pd.DataFrame(cols).to_parquet(OUT); print(f"[repull] {i+1}/{len(todo)} ({tk})  saved {len(cols)}", flush=True)
    time.sleep(0.03)
pd.DataFrame(cols).sort_index().to_parquet(OUT)
print(f"[repull] DONE — {len(cols)} tickers -> {OUT}", flush=True)
