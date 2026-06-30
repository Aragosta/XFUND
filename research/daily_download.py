#!/usr/bin/env python3
"""Bounded daily-return download for the DRIF prototype: top liquid names, 2005-present.
Writes a SEPARATE daily checkpoint so the monthly one is untouched. Resumable."""
import warnings; warnings.filterwarnings("ignore")
import os, time, threading
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API   = "102cb09d2f83b832d38f00437fd18de26e025d95"
MONTH = "/Users/enzokreeft/XFUND/tiingo_download_checkpoint.parquet"
OUT   = "/Users/enzokreeft/XFUND/tiingo_daily_checkpoint.parquet"
N, START = 750, "2005-01-01"
END = pd.Timestamp.today().strftime("%Y-%m-%d")

ck = pd.read_parquet(MONTH)
dollar = (ck["close"] * ck["volume"])
top = dollar.median(axis=0, skipna=True).sort_values(ascending=False).head(N).index.tolist()
print(f"[daily] selected {len(top)} liquid tickers by median $-volume", flush=True)

done = {}
if os.path.exists(OUT):
    prev = pd.read_parquet(OUT)
    for t in prev.columns:
        done[t] = prev[t].dropna()
    print(f"[daily] resume: {len(done)} already present", flush=True)
remaining = [t for t in top if t not in done]
print(f"[daily] {len(remaining)} to fetch ({START}–{END})", flush=True)

BASE = "https://api.tiingo.com/tiingo/daily"
HEAD = {"Authorization": f"Token {API}", "Content-Type": "application/json"}
lock = threading.Lock(); last429 = [0.0]

def fetch(t):
    url = f"{BASE}/{t}/prices?startDate={START}&endDate={END}&token={API}"
    for attempt in range(5):
        with lock:
            wait = last429[0] + 65 - time.time()
        if wait > 0:
            time.sleep(wait)
        try:
            r = requests.get(url, headers=HEAD, timeout=30)
            if r.status_code == 404:
                return t, None
            if r.status_code == 429:
                with lock:
                    last429[0] = time.time()
                time.sleep(65); continue
            r.raise_for_status()
            d = r.json()
            if not d:
                return t, None
            df = pd.DataFrame(d)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            return t, df.set_index("date")["adjClose"]
        except Exception:
            time.sleep(2 ** attempt)
    return t, None

def save():
    tmp = OUT + ".tmp"
    pd.DataFrame(done).sort_index().to_parquet(tmp)
    os.replace(tmp, OUT)

cnt = 0
with ThreadPoolExecutor(max_workers=20) as ex:
    futs = {ex.submit(fetch, t): t for t in remaining}
    for f in as_completed(futs):
        t, s = f.result()
        if s is not None and len(s):
            done[t] = s
        cnt += 1
        if cnt % 100 == 0:
            print(f"[daily] {cnt}/{len(remaining)}  ({len(done)} with data)", flush=True)
            save()

save()
panel = pd.DataFrame(done).sort_index()
print(f"[daily] DONE {panel.shape}  {panel.index.min().date()}–{panel.index.max().date()} -> {OUT}", flush=True)
