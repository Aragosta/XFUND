#!/usr/bin/env python3
"""Full daily download (adjClose + adjVolume) for the broad Tiingo universe, 2000-present.
Resumable, liquidity-prioritized (most-traded first, so partial runs are maximally useful).
Writes NEW parquets — leaves the monthly + 750-ticker daily checkpoints untouched.

  tiingo_daily_close.parquet   : daily adjClose panel  (dates x tickers)
  tiingo_daily_volume.parquet  : daily adjVolume panel (dates x tickers)

Reconstruct monthly with research/reconstruct_monthly.py once this finishes (or partially).
"""
import warnings; warnings.filterwarnings("ignore")
import os, time, threading
import pandas as pd, requests
from concurrent.futures import ThreadPoolExecutor, as_completed

API   = "102cb09d2f83b832d38f00437fd18de26e025d95"
MONTH = "/Users/enzokreeft/XFUND/tiingo_download_checkpoint.parquet"
OUT_C = "/Users/enzokreeft/XFUND/tiingo_daily_close.parquet"
OUT_V = "/Users/enzokreeft/XFUND/tiingo_daily_volume.parquet"
START = "2000-01-01"; END = pd.Timestamp.today().strftime("%Y-%m-%d")
N = int(os.environ.get("N_TICKERS", "0"))              # 0 = ALL; else cap to top-N by liquidity

ck = pd.read_parquet(MONTH)
liq = (ck["close"] * ck["volume"]).median(axis=0, skipna=True).sort_values(ascending=False)
tickers = liq.index.tolist()
if N > 0: tickers = tickers[:N]
print(f"[full] {len(tickers)} tickers (liquidity-sorted), {START}->{END}", flush=True)

close_done, vol_done = {}, {}
if os.path.exists(OUT_C):
    pc = pd.read_parquet(OUT_C); close_done = {t: pc[t].dropna() for t in pc.columns}
    if os.path.exists(OUT_V):
        pv = pd.read_parquet(OUT_V); vol_done = {t: pv[t].dropna() for t in pv.columns}
    print(f"[full] resume: {len(close_done)} tickers already present", flush=True)
remaining = [t for t in tickers if t not in close_done]
print(f"[full] {len(remaining)} to fetch", flush=True)

BASE = "https://api.tiingo.com/tiingo/daily"; HEAD = {"Authorization": f"Token {API}", "Content-Type": "application/json"}
lock = threading.Lock(); last429 = [0.0]

def fetch(t):
    url = f"{BASE}/{t}/prices?startDate={START}&endDate={END}&token={API}"
    for attempt in range(5):
        with lock:
            wait = last429[0] + 65 - time.time()
        if wait > 0: time.sleep(wait)
        try:
            r = requests.get(url, headers=HEAD, timeout=30)
            if r.status_code == 404: return t, None, None
            if r.status_code == 429:
                with lock: last429[0] = time.time()
                time.sleep(65); continue
            r.raise_for_status(); j = r.json()
            if not j: return t, None, None
            df = pd.DataFrame(j); df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.set_index("date")
            return t, df["adjClose"], df.get("adjVolume", df.get("volume"))
        except Exception:
            time.sleep(2 ** attempt)
    return t, None, None

def save():
    for out, dd in [(OUT_C, close_done), (OUT_V, vol_done)]:
        tmp = out + ".tmp"; pd.DataFrame(dd).sort_index().to_parquet(tmp); os.replace(tmp, out)

cnt = 0
with ThreadPoolExecutor(max_workers=20) as ex:
    futs = {ex.submit(fetch, t): t for t in remaining}
    for f in as_completed(futs):
        t, c, v = f.result()
        if c is not None and len(c):
            close_done[t] = c
            if v is not None: vol_done[t] = v
        cnt += 1
        if cnt % 200 == 0:
            print(f"[full] {cnt}/{len(remaining)}  ({len(close_done)} with data)", flush=True); save()
save()
print(f"[full] DONE close={pd.DataFrame(close_done).shape} vol={pd.DataFrame(vol_done).shape}", flush=True)
