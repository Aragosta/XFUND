#!/usr/bin/env python3
"""
edgar_build2.py — SECOND PASS: extend EDGAR fundamentals coverage to the NEW tickers added by the Tiingo bulk pull
(research/tiingo_bulk.py), which are not in the original price universe (data/daily.parquet) that edgar_build.py
covered. Reuses the same SEC company_tickers.json CIK map + the same concept/tag logic as edgar_build.py.

Waits for BOTH source jobs (Tiingo manifest complete AND facts_v2.parquet on disk) before running, then downloads
fundamentals for (new tickers) MINUS (already in facts_v2), and merges into ONE consolidated
data/edgar/facts_v2.parquet — no duplicate work on the 5,099 tickers edgar_build.py already pulled.

Run:  python3 research/edgar_build2.py   (safe to run any time; it waits + is idempotent via the merge)
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import requests, pandas as pd
sys.path.insert(0, "/Users/enzokreeft/XFUND"); os.chdir("/Users/enzokreeft/XFUND")
from edgar_build import STOCK, FLOW, _pick, UA          # reuse the exact same tag logic as pass 1

V2 = "data/edgar/facts_v2.parquet"
TIINGO_MANIFEST = "data/tiingo_eod/_done.json"
TIINGO_UNIVERSE = 22996                                  # from the catalog() call in tiingo_bulk.py (US Stock+ETF)


def wait_for_sources(poll=60, stable_checks=3):
    """Wait for facts_v2.parquet AND the Tiingo manifest to stop growing (some tickers will always error
    out — timeouts/bad symbols — so we can't wait for an exact count; wait for it to plateau instead)."""
    last, stable = -1, 0
    while True:
        edgar_v1_done = os.path.exists(V2)
        n = len(json.load(open(TIINGO_MANIFEST))) if os.path.exists(TIINGO_MANIFEST) else 0
        pct = 100 * n / TIINGO_UNIVERSE
        stable = stable + 1 if n == last else 0
        last = n
        tiingo_done = (n >= TIINGO_UNIVERSE) or (pct >= 98.0) or (stable >= stable_checks)
        if tiingo_done and edgar_v1_done:
            print(f"[edgar2] sources ready: tiingo={n}/{TIINGO_UNIVERSE} ({pct:.1f}%, stable={stable}) edgar_v1=True", flush=True)
            return
        print(f"[edgar2] waiting: tiingo={n}/{TIINGO_UNIVERSE} ({pct:.1f}%, stable={stable}/{stable_checks}) edgar_v1_done={edgar_v1_done}", flush=True)
        time.sleep(poll)


def main():
    wait_for_sources()
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
    t2c = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in json.loads(r.text).values()}

    new_price_tickers = set(json.load(open(TIINGO_MANIFEST)))
    v2 = pd.read_parquet(V2)
    have = set(v2["ticker"].str.upper().unique())
    todo = sorted(t for t in new_price_tickers if t.upper() in t2c and t.upper() not in have)
    print(f"[edgar2] {len(new_price_tickers)} tiingo tickers · {len(todo)} NEW tickers with a CIK not yet fetched", flush=True)

    rows = []; sess = requests.Session(); sess.headers.update(UA); ok = 0; t0 = time.time()
    for i, tk in enumerate(todo):
        cik = t2c[tk.upper()]
        try:
            resp = sess.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json", timeout=30)
            if resp.status_code != 200:
                continue
            allfacts = json.loads(resp.text).get("facts", {})
            gaap = allfacts.get("us-gaap", {}); dei = allfacts.get("dei", {})
            ubt = {}
            for tag, node in gaap.items():
                units = node.get("units", {})
                ubt[tag] = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
            for tag, node in dei.items():
                units = node.get("units", {})
                ubt[f"dei:{tag}"] = units.get("shares") or units.get("USD") or next(iter(units.values()), [])
            for concept, tags in {**STOCK, **FLOW}.items():
                for end, filed, val, form in _pick(ubt, tags, annual=(concept in FLOW)):
                    rows.append((tk, concept, end, filed, val, form))
            ok += 1
        except Exception:
            pass
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(todo)}  ok={ok}  rows={len(rows)}  {(i+1)/(time.time()-t0):.1f} req/s", flush=True)
        time.sleep(0.11)

    if rows:
        new_df = pd.DataFrame(rows, columns=["ticker", "concept", "end", "filed", "val", "form"])
        new_df["end"] = pd.to_datetime(new_df["end"]); new_df["filed"] = pd.to_datetime(new_df["filed"])
        merged = pd.concat([v2, new_df], ignore_index=True)
        merged = merged.sort_values("filed").drop_duplicates(["ticker", "concept", "end"], keep="last")
        merged.to_parquet(V2)
        print(f"[edgar2] merged +{len(new_df)} rows, +{new_df.ticker.nunique()} new tickers "
              f"-> {V2}: {len(merged)} rows total, {merged.ticker.nunique()} tickers", flush=True)
    else:
        print("[edgar2] no new rows fetched", flush=True)


if __name__ == "__main__":
    main()
