#!/usr/bin/env python3
"""
build_sic.py — build a full sector/industry map (SIC codes) for the WIDER price universe, extending the existing
data/edgar/sic.parquet (which covered only 4,386 of 22,911 tickers). SEC's submissions endpoint carries `sic` +
`sicDescription` per filer. Same schema as the existing file (ticker, sic, sicDescription, sector2) so DATAHUB
picks it up unchanged. sector2 = 2-digit SIC prefix (the grouping key sleeves sector-neutralize on).

Merges: fresh SEC pull is authoritative; existing rows for tickers we can't refetch are preserved.
Run: python3 research/build_sic.py   (SEC rate-limited, ~18 min for ~7k filers)
"""
import os, sys, time, json, warnings
warnings.filterwarnings("ignore")
import requests, pandas as pd
sys.path.insert(0, "/Users/enzokreeft/XFUND"); os.chdir("/Users/enzokreeft/XFUND")
UA = {"User-Agent": "XFUND research contact@xfund.example"}

def main():
    r = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA, timeout=30)
    t2c = {v["ticker"].upper(): str(v["cik_str"]).zfill(10) for v in json.loads(r.text).values()}
    price_tickers = pd.read_parquet("data/daily.parquet")["close"].columns
    todo = [t for t in price_tickers if t.upper() in t2c]
    print(f"[sic] {len(todo)} price tickers map to a CIK; fetching submissions for SIC ...", flush=True)

    rows = []; sess = requests.Session(); sess.headers.update(UA); ok = 0; t0 = time.time()
    for i, tk in enumerate(todo):
        cik = t2c[tk.upper()]
        try:
            resp = sess.get(f"https://data.sec.gov/submissions/CIK{cik}.json", timeout=30)
            if resp.status_code == 200:
                j = json.loads(resp.text)
                sic = j.get("sic", "") or ""
                desc = j.get("sicDescription", "") or ""
                sector2 = str(sic).zfill(4)[:2] if str(sic).strip() and str(sic).isdigit() else ""
                rows.append((tk, sic, desc, sector2)); ok += 1
        except Exception:
            pass
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(todo)}  ok={ok}  {(i+1)/(time.time()-t0):.1f} req/s", flush=True)
        time.sleep(0.12)                                                 # ≤10 req/s SEC courtesy

    fresh = pd.DataFrame(rows, columns=["ticker", "sic", "sicDescription", "sector2"])
    # merge: fresh authoritative, keep existing rows for tickers not refetched
    old = pd.read_parquet("data/edgar/sic.parquet")
    keep_old = old[~old["ticker"].isin(fresh["ticker"])]
    out = pd.concat([fresh, keep_old], ignore_index=True).drop_duplicates("ticker", keep="first")
    out.to_parquet("data/edgar/sic.parquet")
    have_sector = (out["sector2"].str.len() == 2).sum()
    print(f"[sic] wrote data/edgar/sic.parquet: {len(out):,} tickers ({have_sector:,} with a valid 2-digit sector) "
          f"— was 4,386", flush=True)
    print("  top sectors:", out["sector2"].value_counts().head(8).to_dict(), flush=True)


if __name__ == "__main__":
    main()
