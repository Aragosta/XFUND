#!/usr/bin/env python3
"""
tiingo_bulk.py — one-time bulk pull of the FULL Tiingo US equity EOD universe before the subscription lapses.

Grabs every US Stock+ETF (NYSE/NASDAQ/AMEX/ARCA/BATS incl. DELISTED = survivorship-bias-free), full history,
raw + adjusted OHLCV + splitFactor + divCash (format=csv). RESUMABLE: writes batched parquets + a manifest of
completed tickers; re-run to continue after any interruption. Throttled to the Power-plan limit (10k req/hr).

Env: TIINGO_TOKEN (required), OUTDIR (default data/tiingo_eod), BATCH (500), RPS (2.4 → ~8.6k/hr, under the cap),
     UNIVERSE ("us" default | "cn" for Shenzhen/Shanghai). Run:  TIINGO_TOKEN=... python3 research/tiingo_bulk.py
"""
import os, sys, io, time, json, zipfile, warnings
warnings.filterwarnings("ignore")
import requests, pandas as pd
sys.path.insert(0, "/Users/enzokreeft/XFUND"); os.chdir("/Users/enzokreeft/XFUND")

TOKEN = os.environ.get("TIINGO_TOKEN", "")
if not TOKEN:
    sys.exit("set TIINGO_TOKEN")
OUTDIR = os.environ.get("OUTDIR", "data/tiingo_eod"); os.makedirs(OUTDIR, exist_ok=True)
BATCH = int(os.environ.get("BATCH", 500)); RPS = float(os.environ.get("RPS", 2.4))
UNIVERSE = os.environ.get("UNIVERSE", "us")
UA = {"User-Agent": "XFUND research contact@xfund.example"}
MANIFEST = os.path.join(OUTDIR, "_done.json")


def catalog():
    r = requests.get("https://apimedia.tiingo.com/docs/tiingo/daily/supported_tickers.zip", headers=UA, timeout=90)
    z = zipfile.ZipFile(io.BytesIO(r.content)); df = pd.read_csv(z.open(z.namelist()[0]))
    if UNIVERSE == "cn":
        exch = ["SHE", "SHG"]
    else:
        exch = ["NYSE", "NASDAQ", "AMEX", "NYSE MKT", "NYSE ARCA", "BATS"]
    df = df[df["exchange"].isin(exch) & df["assetType"].isin(["Stock", "ETF"]) & (df["priceCurrency"] == "USD")]
    return sorted(df["ticker"].dropna().astype(str).unique().tolist())


def load_done():
    return set(json.load(open(MANIFEST))) if os.path.exists(MANIFEST) else set()


def save_done(done):
    json.dump(sorted(done), open(MANIFEST, "w"))


def main():
    tickers = catalog(); done = load_done()
    todo = [t for t in tickers if t not in done]
    if int(os.environ.get("LIMIT", 0)):
        todo = todo[:int(os.environ["LIMIT"])]
    print(f"[tiingo] universe={UNIVERSE} · {len(tickers)} tickers · {len(done)} done · {len(todo)} to fetch · RPS={RPS}", flush=True)
    sess = requests.Session(); sess.headers.update(UA)
    buf, nbatch, t0, delay = [], len([f for f in os.listdir(OUTDIR) if f.startswith("batch_")]), time.time(), 1.0 / RPS

    def flush(buf, nbatch, done):
        if not buf:
            return nbatch
        pd.concat(buf, ignore_index=True).to_parquet(os.path.join(OUTDIR, f"batch_{nbatch:04d}.parquet"))
        save_done(done)
        print(f"  wrote batch_{nbatch:04d} ({sum(len(b) for b in buf)} rows) · {len(done)} tickers done · "
              f"{len(done)/(time.time()-t0)*3600:.0f}/hr", flush=True)
        return nbatch + 1

    for i, tk in enumerate(todo):
        url = (f"https://api.tiingo.com/tiingo/daily/{tk}/prices?startDate=1990-01-01"
               f"&format=csv&token={TOKEN}")
        try:
            r = sess.get(url, timeout=40)
            if r.status_code == 429:                                        # rate limited → back off
                print("  429 rate-limit; sleeping 90s", flush=True); time.sleep(90); continue
            if r.status_code == 200 and len(r.text) > 50:
                d = pd.read_csv(io.StringIO(r.text)); d["ticker"] = tk
                buf.append(d)
            done.add(tk)
        except Exception:
            pass
        if len(buf) >= BATCH:
            nbatch = flush(buf, nbatch, done); buf = []
        time.sleep(delay)
    flush(buf, nbatch, done)
    print(f"[tiingo] DONE · {len(done)} tickers · batches in {OUTDIR}/", flush=True)


if __name__ == "__main__":
    main()
