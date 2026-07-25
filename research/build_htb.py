#!/usr/bin/env python3
"""build_htb.py — HARD-TO-BORROW proxy panel, full DataHub window & universe.

WHY: T39/T40 concluded (4× independently) that momentum has no short-side premium and that a positive
market-neutral book needs a SHORT leg carrying a DIFFERENT premium. The literature puts that premium in the
SECURITIES LENDING market (Da-Gurun-Warachka; D'Avolio): high borrow-fee / constrained-supply names earn
abnormally low returns. We cannot buy the fee (Markit starts 2008 and is paywalled), so we build the best
FREE full-history PROXY for lending-market constraint and gate any data purchase on whether it shows anything.

TWO FREE SOURCES, both covering the DataHub window:
  1. SEC Fails-to-Deliver (CNS fails)   2004Q1 → present, bi-monthly files (quarterly zips pre-2013).
     A settlement FAIL is the mechanical footprint of a borrow that could not be sourced → the classic
     hard-to-borrow proxy. Reg SHO's own threshold list is DERIVED from persistent fails, so we reconstruct
     a threshold-style persistence flag ourselves rather than fetching it separately.
  2. FINRA Reg SHO daily short-sale volume, per-exchange TRF files (FNSQ = Nasdaq TRF, FNYX = NYSE TRF),
     2009-08 → present. NOTE we deliberately use the PER-EXCHANGE files, not the consolidated CNMSshvol
     file, because CNMS only exists from ~2019 — FNSQ+FNYX is the only series that is CONSISTENT across the
     whole window, which matters more for a time-series signal than absolute level coverage.

OUTPUT data/htb/htb.parquet — a monthly, month-END-aligned long panel keyed (date, ticker) with:
     ftd_sh       shares failing to deliver, month mean
     ftd_ratio    ftd_sh / shares outstanding   (the scaled borrow-constraint measure)
     ftd_dv       ftd_sh * price / ADV          (fails in days of volume — size-free alternative)
     ftd_days     # of file-dates in the month with a nonzero fail (persistence)
     ftd_thresh   threshold-style flag: fails > 10k shares AND > 0.5% of shares out, persistently
     svol_ratio   ShortVolume / TotalVolume across FNSQ+FNYX, month mean (shorting FLOW intensity)
     svol_exempt  ShortExemptVolume / TotalVolume, month mean (exempt = market-maker/locate-exempt flow)
     svol_abn     svol_ratio minus its own trailing 12m mean (abnormal shorting pressure)

PIT SAFETY: every field is a within-month aggregate stamped at that month-end. FTD files are published with a
~2-week lag, so a month-end stamp is CONSERVATIVE for the second half of the month but OPTIMISTIC for the last
few days. Consumers MUST lag this panel by 1 month before use (as all sleeves already do with hub.elig).

Usage:  python research/build_htb.py            # incremental; skips already-downloaded raw files
        python research/build_htb.py --panel    # re-aggregate the panel from cached raw only (no network)
"""
import os, sys, io, re, time, zipfile, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UA   = "XFUND research enzokreeft@gmail.com"
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "htb")
RAWF = os.path.join(ROOT, "raw_ftd"); RAWS = os.path.join(ROOT, "raw_svol")
for d in (ROOT, RAWF, RAWS): os.makedirs(d, exist_ok=True)


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    if r.headers.get("Content-Encoding") == "gzip":
        import gzip; raw = gzip.decompress(raw)
    return raw


# ── 1. SEC fails-to-deliver ────────────────────────────────────────────────────
def ftd_urls():
    """Scrape the SEC FTD landing page for every .zip (the path prefix varies by vintage)."""
    html = _get("https://www.sec.gov/data-research/sec-markets-data/fails-deliver-data").decode("utf8", "ignore")
    out = []
    for m in re.finditer(r'href="([^"]*(?:cnsfails|cnsp_sec_fails)[^"]*\.zip)"', html):
        u = m.group(1)
        out.append(u if u.startswith("http") else "https://www.sec.gov" + u)
    return sorted(set(out))


def fetch_ftd():
    urls = ftd_urls(); print(f"[ftd] {len(urls)} zip files listed", flush=True)
    got = 0
    for u in urls:
        fn = os.path.join(RAWF, os.path.basename(u))
        if os.path.exists(fn) and os.path.getsize(fn) > 200: continue
        try:
            open(fn, "wb").write(_get(u)); got += 1; time.sleep(0.15)
        except Exception as e:
            print(f"[ftd] MISS {os.path.basename(u)}: {e}", flush=True)
    print(f"[ftd] downloaded {got} new; {len(os.listdir(RAWF))} cached", flush=True)


def parse_ftd():
    """→ long frame [date, ticker, ftd_sh, px_ftd]. Files are pipe-delimited, one row per (settle-date, cusip)."""
    rows = []
    for fn in sorted(os.listdir(RAWF)):
        p = os.path.join(RAWF, fn)
        try:
            with zipfile.ZipFile(p) as z:
                for nm in z.namelist():
                    if not nm.lower().endswith((".txt", ".csv")): continue
                    with z.open(nm) as fh:
                        df = pd.read_csv(fh, sep="|", encoding="latin1", on_bad_lines="skip",
                                         dtype=str, engine="python")
                    df.columns = [c.strip().upper() for c in df.columns]
                    need = {"SETTLEMENT DATE", "SYMBOL", "QUANTITY (FAILS)"}
                    if not need <= set(df.columns): continue
                    d = pd.DataFrame({
                        "date":   pd.to_datetime(df["SETTLEMENT DATE"], format="%Y%m%d", errors="coerce"),
                        "ticker": df["SYMBOL"].str.strip().str.upper(),
                        "ftd_sh": pd.to_numeric(df["QUANTITY (FAILS)"], errors="coerce"),
                        "px_ftd": pd.to_numeric(df.get("PRICE"), errors="coerce")})
                    rows.append(d.dropna(subset=["date", "ticker", "ftd_sh"]))
        except Exception as e:
            print(f"[ftd] parse fail {fn}: {e}", flush=True)
    if not rows: return pd.DataFrame(columns=["date", "ticker", "ftd_sh", "px_ftd"])
    out = pd.concat(rows, ignore_index=True)
    print(f"[ftd] parsed {len(out):,} rows  {out.date.min().date()} .. {out.date.max().date()}", flush=True)
    return out


# ── 2. FINRA Reg SHO daily short volume (per-exchange TRF, consistent across the whole window) ──
def svol_dates(start="2009-08-01", end=None):
    end = end or pd.Timestamp.today().normalize()
    return pd.bdate_range(start, end)


def fetch_svol(workers=8):
    """One cached parquet per YEAR-MONTH; within a month fetch FNSQ+FNYX per business day in parallel."""
    todo = {}
    for d in svol_dates():
        todo.setdefault(d.strftime("%Y%m"), []).append(d)
    for ym, days in sorted(todo.items()):
        out = os.path.join(RAWS, f"{ym}.parquet")
        if os.path.exists(out): continue
        jobs = [(d, mk) for d in days for mk in ("FNSQ", "FNYX")]

        def one(job):
            d, mk = job
            url = f"https://cdn.finra.org/equity/regsho/daily/{mk}shvol{d.strftime('%Y%m%d')}.txt"
            try:
                txt = _get(url, timeout=30).decode("utf8", "ignore")
            except Exception:
                return None                                  # holiday / not published
            if "Symbol" not in txt[:200]: return None
            df = pd.read_csv(io.StringIO(txt), sep="|", on_bad_lines="skip")
            df.columns = [c.strip() for c in df.columns]
            if "Symbol" not in df.columns: return None
            return df[df.Symbol.notna()]

        with ThreadPoolExecutor(workers) as ex:
            parts = [r for r in ex.map(one, jobs) if r is not None and len(r)]
        if not parts:
            pd.DataFrame(columns=["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"]).to_parquet(out)
            continue
        df = pd.concat(parts, ignore_index=True)
        for c in ("ShortVolume", "ShortExemptVolume", "TotalVolume"):
            df[c] = pd.to_numeric(df.get(c), errors="coerce")
        # sum the two TRFs per (day, symbol) so the ratio is a properly pooled ratio, not a mean of ratios
        g = (df.groupby([df.Date.astype(str), df.Symbol.astype(str).str.strip().str.upper()])
               [["ShortVolume", "ShortExemptVolume", "TotalVolume"]].sum().reset_index())
        g.columns = ["Date", "Symbol", "ShortVolume", "ShortExemptVolume", "TotalVolume"]
        g.to_parquet(out)
        print(f"[svol] {ym}: {len(g):,} rows", flush=True)


def parse_svol():
    parts = []
    for fn in sorted(os.listdir(RAWS)):
        try:
            d = pd.read_parquet(os.path.join(RAWS, fn))
            if len(d): parts.append(d)
        except Exception: pass
    if not parts: return pd.DataFrame()
    out = pd.concat(parts, ignore_index=True)
    out["date"] = pd.to_datetime(out.Date.astype(str), format="%Y%m%d", errors="coerce")
    out = out.rename(columns={"Symbol": "ticker"}).dropna(subset=["date", "ticker"])
    print(f"[svol] parsed {len(out):,} rows  {out.date.min().date()} .. {out.date.max().date()}", flush=True)
    return out


# ── 3. aggregate to the DataHub monthly grid & universe ────────────────────────

def _align(p, me, cols):
    """Align a CALENDAR-month-indexed panel onto DataHub's month-end grid.

    BUG THIS KILLS (found 2026-07-25): hub.me is the last TRADING day of each month (2026-05-29), while
    to_timestamp("M") yields the last CALENDAR day (2026-05-31). A plain .reindex(me) therefore silently
    dropped EVERY name in the 95-of-319 months whose calendar end was a weekend/holiday — the panel looked
    ~45-61% covered when the underlying data was near-complete. Match on the month PERIOD instead.
    """
    p = p.copy(); p.index = pd.DatetimeIndex(p.index).to_period("M")
    p = p.reindex(pd.PeriodIndex(me, freq="M"))
    p.index = me
    return p.reindex(columns=cols)


def build_panel():
    from DATAHUB import DataHub
    hub = DataHub(start="2000-01-01", min_days=0)
    me   = hub.me                                            # month-end index
    cols = hub.m_px.columns                                  # the full DataHub universe
    shares = hub.shares("monthly").reindex(index=me, columns=cols)
    adv    = hub.mdv.reindex(index=me, columns=cols)          # month dollar volume
    px     = hub.m_px.reindex(index=me, columns=cols)

    def to_month(df, valcols, how="mean"):
        """long (date,ticker,vals) → month-end × ticker panels, restricted to the DataHub universe."""
        d = df[df.ticker.isin(set(cols))].copy()
        d["m"] = d.date.values.astype("datetime64[M]")
        g = d.groupby(["m", "ticker"])[valcols]
        g = g.mean() if how == "mean" else g.sum()
        out = {}
        for v in valcols:
            p = g[v].unstack("ticker")
            out[v] = _align(p, me, cols)
        return out, d

    res = {}

    # --- FTD ---
    f = parse_ftd()
    if len(f):
        pf, fd = to_month(f, ["ftd_sh"], "mean")
        # ZERO-FILL: verified against the raw files — the SEC FTD feed has NO size floor (min = 1 share,
        # 84% of rows < 10,000 shares, median 766). A security absent on a settlement date therefore had
        # NO fails, i.e. it was UNCONSTRAINED. Absence is information, not missingness. Fill 0 wherever the
        # name was alive (has a price) and the FTD feed was publishing. Raises coverage 45% -> ~100%.
        first_ftd = pf["ftd_sh"].notna().any(axis=1)
        first_ftd = first_ftd[first_ftd].index.min()
        alive = px.notna() & pd.Series(me >= first_ftd, index=me).values[:, None]
        pf["ftd_sh"] = pf["ftd_sh"].fillna(0.0).where(alive)
        res["ftd_sh"] = pf["ftd_sh"]
        res["ftd_ratio"] = (pf["ftd_sh"] / shares.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        res["ftd_dv"] = ((pf["ftd_sh"] * px) / adv.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
        cnt, _ = to_month(f.assign(n=(f.ftd_sh > 0).astype(float)), ["n"], "sum")
        res["ftd_days"] = cnt["n"].fillna(0.0).where(alive)
        # Reg SHO threshold-style: material fails, persistent within the month
        res["ftd_thresh"] = (((pf["ftd_sh"] > 10_000) & (res["ftd_ratio"] > 0.005)).astype(float)
                             .where(pf["ftd_sh"].notna()) * (res["ftd_days"] >= 3).astype(float))

    # --- short volume ---
    s = parse_svol()
    if len(s):
        s = s[s.ticker.isin(set(cols))].copy()
        s["m"] = s.date.values.astype("datetime64[M]")
        g = s.groupby(["m", "ticker"])[["ShortVolume", "ShortExemptVolume", "TotalVolume"]].sum()
        tot = g["TotalVolume"].replace(0, np.nan)
        for nm, num in (("svol_ratio", g["ShortVolume"]), ("svol_exempt", g["ShortExemptVolume"])):
            res[nm] = _align((num / tot).unstack("ticker"), me, cols)
        res["svol_abn"] = res["svol_ratio"] - res["svol_ratio"].rolling(12, min_periods=6).mean()

    if not res:
        print("[htb] nothing to build"); return
    long = pd.concat({k: v.stack(dropna=True) for k, v in res.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    long = long.dropna(how="all")
    out = os.path.join(ROOT, "htb.parquet"); long.to_parquet(out)
    print(f"\n[htb] saved {out}\n      {len(long):,} (date,ticker) rows × {long.shape[1]} fields", flush=True)
    print(f"      window {long.index.get_level_values(0).min().date()} .. "
          f"{long.index.get_level_values(0).max().date()}", flush=True)
    print(f"      tickers {long.index.get_level_values(1).nunique():,}", flush=True)
    print("\nCOVERAGE (non-null share of the liquid universe, by field):")
    el = hub.elig("liquid", "monthly")
    for k, v in res.items():
        cov = (v.notna() & el).sum().sum() / max(el.sum().sum(), 1)
        print(f"  {k:12s} {cov:6.1%}   nonnull rows {int(v.notna().sum().sum()):>10,}")
    return long


# ── 4. GATE TEST — is there anything in the proxy worth paying Markit for? ─────
def gate_test():
    """The spend gate. If a NOISY free proxy for lending-market constraint shows a monotone forward-return
    pattern, buying the real borrow fee (Markit, 2008+, paywalled) is justified. If it shows nothing, close
    the market-neutral-momentum program instead of buying data to confirm a null.

    Design (per RESEARCH_PROTOCOL: define → measure → analyse):
      - universe: hub.elig('liquid'), monthly, the sleeve's own universe
      - every field LAGGED 1 month (FTD files publish with ~2wk lag; 1m lag is conservative)
      - quintile sort on each field → equal-weight forward 1m return, gross
      - report Q1..Q5, the Q5−Q1 spread with a Newey-West-free t, and the monotonicity
      - CRITICAL CONTROL: spread of the field AFTER cross-sectionally residualising out 12-1 momentum,
        so we learn whether it is a SEPARATE premium (T39/T40's requirement) or just momentum re-labelled.
    """
    from DATAHUB import DataHub
    hub = DataHub(start="2000-01-01", min_days=0)
    long = pd.read_parquet(os.path.join(ROOT, "htb.parquet"))
    me, cols = hub.me, hub.m_px.columns
    el  = hub.elig("liquid", "monthly")
    ret = hub.mret.reindex(index=me, columns=cols)                 # realised monthly return
    pm  = hub.m_px.reindex(index=me, columns=cols)
    mom = (pm.shift(1) / pm.shift(12) - 1)                         # 12-1 momentum control

    print(f"\n[GATE] liquid universe · quintiles · fwd-1m equal-weight gross · fields lagged 1m", flush=True)
    print(f"{'field':14}{'Q1(low)':>9}{'Q2':>8}{'Q3':>8}{'Q4':>8}{'Q5(high)':>10}{'Q5-Q1':>9}{'t':>7}{'mono':>7}{'|resid':>9}{'t':>7}", flush=True)
    for f in long.columns:
        P = long[f].unstack("ticker").reindex(index=me, columns=cols).shift(1)
        P = P.where(el)
        rows, rows_r = [], []
        for i, d in enumerate(me):
            x = P.loc[d].dropna(); y = ret.loc[d].reindex(x.index)
            ok = y.notna() & np.isfinite(x)
            x, y = x[ok], y[ok]
            if len(x) < 100: continue
            q = pd.qcut(x.rank(method="first"), 5, labels=False, duplicates="drop")
            if q is None or pd.Series(q).nunique() < 5: continue
            rows.append(y.groupby(q.values).mean())
            m = mom.loc[d].reindex(x.index)
            v = pd.concat([x.rename("x"), m.rename("m")], axis=1).dropna()
            if len(v) > 100:                                        # residualise the field on momentum
                A = np.c_[np.ones(len(v)), pd.Series(v.m).rank(pct=True).values]
                b, *_ = np.linalg.lstsq(A, pd.Series(v.x).rank(pct=True).values, rcond=None)
                e = pd.Series(pd.Series(v.x).rank(pct=True).values - A @ b, index=v.index)
                yy = y.reindex(e.index)
                qq = pd.qcut(e.rank(method="first"), 5, labels=False, duplicates="drop")
                if qq is not None and pd.Series(qq).nunique() == 5:
                    rows_r.append(yy.groupby(qq.values).mean())
        if not rows: continue
        R = pd.DataFrame(rows); sp = R[4] - R[0]
        t = sp.mean() / (sp.std() / np.sqrt(len(sp)) + 1e-12)
        mu = R.mean()
        mono = "yes" if (mu.diff().dropna() > 0).all() or (mu.diff().dropna() < 0).all() else "no"
        if rows_r:
            Rr = pd.DataFrame(rows_r); spr = Rr[4] - Rr[0]
            tr = spr.mean() / (spr.std() / np.sqrt(len(spr)) + 1e-12); srm = spr.mean() * 12
        else:
            srm, tr = np.nan, np.nan
        print(f"{f:14}" + "".join(f"{mu[i]*12:>9.1%}" if i == 0 else
                                  (f"{mu[i]*12:>8.1%}" if i < 4 else f"{mu[i]*12:>10.1%}") for i in range(5))
              + f"{sp.mean()*12:>9.1%}{t:>7.2f}{mono:>7}{srm:>9.1%}{tr:>7.2f}", flush=True)
    print("\n  Q5−Q1 = high-minus-low. For a SHORT premium we want the CONSTRAINED (high FTD / high short-vol)"
          "\n  quintile to UNDERPERFORM, and to survive the momentum residualisation (|resid columns).", flush=True)


if __name__ == "__main__":
    panel_only = "--panel" in sys.argv
    if not panel_only:
        fetch_ftd()
        fetch_svol()
    if "--gate" in sys.argv or panel_only:
        pass
    build_panel()
    if "--gate" in sys.argv:
        gate_test()
