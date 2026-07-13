#!/usr/bin/env python3
"""embeddings_strategy.py — the asset-embedding sleeve (Quantitativo/Gabaix-Koijen), through BACKTEST.py.

Faithful to the article:
  - Per-quarter co-holding clusters (data/13f/clusters_ts.parquet), used with a 45-DAY LAG (holdings public
    ~45d after quarter-end) -> point-in-time, no look-ahead.
  - TOP-10 clusters selected each period by within-cluster reversion strength (t-stat of vol-adj residual ->
    next-day return, fit on the PRIOR ~63 trading days).
  - Daily within-cluster relative value: residual = ret - cluster-mean-ret, vol-adjusted; quintile L/S
    (long most-negative residual, short most-positive), dollar-neutral within cluster, equal across clusters.
  - lag=1 execution + tiered trade/borrow costs (honest net, not the article's gross).
Reports SR + full stats + correlation to DM/MOM/BAB (the real test: should be ~0)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
import BACKTEST

LAG_DAYS = 45
WIN = int(os.environ.get("EMB_WIN", 1))                                     # 1 = article's 1-day residual; >1 cumulative
LAG = int(os.environ.get("EMB_LAG", 0))                                     # 0 = article execution (capture the bounce)
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$"); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
dvd = px * vb; dv = dvd.rolling(63, min_periods=40).mean()
dret = px.pct_change(fill_method=None).where(lambda z: z.abs() < 1.0)
vol = dret.rolling(63, min_periods=30).std()
synth = (1 + dret.clip(-0.5, 0.5).fillna(0.0)).cumprod()
mdvd = dvd.resample("ME").sum(); me = pd.DatetimeIndex(mdvd.index)
tc = BACKTEST.tiered_transaction_costs(dv * 21.0); bf = BACKTEST.tiered_borrow_fees(dv * 21.0)

CL = pd.read_parquet(os.environ.get("CLUSTERS", "data/13f/clusters_ts.parquet"))   # [period, ticker, cluster]
CL["avail"] = CL["period"] + pd.Timedelta(days=LAG_DAYS)                    # usable from period+45d
periods = sorted(CL["period"].unique())
print(f"[emb] {len(periods)} quarters, {CL.ticker.nunique()} tickers", flush=True)

def clusters_asof(day):
    """most recent quarter's clusters available (period+45d <= day) -> Series ticker->cluster."""
    elig = CL[CL["avail"] <= day]
    if elig.empty: return None
    p = elig["period"].max()
    s = CL[CL["period"] == p].set_index("ticker")["cluster"]
    return s[~s.index.duplicated()]

# build daily weights: iterate trading days from 2020, refresh cluster-set + top-10 each new quarter-avail
days = px.index[px.index >= "2019-06-01"]
W = pd.DataFrame(0.0, index=px.index, columns=px.columns)
cur_period = None; top10 = None; cmap = None
for d in days:
    cl = clusters_asof(d)
    if cl is None: continue
    p = CL[CL["avail"] <= d]["period"].max()
    if p != cur_period:                                                    # new quarter available -> reselect
        cur_period = p; cmap = cl
        # top-10 clusters by within-cluster reversion t-stat over prior 63 trading days
        hist = px.index[(px.index < d)][-64:]
        if len(hist) < 40: continue
        scores = {}
        R = dret.loc[hist]                                                 # 63 x N
        cmean = {}
        codes = cmap.reindex(px.columns)
        for c in cmap.unique():
            names = codes.index[codes.values == c]
            names = [n for n in names if n in R.columns]
            if len(names) < 10: continue
            sub = R[names]
            resid = sub.sub(sub.mean(axis=1), axis=0)                      # ret - cluster mean
            vadj = resid / (vol.loc[hist, names] + 1e-9)
            x = vadj.iloc[:-1].values.flatten(); y = sub.shift(-1).iloc[:-1].values.flatten()
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() < 200: continue
            xb, yb = x[m], y[m]; b = np.polyfit(xb, yb, 1)[0]
            se = np.std(yb - b*xb) / (np.std(xb)*np.sqrt(len(xb)) + 1e-12)
            scores[c] = b / (se + 1e-12)                                    # t-stat (negative = reversion)
        top10 = [c for c,_ in sorted(scores.items(), key=lambda kv: kv[1])[:10]]   # most-negative = strongest reversion
    if not top10: continue
    codes = cmap.reindex(px.columns)
    win = px.index[px.index <= d][-WIN:]                                    # trailing window for cumulative residual
    wrow = pd.Series(0.0, index=px.columns)
    for c in top10:
        names = [n for n in codes.index[codes.values == c] if n in dret.columns]
        if len(names) < 10: continue
        sub = dret.loc[win, names]
        resid = sub.sub(sub.mean(axis=1), axis=0)                          # daily residual vs cluster
        cum = resid.sum(axis=0)                                            # WIN=1 -> article's 1-day residual
        z = (cum / (vol.loc[d, names].reindex(cum.index) * np.sqrt(len(win)) + 1e-9)).dropna()  # vol-adjusted
        if len(z) < 10: continue
        n = max(1, int(len(z) * 0.20))
        lo = z.nsmallest(n).index; hi = z.nlargest(n).index                # long cheap (drifted down), short rich
        wrow[lo] += 0.5/n; wrow[hi] -= 0.5/n
    g = wrow.abs().sum()
    if g > 0: W.loc[d] = (wrow * (2.0/g)).values                           # gross 2, dollar-neutral
W = W.loc[(W.abs().sum(axis=1) > 0)]

print(f"[book] {len(W)} trading days with positions", flush=True)
sig = list(W.index)
gross = BACKTEST.backtest(W, synth, freq=252, lag=LAG, signal_dates=sig)
net = BACKTEST.backtest(W, synth, freq=252, lag=LAG, signal_dates=sig, transaction_cost=tc, borrow_fee=bf)
print("=" * 78)
print(f"=== ASSET-EMBEDDING sleeve (WIN={WIN} residual, exec lag={LAG}, tiered) ===")
print(f"  gross SR {gross['sharpe']:.2f} | NET SR {net['sharpe']:.2f}  ann {net['ann_return']:.1%}  "
      f"vol {net['ann_vol']:.1%}  maxDD {net['max_drawdown']:.1%}  turn {net['ann_turnover']:.1f}", flush=True)
# GROSS (costless) Sharpe over the article's window vs full, to compare to their reported 2.59
def _sr(r): r = r[r.abs() > 0]; return (r.mean() / (r.std() + 1e-12)) * np.sqrt(252) if len(r) else float("nan")
gr = gross["returns"]
for lab, a, b in [("full", "2019-06-01", "2026-12-31"), ("article 2020-2025", "2020-01-01", "2025-12-31"),
                  ("covid 2020 only", "2020-01-01", "2020-12-31")]:
    w = gr[(gr.index >= a) & (gr.index <= b)]
    print(f"    GROSS SR [{lab:18}] = {_sr(w):.2f}  ann {(1+w).prod()**(252/max(len(w),1))-1:.1%}", flush=True)

nr = net["returns"]; nr = nr[nr.index >= W.index.min()]                     # restrict to the strategy's active period
rm = (1 + nr).resample("ME").prod() - 1
rm.index = pd.DatetimeIndex(rm.index).to_period("M")
print("  --- correlation to existing sleeves (the real test) ---")
for nm, path, key in [("MOM","/tmp/mom_champ.pkl","n1"),("DM","/tmp/dm_streams.pkl",None),("BAB","/tmp/bab_stream.pkl",None)]:
    if not os.path.exists(path): print(f"    {nm}: (no cache)"); continue
    o = pickle.load(open(path,"rb")); s = pd.Series(o[key] if isinstance(o,dict) else (o[1] if isinstance(o,tuple) else o)).dropna()
    if not isinstance(s.index, pd.PeriodIndex): s.index = pd.DatetimeIndex(s.index).to_period("M")
    if nm=="MOM": s = s.shift(1)
    df = pd.DataFrame({"EMB":rm, nm:s}).dropna()
    if len(df)>12: print(f"    corr(EMB,{nm}) = {df.corr().iloc[0,1]:+.2f}  ({len(df)} mo)")
pickle.dump(net["returns"], open("/tmp/emb_stream.pkl","wb")); pickle.dump(W, open("/tmp/emb_weights.pkl","wb"))
print("[done]", flush=True)
