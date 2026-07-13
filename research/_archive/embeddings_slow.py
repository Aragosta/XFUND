#!/usr/bin/env python3
"""embeddings_slow.py — ISOLATE the 13F co-holding premium at LOW turnover, with a control arm.

The first build (embeddings_strategy.py) failed hard: daily within-cluster relative-value = daily reversion,
108x turnover, NET SR -4.13, -100% maxDD. It also changed TWO things at once (new 13F premium AND a daily
reversion construction) so the relational THESIS was never actually tested. This script fixes both:

  ONE CHANGE:  real 13F co-holding clusters  vs  RANDOM (shuffled) clusters of identical sizes.
  Everything else held fixed: monthly rebalance (turnover ~10x not 108x), decile L/S, dollar-neutral,
  lag=1 honest execution, tiered trade+borrow costs, same signal, same universe, same dates.

  SIGNAL (per name, monthly): residual = trailing-21d return  minus  its cluster's mean trailing-21d return,
  vol-adjusted. Long the decile that has LAGGED its co-held peers (low residual), short the decile that has
  RUN AHEAD (high residual), dollar-neutral. With REAL clusters this bets on the co-holding/comovement axis;
  with RANDOM clusters the cluster-mean collapses to ~market mean, so the baseline is plain monthly reversal.
  The number that matters is ARM - BASELINE: the marginal alpha of the 13F structure, not the arm's level.

Clusters are point-in-time: usable only period+45d (13F public lag). Window is 13F-limited (~2013+), NOT the
2011+ substrate — labelled, and NOT directly comparable to DM/MOM/BAB levels (report corr, judge arm-baseline).

Decision bar (pre-registered): net-SR(real) - net-SR(random) >= +0.10 AND real IC > 0. Else the co-holding
axis carries no slow tradeable premium -> the 13F sleeve is dead in a liquid universe."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
import BACKTEST

MINPX, MINDV = 5.0, 5e6
LAG_DAYS = 45                                                              # 13F public ~45d after quarter-end
MIN_CLUST = 10                                                            # need >=10 names for a meaningful peer mean
N_RAND = int(os.environ.get("EMB_NRAND", 5))                             # random-cluster baseline permutations
WIN = 21                                                                  # trailing-return / signal window (days)

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
test = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"])
px = px.loc[:, ~test]                                                     # drop NASDAQ test tickers (known leak)
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
dvd = px * vb; dv = dvd.rolling(63, min_periods=40).mean()
elig = (px > MINPX) & (dv > MINDV)
dret = px.pct_change(fill_method=None).where(lambda z: z.abs() < 1.0)
vol = dret.rolling(WIN, min_periods=15).std()
r21 = px / px.shift(WIN) - 1                                              # trailing-21d return (the raw displacement)
synth = (1 + dret.clip(-0.5, 0.5).fillna(0.0)).cumprod()                  # cleaned price grid for the engine
mdvq = dv * 21.0
TC = BACKTEST.tiered_transaction_costs(mdvq); BF = BACKTEST.tiered_borrow_fees(mdvq)

CL = pd.read_parquet(os.environ.get("CLUSTERS", "data/13f/clusters_ts.parquet"))  # [period, ticker, cluster], PIT
CL["avail"] = CL["period"] + pd.Timedelta(days=LAG_DAYS)
def clusters_asof(day):
    elig_p = CL[CL["avail"] <= day]
    if elig_p.empty: return None
    p = elig_p["period"].max()
    s = CL[CL["period"] == p].set_index("ticker")["cluster"]
    return s[~s.index.duplicated()]

# monthly rebalance = last trading day of each month; first usable ~ first cluster avail + WIN
me = px.index.to_series().groupby(px.index.to_period("M")).last()
rebal = [d for d in me if d >= (CL["avail"].min() + pd.Timedelta(days=WIN*2))]
print(f"[emb-slow] {len(rebal)} monthly rebalances {rebal[0].date()}..{rebal[-1].date()}, {CL.ticker.nunique()} tickers", flush=True)

def signal_row(d, cmap):
    """decile L/S weights on residual(trailing-21d return vs cluster-mean), vol-adj, dollar-neutral gross 2.
    cmap: Series ticker->cluster (already restricted to eligible names present at d)."""
    codes = cmap
    rr = r21.loc[d].reindex(codes.index); vv = vol.loc[d].reindex(codes.index)
    df = pd.DataFrame({"r": rr, "v": vv, "c": codes.values}).dropna()
    if df.empty: return None
    cmean = df.groupby("c")["r"].transform("mean")
    csz = df.groupby("c")["r"].transform("size")
    df = df[csz >= MIN_CLUST]
    if len(df) < 50: return None
    z = ((df["r"] - cmean.loc[df.index]) / (df["v"] + 1e-9))              # vol-adj residual vs co-held peers
    r = z.rank(pct=True)
    lo = z.index[r <= 0.10]; hi = z.index[r >= 0.90]                      # long laggards vs peers / short leaders
    if len(lo) == 0 or len(hi) == 0: return None
    w = pd.Series(0.0, index=px.columns)
    w[lo] = 0.5/len(lo); w[hi] = -0.5/len(hi)
    return w, z, dret.loc[d:].iloc[1:WIN+1].reindex(columns=[*z.index]).sum().reindex(z.index)  # fwd-WIN ret for IC

def build(shuffle_seed=None):
    """build monthly weight matrix. shuffle_seed=None -> real clusters; int -> permuted labels (same sizes)."""
    rows = {}; ics = []
    rng = np.random.default_rng(shuffle_seed) if shuffle_seed is not None else None
    for d in rebal:
        cl = clusters_asof(d)
        if cl is None: continue
        el = elig.loc[d]; cl = cl[cl.index.isin(el.index[el.values])]      # eligible-only membership
        cl = cl[cl.index.isin(px.columns)]
        if len(cl) < 50: continue
        if rng is not None:                                               # shuffle labels across the SAME names
            cl = pd.Series(rng.permutation(cl.values), index=cl.index)
        out = signal_row(d, cl)
        if out is None: continue
        w, z, fwd = out; rows[d] = w
        ic = pd.Series(-z).corr(fwd, method="spearman")                   # -z (long-low) vs fwd return
        if np.isfinite(ic): ics.append(ic)
    W = pd.DataFrame(rows).T.reindex(columns=synth.columns)
    return W, (np.mean(ics) if ics else np.nan)

def run(W, ic):
    sig = list(W.index)
    g = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig)
    n = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig, transaction_cost=TC, borrow_fee=BF)
    return g, n, ic

print("[arm] real 13F clusters ...", flush=True)
Wr, icr = build(shuffle_seed=None); gr, nr, icr = run(Wr, icr)

print(f"[baseline] {N_RAND} random-cluster permutations ...", flush=True)
brs = []
for s in range(N_RAND):
    Wb, icb = build(shuffle_seed=s); gb, nb, icb = run(Wb, icb); brs.append((gb, nb, icb))
gb_sr = np.mean([b[0]["sharpe"] for b in brs]); nb_sr = np.mean([b[1]["sharpe"] for b in brs])
nb_ann = np.mean([b[1]["ann_return"] for b in brs]); nb_ic = np.nanmean([b[2] for b in brs])

print("\n" + "=" * 82)
print("=== 13F co-holding sleeve — SLOW isolation test (monthly, lag=1, tiered) ===")
print(f"{'arm':>22}{'gSR':>7}{'nSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}{'IC':>7}")
print(f"{'real clusters':>22}{gr['sharpe']:>7.2f}{nr['sharpe']:>7.2f}{nr['ann_return']:>8.1%}"
      f"{nr['max_drawdown']:>8.1%}{nr['ann_turnover']:>7.1f}{icr:>7.3f}")
print(f"{'random clusters (base)':>22}{gb_sr:>7.2f}{nb_sr:>7.2f}{nb_ann:>8.1%}{'—':>8}{'—':>7}{nb_ic:>7.3f}")
print(f"{'ARM - BASELINE':>22}{gr['sharpe']-gb_sr:>7.2f}{nr['sharpe']-nb_sr:>7.2f}"
      f"{'':>8}{'':>8}{'':>7}{icr-nb_ic:>7.3f}")
print(f"\nPRE-REG BAR: net-SR(real)-net-SR(random) >= +0.10 AND real IC>0 -> {'PASS' if (nr['sharpe']-nb_sr>=0.10 and icr>0) else 'FAIL (13F axis carries no slow tradeable premium)'}")

# correlation of the REAL arm to existing sleeves (only meaningful if the arm is net-positive)
nrr = nr["returns"]; nrr = nrr[nrr.index >= Wr.index.min()]               # restrict to the arm's active period
rm = (1 + nrr).resample("ME").prod() - 1
rm.index = pd.DatetimeIndex(rm.index).to_period("M")
print("\n--- corr(real-cluster arm, sleeve) monthly ---")
for nm, path, key in [("MOM","/tmp/mom_champ.pkl","n1"),("DM","/tmp/dm_streams.pkl",None),("BAB","/tmp/bab_stream.pkl",None)]:
    if not os.path.exists(path): print(f"  {nm}: (no cache)"); continue
    o = pickle.load(open(path,"rb")); s = pd.Series(o[key] if isinstance(o,dict) else (o[1] if isinstance(o,tuple) else o)).dropna()
    if not isinstance(s.index, pd.PeriodIndex): s.index = pd.DatetimeIndex(s.index).to_period("M")
    if nm=="MOM": s = s.shift(1)
    df = pd.DataFrame({"EMB":rm, nm:s}).dropna()
    if len(df) > 12: print(f"  corr(EMB,{nm}) = {df.corr().iloc[0,1]:+.2f}  ({len(df)} mo)")
pickle.dump(nr["returns"], open("/tmp/emb_slow_stream.pkl","wb"))
print("[done]", flush=True)
