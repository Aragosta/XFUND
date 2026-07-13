#!/usr/bin/env python3
"""reversion_ml.py — the StatArb-tier reversion sleeve = OUR momentum machine, retargeted to a 3-day horizon.

Same architecture as the rank-space momentum champion (Gaussian rank-space features -> XGB regressor on the
Gaussian-rank of the forward return -> decile long/short, dollar-neutral, leak-free embargoed walk-forward).
The ONLY change is the HORIZON: target = 3-DAY forward return (not months). At 3 days price autocorrelation
is NEGATIVE, so the identical machine now harvests REVERSION instead of momentum (opposite premium, ~0 corr).

Daily cross-section, rebalance every 3 trading days (non-overlapping => no label leak), liquid top-1000,
net of cost. Benchmark to beat = Quantitativo StatArb (SR 1.72, DD 16.7%, corr 0.11).

VERDICT: CUT (see [[reversion-catalog-synthesis]]). Through the honest engine this nets ~0.16 (turn 39-74x,
cost-killed); every honest reversion variant lands 0.2-0.3 net in a liquid universe, below the 0.5 bar.
Kept for reference only — NOT a deployed sleeve. Do not re-run as if promising."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
from scipy.stats import norm
from xgboost import XGBRegressor
import BACKTEST

MINPX = 5.0
MINDV = float(os.environ.get("QP_MINDV", 5e6))
TOPN = int(os.environ.get("QP_TOPN", 1000))        # 0 = full eligible universe (no liquidity-rank cap)
STEP, WIN, RETRAIN, SEEDS = 3, 500, 42, 1          # rebalance step (days), rolling train steps, retrain cadence, seeds
REG = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
test = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"])
px = px.loc[:, ~test]                                                    # drop NASDAQ test tickers (known leak)
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
dvd = px * vb; dv = dvd.rolling(63, min_periods=40).mean()
elig = (px > MINPX) & (dv > MINDV)
if TOPN > 0:
    rankdv = dv.where(elig).rank(axis=1, ascending=False); elig = elig & (rankdv <= TOPN)
elig = elig.fillna(False)
dret = px.pct_change(fill_method=None).where(lambda z: z.abs() < 1.0)
# REAL tiered costs (BACKTEST engine): tiers are keyed on MONTHLY $vol -> pass daily 63d-avg $vol x21
mdvq = dv * 21.0
TC = BACKTEST.tiered_transaction_costs(mdvq)                             # per-name one-way trade cost fraction
BF = BACKTEST.tiered_borrow_fees(mdvq)                                   # per-name annual short-borrow fee

def grank(a): r = pd.Series(a).rank(method="average"); return norm.ppf((r - 0.5) / len(r))
def sma(n): return px.rolling(n, min_periods=int(n*0.7)).mean()

print("[feat] ...", flush=True)
feat = {}
for n in (1, 2, 3, 5, 10, 21): feat[f"ret{n}"] = (px / px.shift(n) - 1).where(lambda z: z.abs() < 3)
feat["qp3"]   = (px / px.shift(3) - 1).rolling(1260, min_periods=252).rank(pct=True)
for n in (20, 50, 200): feat[f"d{n}"] = px / sma(n) - 1
feat["rvol"]  = dret.rolling(20, min_periods=15).std()
feat["avol"]  = dvd.rolling(3, min_periods=3).mean() / dvd.rolling(63, min_periods=40).mean().shift(3)
feat["dvtr"]  = np.log(dvd.rolling(21, min_periods=15).mean() / dvd.rolling(252, min_periods=150).mean())
FN = list(feat.keys())
fwd3 = px.shift(-3) / px - 1                                             # 3-day forward return = the target

dates = px.index; T = len(dates)
warm = 1260; start = next(i for i in range(warm, T) if dates[i].year >= 2009)   # train history begins ~2009
rebal = list(range(start, T - STEP, STEP))

print(f"[pool] {len(rebal)} rebalance days ...", flush=True)
pool = {}
for k in rebal:
    dt = dates[k]; el = elig.iloc[k]; idx0 = el.index[el.values]
    P = pd.DataFrame({c: feat[c].iloc[k].reindex(idx0) for c in FN})
    y = fwd3.iloc[k].reindex(idx0)
    ok = P.notna().all(axis=1) & y.notna()
    idx = P.index[ok.values]
    if len(idx) < 100: continue
    X = np.column_stack([grank(P.loc[idx, c].values) for c in FN]).astype(np.float32)
    # train target = rank of RAW fwd return (rank is robust); realized pnl = WINSORIZED (kill data-glitch ticks)
    pool[k] = dict(idx=idx, X=X, Yg=grank(y.loc[idx].values), pnl=y.loc[idx].clip(-0.5, 0.5), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if dates[k].year >= 2011)

print(f"[walk] seeds={SEEDS}, retrain/{RETRAIN} steps, rolling {WIN} ...", flush=True)
store = {}; mdl = None; last_train = -10**9
for i in range(fp, len(keys)):
    k = keys[i]
    if mdl is None or i - last_train >= RETRAIN:
        tr = [keys[j] for j in range(i) if keys[j] <= k - 2*STEP][-WIN:]  # embargo one step; rolling window
        if len(tr) >= 60:
            X = np.vstack([pool[t]["X"] for t in tr]); Y = np.concatenate([pool[t]["Yg"] for t in tr])
            mdl = [XGBRegressor(**REG, random_state=s).fit(X, Y) for s in range(SEEDS)]
            last_train = i
    if mdl is None: continue
    sc = np.mean([m.predict(pool[k]["X"]) for m in mdl], axis=0)
    store[k] = dict(idx=pool[k]["idx"], sc=sc, pnl=pool[k]["pnl"], dt=pool[k]["dt"])
tks = sorted(store)

# daily synthetic prices from cleaned returns (kill data-glitch ticks) -> engine price grid; drifts weights
# between 3-day rebalances. Signal at close k executes NEXT day (lag=1) -> no same-bar look-ahead.
synth = (1 + dret.clip(-0.5, 0.5).fillna(0.0)).cumprod()
def wmat(nsel):
    rows = {}
    for k in tks:
        s = pd.Series(store[k]["sc"], index=store[k]["idx"]); n = max(1, int(len(s)*nsel) if nsel < 1 else nsel)
        w = pd.Series(0.0, index=s.index); w[s.nlargest(n).index] = 0.5/n; w[s.nsmallest(n).index] = -0.5/n
        rows[dates[k]] = w
    return pd.DataFrame(rows).T.reindex(columns=synth.columns)

print("\n" + "=" * 70)
print(f"=== Reversion (stat-arb) via BACKTEST.py — top-{TOPN}, MINDV=${MINDV:.0e}, lag=1, TIERED ===")
print(f"{'book':>10}{'grossSR':>9}{'netSR':>7}{'netAnn':>8}{'netDD':>8}{'turn':>7}")
streams = {}; Wsave = None
for nm, nsel in [("decile", 0.10), ("N=50", 50), ("N=20", 20), ("N=10", 10)]:
    W = wmat(nsel); sig = list(W.index)
    if nm == "N=20": Wsave = W                                              # deployable weights for attribution
    g = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig)
    n = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig, transaction_cost=TC, borrow_fee=BF)
    streams[nm] = n["returns"]
    print(f"{nm:>10}{g['sharpe']:>9.2f}{n['sharpe']:>7.2f}{n['ann_return']:>8.1%}{n['max_drawdown']:>8.1%}{n['ann_turnover']:>7.1f}", flush=True)
pickle.dump(Wsave, open("/tmp/rev_weights.pkl", "wb"))                      # dates x tickers, for attribution

# correlation vs DM / MOM (N=20 net, compounded to monthly)
rm = (1 + streams["N=20"]).resample("ME").prod() - 1; rm.index = pd.DatetimeIndex(rm.index).to_period("M")
print("\n--- corr (N=20 tiered net, monthly) vs momentum sleeves ---")
for nm, path, key in [("MOM champ", "/tmp/mom_champ.pkl", "n1"), ("DM", "/tmp/dm_streams.pkl", None)]:
    if not os.path.exists(path): print(f"  {nm}: (no cache)"); continue
    obj = pickle.load(open(path, "rb")); s = pd.Series(obj[1] if key is None else obj[key]).dropna()
    s.index = pd.DatetimeIndex(s.index).to_period("M")
    df = pd.DataFrame({"REV": rm, nm: s}).dropna()
    if len(df) > 12: print(f"  corr(REV, {nm:9}) = {df.corr().iloc[0,1]:+.2f}  ({len(df)} mo)")
pickle.dump({"daily_net": streams, "N20_monthly": rm}, open("/tmp/rev_ml.pkl", "wb"))
print("[done]", flush=True)
