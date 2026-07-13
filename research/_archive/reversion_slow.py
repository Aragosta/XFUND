#!/usr/bin/env python3
"""reversion_slow.py — rebuild the stat-arb arm in SLOW signal spaces (illiquidity + volume), not returns.

The returns-reversion arm is dead (net 0.16, turn 74x, day-1 look-ahead + cost). The signal-space test showed
the only net-positive space is slow (Amihud illiquidity +0.32 at 5.2x turnover). So: same ML architecture as
the momentum champion (Gaussian rank-space features -> XGB regressor on rank of forward return -> decile L/S,
dollar-neutral, leak-free walk-forward), but with SLOW features ONLY (Amihud, $-vol trend/autocorr/abnormality,
liquidity level, realized vol) and NO recent returns. Longer horizon (5d) + 5-day rebalance keep turnover low.
Evaluated via BACKTEST.py (lag=1, tiered). Decision bar: clear ~0.5 net or the arm isn't worth a sleeve."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
from scipy.stats import norm
from xgboost import XGBRegressor
import BACKTEST

MINPX = 5.0
MINDV = float(os.environ.get("QP_MINDV", 5e6))
TOPN = int(os.environ.get("QP_TOPN", 1000))
STEP, WIN, RETRAIN, SEEDS = 5, 400, 40, 1                                   # 5-day horizon/rebalance (slower)
REG = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
test = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"])
px = px.loc[:, ~test]                                                       # drop NASDAQ test tickers (data hygiene)
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
dvd = px * vb; dv = dvd.rolling(63, min_periods=40).mean()
elig = (px > MINPX) & (dv > MINDV)
if TOPN > 0:
    rankdv = dv.where(elig).rank(axis=1, ascending=False); elig = elig & (rankdv <= TOPN)
elig = elig.fillna(False)
dret = px.pct_change(fill_method=None).where(lambda z: z.abs() < 1.0)
mdvq = dv * 21.0
TC = BACKTEST.tiered_transaction_costs(mdvq); BF = BACKTEST.tiered_borrow_fees(mdvq)

def grank(a): r = pd.Series(a).rank(method="average"); return norm.ppf((r - 0.5) / len(r))

print("[feat] SLOW spaces only (no recent returns) ...", flush=True)
dv21 = dvd.rolling(21, min_periods=15).mean(); dv252 = dvd.rolling(252, min_periods=150).mean()
feat = {
    "amihud":  (dret.abs() / dvd.replace(0, np.nan)).rolling(63, min_periods=40).mean(),
    "amihud_tr": np.log((dret.abs()/dvd.replace(0,np.nan)).rolling(21,min_periods=15).mean() /
                        (dret.abs()/dvd.replace(0,np.nan)).rolling(252,min_periods=150).mean()),
    "dvtrend": np.log(dv21 / dv252),
    "abnvol":  dvd.rolling(3, min_periods=3).mean() / dv252.shift(3),
    "volac":   dvd.rolling(20, min_periods=15).corr(dvd.shift(1)),          # $-volume autocorrelation
    "size":    np.log(dv),                                                  # liquidity level
    "rvol":    dret.rolling(20, min_periods=15).std(),
    "illiq_x_vol": ((dret.abs()/dvd.replace(0,np.nan)).rolling(63,min_periods=40).mean()) * dret.rolling(20,min_periods=15).std(),
}
FN = list(feat.keys())
fwdH = px.shift(-STEP) / px - 1                                             # STEP-day forward return = target

dates = px.index; T = len(dates)
start = next(i for i in range(300, T) if dates[i].year >= 2009)
rebal = list(range(start, T - STEP, STEP))
print(f"[pool] {len(rebal)} rebalance days (STEP={STEP}) ...", flush=True)
pool = {}
for k in rebal:
    el = elig.iloc[k]; idx0 = el.index[el.values]
    P = pd.DataFrame({c: feat[c].iloc[k].reindex(idx0) for c in FN})
    y = fwdH.iloc[k].reindex(idx0)
    ok = P.notna().all(axis=1) & y.notna()
    idx = P.index[ok.values]
    if len(idx) < 100: continue
    X = np.column_stack([grank(P.loc[idx, c].values) for c in FN]).astype(np.float32)
    pool[k] = dict(idx=idx, X=X, Yg=grank(y.loc[idx].values), dt=dates[k])
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if dates[k].year >= 2011)

print(f"[walk] seeds={SEEDS} ...", flush=True)
store = {}; mdl = None; last = -10**9
for i in range(fp, len(keys)):
    k = keys[i]
    if mdl is None or i - last >= RETRAIN:
        tr = [keys[j] for j in range(i) if keys[j] <= k - 2*STEP][-WIN:]
        if len(tr) >= 60:
            X = np.vstack([pool[t]["X"] for t in tr]); Y = np.concatenate([pool[t]["Yg"] for t in tr])
            mdl = [XGBRegressor(**REG, random_state=s).fit(X, Y) for s in range(SEEDS)]; last = i
    if mdl is None: continue
    sc = np.mean([m.predict(pool[k]["X"]) for m in mdl], axis=0)
    store[k] = dict(idx=pool[k]["idx"], sc=sc, dt=pool[k]["dt"])
tks = sorted(store)

synth = (1 + dret.clip(-0.5, 0.5).fillna(0.0)).cumprod()
def wmat(nsel):
    rows = {}
    for k in tks:
        s = pd.Series(store[k]["sc"], index=store[k]["idx"]); n = max(1, int(len(s)*nsel) if nsel < 1 else nsel)
        w = pd.Series(0.0, index=s.index); w[s.nlargest(n).index] = 0.5/n; w[s.nsmallest(n).index] = -0.5/n
        rows[dates[k]] = w
    return pd.DataFrame(rows).T.reindex(columns=synth.columns)

print("\n" + "=" * 70)
print(f"=== SLOW-space stat-arb via BACKTEST.py (top-{TOPN}, STEP={STEP}d, lag=1, tiered) ===")
print(f"{'book':>10}{'grossSR':>9}{'netSR':>8}{'netAnn':>8}{'netDD':>8}{'turn':>7}")
streams = {}
for nm, nsel in [("decile", 0.10), ("N=50", 50), ("N=20", 20), ("N=10", 10)]:
    W = wmat(nsel); sig = list(W.index)
    g = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig)
    n = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig, transaction_cost=TC, borrow_fee=BF)
    streams[nm] = n["returns"]
    print(f"{nm:>10}{g['sharpe']:>9.2f}{n['sharpe']:>8.2f}{n['ann_return']:>8.1%}{n['max_drawdown']:>8.1%}{n['ann_turnover']:>7.1f}", flush=True)

rm = (1 + streams["N=20"]).resample("ME").prod() - 1; rm.index = pd.DatetimeIndex(rm.index).to_period("M")
print("\n--- corr (N=20 net, monthly) vs momentum sleeves ---")
for nm, path, key in [("MOM", "/tmp/mom_champ.pkl", "n1"), ("DM", "/tmp/dm_streams.pkl", None)]:
    if not os.path.exists(path): print(f"  {nm}: (no cache)"); continue
    obj = pickle.load(open(path, "rb")); s = pd.Series(obj[1] if key is None else obj[key]).dropna()
    s.index = pd.DatetimeIndex(s.index).to_period("M")
    if key == "n1": s = s.shift(1)                                          # MOM realization-align
    df = pd.DataFrame({"SLOW": rm, nm: s}).dropna()
    if len(df) > 12: print(f"  corr(SLOW, {nm:4}) = {df.corr().iloc[0,1]:+.2f}  ({len(df)} mo)")
pickle.dump(streams, open("/tmp/rev_slow.pkl", "wb"))
print("[done]", flush=True)
