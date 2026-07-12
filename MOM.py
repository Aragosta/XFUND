#!/usr/bin/env python3
"""MOM.py — the MOM sleeve (CHAMPION): MULTI-HORIZON Gaussian rank-space multi-output regression.

XGBoost vector-leaf `multi_strategy="multi_output_tree"` (XGBRegressor) on the Gaussian-rank of forward returns
at HORIZONS [t+4, t+5, t+6]. FEATURES = ret/nret{3,6,12}, Baz MACD (y1/y2/y3 + composite), FFD, amihud, dvtrend,
abnvol, residual-momentum. Full broad universe, seeds=5, decile L/S dollar-neutral, net via BACKTEST.py.
Beta-neutralize the output book with BETANEUT.py before combining. Saves the decile book -> /tmp/mom_weights.pkl."""
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor
import deep_momentum_xgb as d
import BACKTEST

SEEDS, MINPX, MINDV, DELIST, BW = 5, 5.0, 5e6, -0.30, 60
REG = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
HZ = (4, 5, 6)

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
V = m_px.notna().values; last = len(me) - 1
for j in range(V.shape[1]):
    w = np.where(V[:, j])[0]
    if len(w) and w[-1] < last - 1 and w[-1] + 1 < len(me): mret.iat[w[-1] + 1, j] = DELIST
dret = px.pct_change(fill_method=None); vol_d = dret.ewm(span=63, min_periods=20).std(); T = len(me)
dvd = px * vb; dvm = dvd.resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mdv = dvm.reindex(pd.PeriodIndex(me, freq="M")); mdv.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > MINPX) & (cov > 0.9) & (mdv > MINDV)
tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zrow(df): return df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0)
def grank(a): r = pd.Series(a).rank(method="average"); return norm.ppf((r - 0.5) / len(r))

mkt = mret.where(elig).mean(axis=1)
mean_r = mret.rolling(BW, min_periods=36).mean(); mean_m = mkt.rolling(BW, min_periods=36).mean()
mean_rm = mret.mul(mkt, axis=0).rolling(BW, min_periods=36).mean(); var_m = (mkt**2).rolling(BW, min_periods=36).mean() - mean_m**2
beta = mean_rm.sub(mean_r.mul(mean_m, axis=0)).div(var_m, axis=0); res = mret.sub(beta.shift(1).mul(mkt, axis=0))

print("[feat] ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1).where(lambda z: z.abs() < 5); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (Sh, Lg) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(Sh)).mean() - px.ewm(halflife=hl(Lg)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp)
ffd = d._ffd_from_training_window(m_px, int(np.argmax(me.year >= 2011)))
for m, v in ffd.items():
    z_ = v.reindex(me); pf[f"ffd{m}"] = z_.sub(z_.mean(1), 0).div(z_.std(1) + 1e-9, 0)
COREREQ = list(pf.keys())
dv63 = dvd.rolling(63, min_periods=40).mean(); dv252 = dvd.rolling(252, min_periods=150).mean()
pf["amihud"]  = zrow(at_me((dret.abs() / dvd.replace(0, np.nan)).rolling(252, min_periods=120).mean()))
pf["dvtrend"] = zrow(at_me(np.log(dv63 / dv252)))
pf["abnvol"]  = zrow(at_me(dvd.rolling(21, min_periods=15).mean() / dv252))
pf["resmom"]  = zrow((res.shift(1).rolling(11, min_periods=8).sum() / res.rolling(11, min_periods=8).std().shift(1)).reindex(me))
PF = list(pf.keys())

print("[pool] ...", flush=True)
HMAX = 6
pool = {}
for k in range(BW + 13, T - HMAX):
    dt = me[k]; el = elig.loc[dt].fillna(False); idx0 = el.index[el.values]
    P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in pf})
    ok = P[COREREQ].notna().all(axis=1)
    for h in range(1, HMAX + 1): ok &= mret.iloc[k+h].reindex(idx0).notna()
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Prank = np.column_stack([grank(P.loc[idx].fillna(0.0)[c].values) for c in PF])
    pool[k] = dict(Prank=Prank.astype(np.float32), idx=idx, Yg={h: grank(mret.iloc[k + h].reindex(idx).values) for h in HZ},
                   pnl=mret.iloc[k+1].reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
print(f"[pool] {len(keys)} months, avg {np.mean([len(pool[k]['idx']) for k in keys if me[k].year>=2011]):.0f}", flush=True)

print("[walk] seeds=5 ...", flush=True)
emb = max(HZ); store = {}; mdl = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month in (1, 4, 7, 10) or mdl is None:
        tr = [keys[j] for j in range(i) if keys[j] <= k - emb]
        if len(tr) >= 36:
            X = np.vstack([pool[t]["Prank"] for t in tr]); Y = np.column_stack([np.concatenate([pool[t]["Yg"][h] for t in tr]) for h in HZ])
            mdl = [XGBRegressor(**REG, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(SEEDS)]
    if mdl is None: continue
    p = np.mean([m.predict(pool[k]["Prank"]).mean(1) for m in mdl], axis=0)
    s = pd.Series(p, index=pool[k]["idx"]); n = max(1, int(len(s) * 0.10)); w = pd.Series(0.0, index=s.index)
    w[s.nlargest(n).index] = 1.0 / n; w[s.nsmallest(n).index] = -1.0 / n
    store[k] = dict(w=w, pnl=pool[k]["pnl"], dt=dt)
tks = sorted(store)

# monthly synthetic prices reproducing the delist-adjusted returns the model trades -> engine price grid
synth = (1 + mret.fillna(0.0)).cumprod()
def wmat(H, cg):
    """monthly weight matrix: H-month overlapping average of decile books, optional const-gross renorm."""
    rows = {}
    for i in range(len(tks)):
        k = tks[i]; W = pd.concat([store[tks[j]]["w"] for j in range(max(0, i - H + 1), i + 1)], axis=1).mean(axis=1)
        if cg:
            g = W.abs().sum()
            if g > 0: W = W * (2.0 / g)
        rows[store[k]["dt"]] = W
    return pd.DataFrame(rows).T.reindex(columns=synth.columns)

# ONE engine: signal at month t earns t->t+1 (lag=0), freq=12, tiered trade + borrow costs
out = {}; Wsave = None
for tag, H, cg in [("1", 1, False), ("3", 3, True)]:
    W = wmat(H, cg); sig = list(W.index)
    if tag == "1": Wsave = W                                                # champion weights (H=1) for attribution
    g = BACKTEST.backtest(W, synth, freq=12, lag=0, signal_dates=sig)
    n = BACKTEST.backtest(W, synth, freq=12, lag=0, signal_dates=sig, transaction_cost=tc, borrow_fee=bf)
    out[f"g{tag}"], out[f"n{tag}"] = g["returns"], n["returns"]
    lbl = "H=1       " if tag == "1" else "H=3 constG"
    print(f"[MOM] {lbl} net SR {n['sharpe']:.2f}  ann {n['ann_return']:.1%}  maxDD {n['max_drawdown']:.1%}"
          f"  turn {n['ann_turnover']:.1f}", flush=True)
pickle.dump(out, open("/tmp/mom_champ.pkl", "wb"))
pickle.dump(Wsave, open("/tmp/mom_weights.pkl", "wb"))                      # dates x tickers, for attribution
print("[done] saved /tmp/mom_champ.pkl", flush=True)
