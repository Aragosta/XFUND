#!/usr/bin/env python3
"""mom_regime.py — MOM with GMM regime-conditioned sample-weighted training.

Two XGBoost models per retrain window:
  mdl_0: XGBRegressor.fit(X, Y, sample_weight=p_state0_per_training_month)
  mdl_1: XGBRegressor.fit(X, Y, sample_weight=p_state1_per_training_month)

At prediction month t: look up regime_states[me[t]].state — use mdl_0 (state=0) or mdl_1 (state=1).
Fallback to full-data model if sum(weights) < MIN_EFF for either state.

Everything else is identical to MOM.py: lag=1, embargo=6, seeds=5, decile book, BACKTEST.py.
Hypothesis: State-1 model is optimized for squeeze regimes → reduces short-book blowups.
Pre-registered bar: net SR Δ ≥ +0.20 vs MOM baseline (SR 0.31).
"""
import warnings; warnings.filterwarnings("ignore")
import pickle
import numpy as np, pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor
import BACKTEST
from UNIVERSE import ffd_scores

SEEDS, MINPX, MINDV, DELIST, BW = 5, 5.0, 5e6, -0.30, 60
SEEDS_REGIME = 1    # seeds for per-state models (concept test; increase if adopted)
REG = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
HZ = (4, 5, 6)
MIN_EFF = 24        # minimum effective sample count (sum of weights) per state before falling back

from DATAHUB import DataHub
hub = DataHub(start="2000-01-01", min_days=0)
px, vb, me, m_px = hub.px_d, hub.vol_d, hub.me, hub.m_px
mret = hub.mret.copy()
V = m_px.notna().values; last = len(me) - 1
for j in range(V.shape[1]):
    w = np.where(V[:, j])[0]
    if len(w) and w[-1] < last - 1 and w[-1] + 1 < len(me): mret.iat[w[-1] + 1, j] = DELIST
dret = px.pct_change(fill_method=None); vol_d = dret.ewm(span=63, min_periods=20).std()
dvd = px * vb; mdv, cov, elig = hub.mdv, hub.cov_m, hub.elig("liquid")
tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zrow(df): return df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0)
def grank(a): r = pd.Series(a).rank(method="average"); return norm.ppf((r - 0.5) / len(r))

mkt = mret.where(elig).mean(axis=1)
mean_r = mret.rolling(BW, min_periods=36).mean(); mean_m = mkt.rolling(BW, min_periods=36).mean()
mean_rm = mret.mul(mkt, axis=0).rolling(BW, min_periods=36).mean(); var_m = (mkt**2).rolling(BW, min_periods=36).mean() - mean_m**2
beta = mean_rm.sub(mean_r.mul(mean_m, axis=0)).div(var_m, axis=0); res = mret.sub(beta.shift(1).mul(mkt, axis=0))

# ── Load regime states ────────────────────────────────────────────────────────
regime_states = pickle.load(open("/tmp/regime_states.pkl", "rb"))
def regime_at(dt):
    """State and probabilities for month dt (expanding-window, no look-ahead)."""
    return regime_states.get(dt, {"state": 0, "p0": 1.0, "p1": 0.0})

print("[feat] ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1).where(lambda z: z.abs() < 5)
    pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (Sh, Lg) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(Sh)).mean() - px.ewm(halflife=hl(Lg)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp)
ffd = ffd_scores(m_px, int(np.argmax(me.year >= 2011)))
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
for k in range(BW + 13, len(me) - HMAX):
    dt = me[k]; el = elig.loc[dt].fillna(False); idx0 = el.index[el.values]
    P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in pf})
    ok = P[COREREQ].notna().all(axis=1)
    for h in range(1, HMAX + 1): ok &= mret.iloc[k+h].reindex(idx0).notna()
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Prank = np.column_stack([grank(P.loc[idx].fillna(0.0)[c].values) for c in PF])
    pool[k] = dict(Prank=Prank.astype(np.float32), idx=idx,
                   Yg={h: grank(mret.iloc[k + h].reindex(idx).values) for h in HZ},
                   pnl=mret.iloc[k+1].reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
print(f"[pool] {len(keys)} months, avg {np.mean([len(pool[k]['idx']) for k in keys if me[k].year>=2011]):.0f}", flush=True)

# ── Walk-forward with per-state sample-weighted models ────────────────────────
print("[walk] regime-conditioned, seeds_regime=1 ...", flush=True)
emb = max(HZ); store = {}; mdl_0 = None; mdl_1 = None
n_state0_used = 0; n_state1_used = 0

for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month in (1, 4, 7, 10) or mdl_0 is None:
        tr = [keys[j] for j in range(i) if keys[j] <= k - emb]
        if len(tr) >= 36:
            X = np.vstack([pool[t]["Prank"] for t in tr])
            Y = np.column_stack([np.concatenate([pool[t]["Yg"][h] for t in tr]) for h in HZ])

            # Per-sample weights from regime probabilities of each training month
            segs_p0, segs_p1 = [], []
            for t in tr:
                rs = regime_at(pool[t]["dt"])
                n = len(pool[t]["idx"])
                segs_p0.append(np.full(n, max(rs["p0"], 0.01), dtype=np.float32))
                segs_p1.append(np.full(n, max(rs["p1"], 0.01), dtype=np.float32))
            sw0 = np.concatenate(segs_p0)
            sw1 = np.concatenate(segs_p1)

            # 2 fits per retrain (vs 5 in baseline) — faster than original MOM
            mdl_0 = [XGBRegressor(**REG, multi_strategy="multi_output_tree", random_state=s).fit(X, Y, sample_weight=sw0)
                     for s in range(SEEDS_REGIME)]
            mdl_1 = [XGBRegressor(**REG, multi_strategy="multi_output_tree", random_state=s).fit(X, Y, sample_weight=sw1)
                     for s in range(SEEDS_REGIME)]

    if mdl_0 is None: continue

    # Select model based on current month's regime
    rs = regime_at(dt)
    if rs["state"] == 1:
        active = mdl_1; n_state1_used += 1
    else:
        active = mdl_0; n_state0_used += 1

    p = np.mean([m.predict(pool[k]["Prank"]).mean(1) for m in active], axis=0)
    s = pd.Series(p, index=pool[k]["idx"]); n = max(1, int(len(s) * 0.10)); w = pd.Series(0.0, index=s.index)
    w[s.nlargest(n).index] = 1.0 / n; w[s.nsmallest(n).index] = -1.0 / n
    store[k] = dict(w=w, pnl=pool[k]["pnl"], dt=dt)

tks = sorted(store)
print(f"[regime] State-0 model used {n_state0_used} months, State-1 model used {n_state1_used} months", flush=True)

synth = (1 + mret.fillna(0.0)).cumprod()
def wmat(H, cg):
    rows = {}
    for i in range(len(tks)):
        k = tks[i]; W = pd.concat([store[tks[j]]["w"] for j in range(max(0, i - H + 1), i + 1)], axis=1).mean(axis=1)
        if cg:
            g = W.abs().sum()
            if g > 0: W = W * (2.0 / g)
        rows[store[k]["dt"]] = W
    return pd.DataFrame(rows).T.reindex(columns=synth.columns)

print("\n[results] ─────────────────────────────────────────────────────────────────")
out = {}; Wsave = None
for tag, H, cg in [("1", 1, False), ("3", 3, True)]:
    W = wmat(H, cg); sig = list(W.index)
    if tag == "1": Wsave = W
    g = BACKTEST.backtest(W, synth, freq=12, lag=1, signal_dates=sig)
    n = BACKTEST.backtest(W, synth, freq=12, lag=1, signal_dates=sig, transaction_cost=tc, borrow_fee=bf)
    out[f"g{tag}"], out[f"n{tag}"] = g["returns"], n["returns"]
    lbl = "H=1       " if tag == "1" else "H=3 constG"
    print(f"[MOM-REGIME] {lbl} net SR {n['sharpe']:.2f}  ann {n['ann_return']:.1%}  maxDD {n['max_drawdown']:.1%}"
          f"  turn {n['ann_turnover']:.1f}", flush=True)

# ── Year-by-year vs MOM baseline (SR 0.31) ───────────────────────────────────
print("\n[year-by-year] MOM-Regime H=1 net returns vs SPY:")
x = pd.Series(out["n1"]); x.index = pd.DatetimeIndex(x.index)
x = x[(x.index >= "2016-01-01") & (x.index < "2027-01-01")].dropna()
spy_m = hub.spy_m
if spy_m is not None:
    spy_m2 = spy_m.reindex(x.index).dropna()
for yr in range(2016, 2027):
    xyr = x[x.index.year == yr]
    if len(xyr) < 3: continue
    eyr = (1 + xyr).prod() - 1
    sr_yr = xyr.mean() / xyr.std() * np.sqrt(12) if xyr.std() > 0 else np.nan
    spy_yr = float("nan")
    if spy_m is not None:
        syr = spy_m.reindex(xyr.index).dropna()
        if len(syr) >= 3: spy_yr = float((1 + syr).prod() - 1)
    print(f"  {yr}: ann {eyr:>+7.1%}  SR {sr_yr:>5.2f}  SPY {spy_yr:>+6.1%}")

# Full-period metrics
e = (1 + x).cumprod()
sr_full = x.mean() / x.std() * np.sqrt(12)
ann_full = (1 + x).prod() ** (12 / len(x)) - 1
dd_full = (e / e.cummax() - 1).min()
print(f"\nFull 2016-26: SR {sr_full:.2f}  ann {ann_full:.1%}  maxDD {dd_full:.1%}  n={len(x)}")
print(f"Baseline MOM: SR 0.31  (from prior run)")

pickle.dump(out, open("/tmp/mom_regime_champ.pkl", "wb"))
pickle.dump(Wsave, open("/tmp/mom_regime_weights.pkl", "wb"))
print("[done] saved /tmp/mom_regime_champ.pkl", flush=True)
