#!/usr/bin/env python3
"""vq_regime.py — VALUE x QUALITY (VQ) with GMM regime-conditioned sample-weighted training.

Two XGBoost models per retrain:
  mdl_0: fit on training pool weighted by p_state0 (normal regime months get more weight)
  mdl_1: fit on training pool weighted by p_state1 (squeeze regime months get more weight)

At prediction month d: use regime_states[d].state to select the active model.
BETANEUT is OMITTED to isolate the regime conditioning effect.
EMB=12 months (future fundamentals embargo). Retrains every 3 months.
Pre-registered bar: net SR Δ ≥ +0.20 vs VQ baseline (SR 0.54).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
from scipy.stats import norm
from xgboost import XGBRegressor
from DATAHUB import DataHub
import BACKTEST

hub = DataHub()
me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
sec = hub.sector
tc = BACKTEST.tiered_transaction_costs(hub.mdv)
bf = BACKTEST.tiered_borrow_fees(hub.mdv)
f = hub.fund; mcap = hub.mcap()
gpa  = ((f("rev") - f("cogs")) / f("assets")).where(f("assets") > 0)
roe  = (f("ni") / f("book")).where(f("book") > 0)
bm   = f("book") / mcap

def d12(x): return x / x.shift(12) - 1
feat = {
    "bm": bm, "ep": f("ni")/mcap, "sp": f("rev")/mcap, "cfp": f("ocf")/mcap,
    "gpa": gpa, "roe": roe, "roa": f("ni")/f("assets"),
    "ocfa": f("ocf")/f("assets"), "gmar": f("gross_profit")/f("rev"),
    "ag": d12(f("assets")), "capexa": f("capex")/f("assets"),
    "accr": (f("ni") - f("ocf")) / f("assets"),
    "lev": f("debt_lt")/f("assets"), "curr": f("assets_cur")/f("liab_cur"),
    "casha": f("cash")/f("assets"), "revg": d12(f("rev")), "nig": d12(f("ni")),
    "size": np.log(mcap)
}
def zc(df): z = df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0); return z.clip(-3, 3)
F = {k: zc(v.replace([np.inf, -np.inf], np.nan)) for k, v in feat.items()}
COLS = list(F)
def grank(a): r = pd.Series(a).rank(method="average"); return norm.ppf((r - 0.5) / max(len(r), 2))

# ── Load regime states ────────────────────────────────────────────────────────
regime_states = pickle.load(open("/tmp/regime_states.pkl", "rb"))
def regime_at(dt):
    return regime_states.get(dt, {"state": 0, "p0": 1.0, "p1": 0.0})

MIN_EFF = 24
EMB = 12
fut_gpa = gpa.shift(-12); fut_roe = roe.shift(-12)

def secr(row, idx):
    r = row.where(elig.loc[d0]); r = r - r.groupby(sec).transform("mean")
    return grank(r.reindex(idx).values)

# Build pool
pool = {}
for i, d in enumerate(me):
    live = elig.loc[d] & mcap.loc[d].notna(); idx = live[live].index
    if len(idx) < 100 or i + EMB >= len(me): continue
    d0 = d
    X = np.column_stack([F[c].loc[d].reindex(idx).fillna(0.0).values for c in COLS])
    Y = np.column_stack([secr(fut_gpa.loc[d], idx), secr(fut_roe.loc[d], idx)])
    pool[d] = dict(X=X.astype(np.float32), idx=idx, Y=Y, bm=bm.loc[d].reindex(idx))
dates = [d for d in me if d in pool]

REG = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           tree_method="hist", multi_strategy="multi_output_tree", verbosity=0)

# ── Walk-forward with regime-conditioned models ───────────────────────────────
print("[walk] VQ regime-conditioned (no betaneut) ...", flush=True)
W = pd.DataFrame(0.0, index=me, columns=m_px.columns)
mdl_0 = None; mdl_1 = None; mdl_full = None
n_state0_used = 0; n_state1_used = 0; n_fallback_used = 0

for j, d in enumerate(dates):
    if j < 60: continue
    if mdl_0 is None or j % 3 == 0:
        tr = [dates[k] for k in range(j) if dates[k] <= me[max(0, me.get_loc(d) - EMB)]]
        if len(tr) < 48: continue
        Xtr = np.vstack([pool[t]["X"] for t in tr])
        Ytr = np.vstack([pool[t]["Y"] for t in tr])
        ok = np.isfinite(Ytr).all(1)
        Xtr_ok = Xtr[ok]; Ytr_ok = Ytr[ok]

        # Build per-sample weights from regime probabilities
        dates_ok = []
        for t in tr:
            dates_ok.extend([t] * len(pool[t]["X"]))
        dates_ok = np.array(dates_ok)[ok]

        segs_p0, segs_p1 = [], []
        for t in tr:
            rs = regime_at(t)
            n = len(pool[t]["X"])
            segs_p0.append(np.full(n, max(rs["p0"], 0.01), dtype=np.float32))
            segs_p1.append(np.full(n, max(rs["p1"], 0.01), dtype=np.float32))
        sw0_all = np.concatenate(segs_p0)[ok]
        sw1_all = np.concatenate(segs_p1)[ok]

        mdl_full = XGBRegressor(**REG).fit(Xtr_ok, Ytr_ok)

        eff0 = float(sw0_all.sum()); eff1 = float(sw1_all.sum())
        mdl_0 = XGBRegressor(**REG).fit(Xtr_ok, Ytr_ok, sample_weight=sw0_all) if eff0 >= MIN_EFF else mdl_full
        mdl_1 = XGBRegressor(**REG).fit(Xtr_ok, Ytr_ok, sample_weight=sw1_all) if eff1 >= MIN_EFF else mdl_full

    if mdl_0 is None: continue

    rs = regime_at(d)
    if rs["state"] == 1 and mdl_1 is not mdl_full:
        active = mdl_1; n_state1_used += 1
    elif rs["state"] == 0 and mdl_0 is not mdl_full:
        active = mdl_0; n_state0_used += 1
    else:
        active = mdl_full; n_fallback_used += 1

    q = active.predict(pool[d]["X"]).mean(1)
    s = pd.Series(q, index=pool[d]["idx"])
    # value x quality: add cheapness rank
    b = pool[d]["bm"]
    s = (s - s.mean()) / (s.std() + 1e-9) + (b.rank() - b.rank().mean()) / (b.rank().std() + 1e-9)
    # sector-neutral
    s = s - s.groupby(sec.reindex(s.index)).transform("mean")
    n = max(1, int(len(s.dropna()) * 0.10)); s = s.dropna()
    W.loc[d, s.nlargest(n).index] = 1.0 / n
    W.loc[d, s.nsmallest(n).index] = -1.0 / n

print(f"[regime] State-0 model {n_state0_used} months, State-1 model {n_state1_used} months, "
      f"fallback {n_fallback_used} months", flush=True)

sig = [d for d in W.index if W.loc[d].abs().sum() > 1e-9]
r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=1, signal_dates=sig,
                      transaction_cost=tc, borrow_fee=bf)
x = pd.Series(r["returns"]); x.index = pd.DatetimeIndex(x.index)
x = x[(x.index >= "2013-01-01") & (x.index < "2027-01-01")].dropna()

print("\n[results] ─────────────────────────────────────────────────────────────────")
e = (1 + x).cumprod()
sr = x.mean() / x.std() * np.sqrt(12)
ann = (1 + x).prod() ** (12 / len(x)) - 1
dd = (e / e.cummax() - 1).min()
print(f"[VQ-REGIME] net SR {sr:.2f}  ann {ann:.1%}  maxDD {dd:.1%}  turn {r['ann_turnover']:.1f}")
print(f"Baseline VQ:  SR 0.54  (from prior run)")

print("\n[year-by-year] VQ-Regime net returns vs SPY:")
spy = hub.spy_m
for yr in range(2013, 2027):
    xyr = x[x.index.year == yr]
    if len(xyr) < 3: continue
    eyr = (1 + xyr).prod() - 1
    sr_yr = xyr.mean() / xyr.std() * np.sqrt(12) if xyr.std() > 0 else np.nan
    spy_yr = float("nan")
    if spy is not None:
        syr = spy.reindex(xyr.index).dropna()
        if len(syr) >= 3: spy_yr = float((1 + syr).prod() - 1)
    print(f"  {yr}: ann {eyr:>+7.1%}  SR {sr_yr:>5.2f}  SPY {spy_yr:>+6.1%}")

pickle.dump(x.to_list(), open("/tmp/vq_regime_returns.pkl", "wb"))
print("[done] saved /tmp/vq_regime_returns.pkl", flush=True)
