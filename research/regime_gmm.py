#!/usr/bin/env python3
"""regime_gmm.py — Unsupervised 2-state regime detection via Gaussian Mixture Model.

Features (all computable from DataHub — no VIX feed required):
  1. vol          : 21-day trailing SPY realized vol (annualized)
  2. disp         : cross-sectional std of eligible stocks' monthly returns
  3. skew         : cross-sectional skewness of monthly returns (negative = crash risk)
  4. spy_ret      : SPY 1-month return (sign + magnitude)
  5. squeeze_rate : fraction of eligible stocks with monthly return > +25%

GMM is fit on an expanding window (no look-ahead at time t). Each month gets a soft
probability (p_squeeze) of being in the high-risk / short-squeeze regime (State 1).

State 0 = normal / trend  (low vol, moderate dispersion, orderly market)
State 1 = squeeze / chaos (high vol OR high dispersion OR junk rally)

Saved to /tmp/regime_states.pkl:
    dict{pd.Timestamp -> dict(state=int, p0=float, p1=float)}

Validation target: Jan 2021, Nov 2020, Apr 2020, early 2016 must land in State 1.
"""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from sklearn.mixture import GaussianMixture
from DATAHUB import DataHub

hub = DataHub()
me, mret, elig = hub.me, hub.mret, hub.elig("liquid")

# ── load daily SPY for realized vol ──────────────────────────────────────────
import os
spy_daily = None
for p in ("data/spy.parquet", "/tmp/spy.parquet"):
    if os.path.exists(p):
        try:
            s = pd.read_parquet(p)
            spy_daily = s[s.columns[0]] if hasattr(s,"columns") else s
            spy_daily = spy_daily.sort_index()
            break
        except Exception:
            pass

# ── build regime features, month-end aligned ─────────────────────────────────
feat = pd.DataFrame(index=me)

# 1. SPY realized vol (trailing 21 trading days → annualized)
if spy_daily is not None:
    spy_ret_d = spy_daily.pct_change()
    rvol = spy_ret_d.rolling(21, min_periods=10).std() * np.sqrt(252)
    # align to month-end: take value on the last trading day of each calendar month
    rvol_m = rvol.resample("ME").last()
    rvol_m.index = pd.DatetimeIndex([hub.me[hub.me <= d][-1] if any(hub.me <= d) else d
                                      for d in rvol_m.index])
    feat["vol"] = rvol_m.reindex(me)
else:
    # fallback: cross-sectional vol proxy from price panel
    feat["vol"] = mret.where(elig).std(axis=1).rolling(3).mean()

# 2. Cross-sectional return dispersion (std of cross-section each month)
feat["disp"] = mret.where(elig).std(axis=1)

# 3. Cross-sectional skewness (negative skew → crash regime for short books)
feat["skew"] = mret.where(elig).skew(axis=1)

# 4. SPY monthly return
spy_m = hub.spy_m
if spy_m is not None:
    feat["spy_ret"] = spy_m.reindex(me)
else:
    feat["spy_ret"] = mret.where(elig).mean(axis=1)   # equal-weight market return

# 5. Squeeze rate: fraction of eligible stocks with monthly return > +25%
thresh = 0.25
def squeeze_rate(row):
    v = row.dropna()
    return (v > thresh).sum() / max(len(v), 1)
feat["squeeze_rate"] = mret.where(elig).apply(squeeze_rate, axis=1)

feat = feat.dropna(how="all")
feat = feat.ffill().bfill()
feat = feat[feat.index >= "2011-01-01"]

print(f"Regime features: {feat.shape[0]} months, {feat.shape[1]} features")
print(f"Feature means:\n{feat.mean().round(4)}\n")

# ── expanding-window GMM — no look-ahead ─────────────────────────────────────
MIN_TRAIN = 36      # minimum months before we start fitting
REFIT_EVERY = 6     # refit GMM every N months

state_records = {}
gmm = None
last_fit_i = -999

for i, d in enumerate(feat.index):
    if i < MIN_TRAIN:
        state_records[d] = dict(state=0, p0=1.0, p1=0.0)
        continue

    # Refit expanding GMM periodically
    if gmm is None or (i - last_fit_i) >= REFIT_EVERY:
        past = feat.iloc[:i].values
        # Standardize using past data only
        mu = past.mean(axis=0); sig = past.std(axis=0) + 1e-9
        past_std = (past - mu) / sig
        try:
            gmm = GaussianMixture(n_components=2, covariance_type="full",
                                  n_init=10, random_state=42, max_iter=500)
            gmm.fit(past_std)
            # Identify which component = "squeeze" (higher vol + dispersion)
            vol_idx = list(feat.columns).index("vol")
            means = gmm.means_[:, vol_idx]
            squeeze_comp = int(np.argmax(means))  # higher vol = squeeze state
            gmm._squeeze_comp = squeeze_comp
            gmm._mu = mu; gmm._sig = sig
        except Exception:
            gmm = None
        last_fit_i = i

    if gmm is None:
        state_records[d] = dict(state=0, p0=1.0, p1=0.0)
        continue

    x = feat.loc[d].values
    x_std = (x - gmm._mu) / gmm._sig
    proba = gmm.predict_proba(x_std.reshape(1, -1))[0]
    p_squeeze = float(proba[gmm._squeeze_comp])
    p_normal  = 1.0 - p_squeeze
    state = 1 if p_squeeze >= 0.5 else 0
    state_records[d] = dict(state=state, p0=p_normal, p1=p_squeeze)

# ── results ───────────────────────────────────────────────────────────────────
states = pd.DataFrame(state_records).T
states.index = pd.DatetimeIndex(states.index)
squeeze_months = states[states["state"] == 1].index

print("=" * 72)
print("REGIME GMM — State 1 (squeeze / high-risk) months:")
for d in squeeze_months:
    p1 = state_records[d]["p1"]
    spy_r = feat.loc[d, "spy_ret"] if "spy_ret" in feat.columns else np.nan
    disp  = feat.loc[d, "disp"]
    print(f"  {str(d)[:7]}  p_squeeze={p1:.2f}  spy_ret={spy_r:>+6.1%}  disp={disp:.3f}")

print(f"\nState 0 (normal): {(states['state']==0).sum()} months")
print(f"State 1 (squeeze): {(states['state']==1).sum()} months")

# Validate known crash months are in State 1
known_squeezes = ["2020-04", "2020-11", "2021-01", "2016-07", "2016-11"]
print("\nValidation — known squeeze months:")
for ym in known_squeezes:
    matches = [d for d in states.index if f"{d.year}-{d.month:02d}" == ym]
    for d in matches:
        st = state_records[d]
        mark = "✓ State 1" if st["state"] == 1 else f"✗ State 0 (p1={st['p1']:.2f})"
        print(f"  {ym}: {mark}")

# Cluster means (interpretable)
print("\nGMM cluster means (last-fit, standardized units):")
if gmm is not None:
    for comp_i, label in enumerate(["State 0 (normal)", "State 1 (squeeze)"]):
        actual_comp = comp_i if gmm._squeeze_comp == 1 else (1 - comp_i)
        mean_raw = gmm.means_[actual_comp] * gmm._sig + gmm._mu
        print(f"  {label}: " + "  ".join(f"{feat.columns[j]}={mean_raw[j]:.3f}" for j in range(len(feat.columns))))

# Save
pickle.dump(state_records, open("/tmp/regime_states.pkl", "wb"))
print(f"\nSaved /tmp/regime_states.pkl ({len(state_records)} months)")
