#!/usr/bin/env python3
"""MOM-leg target test (fast, N_SEEDS=1): does a VOL-NORMALIZED target beat the raw-return target?
  raw     : Y_h = zc(r_{t+h})                    (baseline, ~1.05)
  volnorm : Y_h = zc(r_{t+h} / sigma_i)          (risk-adjusted — consistent w/ invvol construction)
Multi-output regressor, rank by mean[t+1,t+2,t+3], inverse-vol decile L/S, daily 750, gross."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRegressor

N_SEEDS = 1
BASE = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
mret = px.reindex(me).pct_change(); T = len(me)
elig = (px.reindex(me) > 1.0) & (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)

print("[feat] ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); PF = list(pf.keys())

print("[pool] both targets ...", flush=True)
pool = {}
for k in range(13, T - 3):
    dt = me[k]; P = pd.DataFrame({c: pf[c].loc[dt] for c in PF}); pn = mret.iloc[k + 1]
    ok = P.notna().all(axis=1) & mret.iloc[k+1].notna() & mret.iloc[k+2].notna() & mret.iloc[k+3].notna() & elig.loc[dt].fillna(False)
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    sig = (vol_d.loc[dt].reindex(idx) * np.sqrt(21))
    sig = sig.clip(lower=sig.median() * 0.3)                                   # realistic floor: no /near-zero
    def volnorm(fwd):
        v = fwd / sig; return zc(v.clip(v.quantile(0.02), v.quantile(0.98)))   # winsorize exploding tails
    Yraw = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    Yvn  = np.column_stack([volnorm(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    pool[k] = dict(P=P.loc[idx], Yraw=Yraw, Yvn=Yvn, pnl=pn.reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)

def bookify(sc_by_k):
    rows, prevw = [], pd.Series(dtype=float)
    for k, sc in sc_by_k.items():
        s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
        iv = (1 / (vol_d.loc[pool[k]["dt"]].reindex(s.index) * np.sqrt(21))).clip(upper=50)
        w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
        w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
        rows.append((pool[k]["dt"], float((w * pool[k]["pnl"]).sum()), w.subtract(prevw, fill_value=0).abs().sum())); prevw = w
    df = pd.DataFrame(rows, columns=["dt", "r", "to"]).set_index("dt"); r = df["r"]
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol, mdd, df["to"].mean() * 12

def run(key):
    sc = {}; ms = None
    for i in range(fp, len(keys)):
        k = keys[i]; dt = pool[k]["dt"]
        if dt.month == 1 or ms is None:
            tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt][-120:]
            if len(tr) >= 36:
                X = pd.concat([pool[t]["P"] for t in tr]).values; Y = np.vstack([pool[t][key] for t in tr])
                ms = [XGBRegressor(**BASE, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
        if ms is None: continue
        sc[k] = np.mean([m.predict(pool[k]["P"].values).mean(1) for m in ms], axis=0)
    return bookify(sc)

print("\n" + "=" * 60)
print("=== MOM target test (daily 750, gross, N_SEEDS=1) ===")
print(f"{'target':22}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}")
for nm, key in [("raw return (baseline)", "Yraw"), ("vol-normalized", "Yvn")]:
    print(f"[run] {nm} ...", flush=True); a, s, m, t = run(key)
    print(f"{nm:22}{a:>8.1%}{s:>8.2f}{m:>9.1%}{t:>10.0%}", flush=True)
print("[done]", flush=True)
