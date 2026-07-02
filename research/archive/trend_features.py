#!/usr/bin/env python3
"""Do FFD + slope-t-stat features lift the multi-output trend mean (baseline Poh = 1.05)?
  tval{3,6,9,12} : t-stat of OLS slope of DAILY log-price over 63/126/189/252 days (trend strength),
                   cross-sectionally z-scored per month.
  ffd{1,3,12}    : FFD log-price level + slopes (build_ffd_scores_v2), cross-sectionally z per month.
Same multi-output regressor, rank by mean of [t+1,t+2,t+3]. Baseline (Poh) vs Extended on identical
rows (pool requires all features valid) for a clean marginal test. Daily 750, inverse-vol decile L/S, gross."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRegressor
import deep_momentum_xgb as d

N_SEEDS = 3
BASE = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
pm_ = px.reindex(me); mret = pm_.pct_change(); T = len(me)
elig = (pm_ > 1.0) & (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)                       # cross-sectional (per-month) z
def zrow(df): return df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0)       # per-row cross-sectional z

print("[feat] Poh ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); POH = list(pf.keys())

print("[feat] slope t-stat (daily OLS) ...", flush=True)
def rolling_tstat(logpx, n):
    x = np.arange(n, dtype=float); xc = x - x.mean(); Sxx = float((xc**2).sum()); xcr = xc[::-1].copy()
    A = logpx.values; Tn, C = A.shape; out = np.full((Tn, C), np.nan); ones = np.ones(n)
    for j in range(C):
        col = A[:, j]; fin = np.isfinite(col).astype(float); y0 = np.where(np.isfinite(col), col, 0.0)
        cnt = np.convolve(fin, ones, "valid"); Sy = np.convolve(y0, ones, "valid")
        Sy2 = np.convolve(y0 * y0, ones, "valid"); Sxy = np.convolve(y0, xcr, "valid")
        slope = Sxy / Sxx; Syy = Sy2 - Sy * Sy / n
        SSE = np.maximum(Syy - Sxy * Sxy / Sxx, 1e-12); se = np.sqrt(SSE / (n - 2) / Sxx)
        tv = slope / se; tv[cnt < n] = np.nan; out[n - 1:, j] = tv
    return pd.DataFrame(out, index=logpx.index, columns=logpx.columns)
logpx = np.log(px)
TVAL = {}
for w, nd in [(3, 63), (6, 126), (9, 189), (12, 252)]:
    TVAL[f"tval{w}"] = zrow(at_me(rolling_tstat(logpx, nd)))

print("[feat] FFD ...", flush=True)
first_2011 = int(np.argmax(me.year >= 2011))
ffd = d._ffd_from_training_window(pm_, first_2011)
FFD = {f"ffd{m}": zrow(v.reindex(me)) for m, v in ffd.items()}

EXT = POH + list(TVAL) + list(FFD)
print(f"[cols] POH={len(POH)}  EXT={len(EXT)} (added {list(TVAL)+list(FFD)})", flush=True)

print("[pool] ...", flush=True)
pool = {}
for k in range(13, T - 3):
    dt = me[k]
    row = {c: pf[c].loc[dt] for c in POH}
    row.update({c: TVAL[c].loc[dt] for c in TVAL}); row.update({c: FFD[c].loc[dt] for c in FFD})
    P = pd.DataFrame(row); pnl = mret.iloc[k + 1]
    ok = P[POH].notna().all(axis=1) & mret.iloc[k+1].notna() & mret.iloc[k+2].notna() & mret.iloc[k+3].notna() & elig.loc[dt].fillna(False)
    idx = P.index[ok.values]                                              # require only POH valid (full universe)
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    pool[k] = dict(P=P.loc[idx].fillna(0.0), Y=Y, pnl=pnl.reindex(idx), dt=dt)   # neutral-fill z-scored extras
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)

def bookify(scores_by_k):
    rows, prevw = [], pd.Series(dtype=float)
    for k, sc in scores_by_k.items():
        s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
        iv = (1 / (vol_d.loc[pool[k]["dt"]].reindex(s.index) * np.sqrt(21))).clip(upper=50)
        w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
        w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
        rows.append((pool[k]["dt"], float((w * pool[k]["pnl"]).sum()), w.subtract(prevw, fill_value=0).abs().sum())); prevw = w
    df = pd.DataFrame(rows, columns=["dt", "r", "to"]).set_index("dt"); r = df["r"]
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol, mdd, df["to"].mean() * 12

def run(cols):
    sc = {}; ms = None
    for i in range(fp, len(keys)):
        k = keys[i]; dt = pool[k]["dt"]
        if dt.month == 1 or ms is None:
            tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt][-120:]
            if len(tr) >= 36:
                X = pd.concat([pool[t]["P"][cols] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
                ms = [XGBRegressor(**BASE, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
        if ms is None: continue
        sc[k] = np.mean([m.predict(pool[k]["P"][cols].values).mean(1) for m in ms], axis=0)
    return bookify(sc)

print("\n" + "=" * 66)
print("=== Trend features: baseline (Poh) vs extended (+tval +FFD), gross ===")
print(f"{'feature set':30}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}")
for nm, cols in [("Poh only (baseline)", POH), ("Poh + tval + FFD", EXT)]:
    print(f"[run] {nm} ...", flush=True); a, s, m, t = run(cols)
    print(f"{nm:30}{a:>8.1%}{s:>8.2f}{m:>9.1%}{t:>10.0%}", flush=True)
print("[done]", flush=True)
