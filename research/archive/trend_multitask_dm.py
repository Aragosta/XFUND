#!/usr/bin/env python3
"""Multi-TASK on DM's Han features: does joint [t+1,t+2,t+3] help HAN features too, or do DM's
CONFLICTING horizons (t+1 reversal vs t+2/t+3 momentum) make the consensus mush? (mirror of LTR)

XGBoost multi-output regression (multi_strategy=multi_output_tree, shared vector-leaf trees) on
cross-sectionally z-scored forward returns. Consensus (mean of predicted horizons) = denoised trend
forecast → rank → inverse-vol decile L/S. Compared vs single-target (t+1) regressor, same construction.
Daily universe, Poh features, gross.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRegressor
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS

N_SEEDS = 3
BASE = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
mret = px.reindex(me).pct_change(); T = len(me)
elig = (px.reindex(me) > 1.0) & (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")

size = pd.DataFrame(1.0, index=me, columns=px.columns)                # no volume -> inert size dummies
print("[feat] Han FFD ...", flush=True)
ffd = d._ffd_from_training_window(pm_ := px.reindex(me), max(MOM_WINDOWS) + 1 + 48)
for _m in list(ffd): ffd[_m] = ffd[_m].reindex(me)
HAN = {k: make_features(px.reindex(me).pct_change(), size, k, ffd_scores=ffd) for k in range(max(MOM_WINDOWS)+1, T-3)}
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)

print("[pool] multi-horizon targets ...", flush=True)
pool = {}
for k in range(max(MOM_WINDOWS)+1, T - 3):
    if k not in HAN: continue
    dt = me[k]; P = HAN[k]
    pnl = mret.iloc[k + 1]
    ok = P.notna().all(axis=1) & mret.iloc[k+1].notna() & mret.iloc[k+2].notna() & mret.iloc[k+3].notna() & elig.loc[dt].fillna(False)
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])   # z-scored fwd rets
    pool[k] = dict(P=P.loc[idx], Y=Y, pnl=pnl.reindex(idx), dt=dt)
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

print("[fit] single vs multi-output ...", flush=True)
single, multi = {}, {}; ss = ms = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month == 1 or ss is None:
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt][-120:]
        if len(tr) >= 36:
            X = pd.concat([pool[t]["P"] for t in tr]).values
            Y = np.vstack([pool[t]["Y"] for t in tr])
            ss = [XGBRegressor(**BASE, random_state=s).fit(X, Y[:, 0]) for s in range(N_SEEDS)]         # t+1 only
            ms = [XGBRegressor(**BASE, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]  # joint
            print(f"  [{dt.year}]", flush=True)
    if ss is None: continue
    Xk = pool[k]["P"].values
    single[k] = np.mean([m.predict(Xk) for m in ss], axis=0)
    multi[k]  = np.mean([m.predict(Xk).mean(axis=1) for m in ms], axis=0)                                # consensus trend

print("\n" + "=" * 66)
print("=== Multi-task trend on HAN/DM features: single t+1 vs joint [t+1,t+2,t+3] ===")
print(f"{'model':34}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}")
for nm, sc in [("single-target (t+1)", single), ("multi-output [t+1,t+2,t+3] mean", multi)]:
    a, s, m, t = bookify(sc); print(f"{nm:34}{a:>8.1%}{s:>8.2f}{m:>9.1%}{t:>10.0%}")
print("[done]", flush=True)
