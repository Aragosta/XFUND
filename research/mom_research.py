#!/usr/bin/env python3
"""MOM upgrades study (fixed sleeve: Poh+FFD, delisting, upside-cap, expanding window):
  (1) universe width — top-1000 (base) vs top-2000 (more assets)
  (2) retrain freq   — quarterly (base) vs monthly
  (3) regressor+ranker ENSEMBLE — z(multi-output regressor) + z(LambdaMART rank:pairwise, 30-quantile)
N_SEEDS=1, clean broad, gross."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRegressor, XGBRanker
import deep_momentum_xgb as d
from features import MOM_WINDOWS

N_SEEDS, MINPX, MINDV, DELIST = 1, 5.0, 5e6, -0.30
REG = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
RNK = dict(objective="rank:pairwise", eval_metric="ndcg", n_estimators=200, max_depth=5, learning_rate=0.1,
           subsample=0.8, colsample_bytree=0.8, verbosity=0)                 # Quantitativo LTR config

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
m_px = px.reindex(me)
mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
V = m_px.notna().values; last = len(me) - 1
for j in range(V.shape[1]):
    w = np.where(V[:, j])[0]
    if len(w) and w[-1] < last - 1 and w[-1] + 1 < len(me): mret.iat[w[-1] + 1, j] = DELIST
vol_d = px.pct_change(fill_method=None).ewm(span=63, min_periods=20).std(); T = len(me)
dvm = (px * vb).resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mom_dvol = dvm.reindex(pd.PeriodIndex(me, freq="M")); mom_dvol.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
dvrank = mom_dvol.rank(axis=1, ascending=False)
base_elig = (m_px > MINPX) & (cov > 0.9) & (mom_dvol > MINDV)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)

print("[feat] Poh + FFD ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1).where(lambda z: z.abs() < 5); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); POH = list(pf.keys())
f2011 = int(np.argmax(me.year >= 2011)); ffd = d._ffd_from_training_window(m_px, f2011)
for m, v in ffd.items():
    z_ = v.reindex(me); pf[f"ffd{m}"] = z_.sub(z_.mean(1), 0).div(z_.std(1) + 1e-9, 0)
PF = list(pf.keys())

def build_pool(topn):
    pool = {}
    for k in range(13, T - 3):
        dt = me[k]; el = (base_elig.loc[dt] & (dvrank.loc[dt] <= topn)).fillna(False); idx0 = el.index[el.values]
        P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in PF}); pn = mret.iloc[k + 1].reindex(idx0)
        ok = P[POH].notna().all(axis=1) & mret.iloc[k+1].reindex(idx0).notna() & mret.iloc[k+2].reindex(idx0).notna() & mret.iloc[k+3].reindex(idx0).notna()
        idx = P.index[ok.values]
        if len(idx) < 50: continue
        fwd1 = mret.iloc[k+1].reindex(idx)
        Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
        r30 = ((fwd1.rank(method="first") - 1) * 30 // len(fwd1)).clip(upper=29).astype(int).values   # 30-quantile relevance
        pool[k] = dict(P=P.loc[idx].fillna(0.0), Y=Y, r30=r30, pnl=pn.reindex(idx), dt=dt)
    return pool

def run(pool, retrain="quarterly", model="reg"):
    keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
    rows, prevw, reg, rnk = [], pd.Series(dtype=float), None, None
    for i in range(fp, len(keys)):
        k = keys[i]; dt = pool[k]["dt"]
        fire = reg is None or (retrain == "quarterly" and dt.month in (1, 4, 7, 10)) or retrain == "monthly"
        if fire:
            tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt]        # expanding
            if len(tr) >= 36:
                X = pd.concat([pool[t]["P"] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
                reg = [XGBRegressor(**REG, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
                if model == "ens":
                    yr = np.concatenate([pool[t]["r30"] for t in tr]); qid = np.concatenate([np.full(len(pool[t]["P"]), j) for j, t in enumerate(tr)])
                    rnk = [XGBRanker(**RNK, random_state=s).fit(X, yr, qid=qid) for s in range(N_SEEDS)]
        if reg is None: continue
        Xk = pool[k]["P"].values
        sc = np.mean([m.predict(Xk).mean(1) for m in reg], axis=0)
        if model == "ens":
            rs = np.mean([m.predict(Xk) for m in rnk], axis=0)
            sc = zc(pd.Series(sc)).values + zc(pd.Series(rs)).values             # combine z-scores
        s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
        iv = (1 / (vol_d.loc[dt].reindex(s.index) * np.sqrt(21))).clip(upper=50)
        w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
        w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
        rows.append((dt, float((w * pool[k]["pnl"]).sum(skipna=True)), w.subtract(prevw, fill_value=0).abs().sum())); prevw = w
    df = pd.DataFrame(rows, columns=["dt", "r", "to"]).set_index("dt"); r = df["r"]
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol, mdd, df["to"].mean() * 12

print("[pool] top-1000 ...", flush=True); p1000 = build_pool(1000)
print("[pool] top-2000 ...", flush=True); p2000 = build_pool(2000)
print("\n" + "=" * 62)
print("=== MOM upgrades (Poh+FFD, delist, expanding, gross) ===")
print(f"{'config':34}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}")
for nm, pool, kw in [("base: top-1000, quarterly, reg", p1000, dict()),
                     ("top-2000 (more assets)",          p2000, dict()),
                     ("monthly retrain (top-1000)",      p1000, dict(retrain="monthly")),
                     ("reg+ranker ENSEMBLE (top-1000)",  p1000, dict(model="ens"))]:
    print(f"[run] {nm} ...", flush=True); a, s, m, t = run(pool, **kw)
    print(f"{nm:34}{a:>8.1%}{s:>8.2f}{m:>9.1%}{t:>10.0%}", flush=True)
print("[done]", flush=True)
