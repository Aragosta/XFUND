#!/usr/bin/env python3
"""Faithful replication of Poh-Lim-Zohren-Roberts 'Learning to Rank for CSM' on our DAILY universe.

Their recipe (not Han's DM features):
  features = raw cumulative returns (3/6/12m) + vol-normalized returns (3/6/12m)
             + Baz et al (2015) MACD trend signals over 3 timescale pairs + composite
  model    = LambdaMART, objective=rank:pairwise, eval_metric=ndcg
  label    = decile of next-month return (10 grades)
  build    = inverse-vol weighted L/S, top/bottom decile
Reported GROSS (their headline is pre-cost). JT benchmark (raw 12m return, same construction) for the
key comparison (their LM 2.16 vs JT 0.55). Daily 750-ticker checkpoint, monthly rebalance.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRanker

N_SEEDS = 3
RANKER = dict(objective="rank:pairwise", eval_metric="ndcg", n_estimators=100, max_depth=6,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

print("[load] daily ...", flush=True)
px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change()
vol_d = rets_d.ewm(span=63, min_periods=20).std()                     # ex-ante daily vol
me = px.index.to_series().resample("ME").last().dropna().values       # last trading day per month
me = pd.DatetimeIndex(me)

def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(frame): return frame.reindex(me, method="ffill")

# ── features (Baz MACD + returns), computed daily then sampled at month-ends ──
print("[feat] building Poh/Baz features ...", flush=True)
feat = {}
for m, d in [(3, 63), (6, 126), (12, 252)]:
    feat[f"ret{m}"]  = at_me(px / px.shift(d) - 1)
    feat[f"nret{m}"] = at_me((px / px.shift(d) - 1) / (vol_d * np.sqrt(d)))
comp = 0.0
for k, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    macd = px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()
    q = macd / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std()
    phi = y * np.exp(-y ** 2 / 4) / 0.89
    feat[f"y{k}"] = at_me(y); feat[f"phi{k}"] = at_me(phi); comp = comp + phi
feat["macd_comp"] = at_me(comp)
FCOLS = list(feat.keys())

# monthly returns + eligibility (price>$1, valid feature row)
mret = px.reindex(me).pct_change()
elig = (px.reindex(me) > 1.0) & at_me(px.notna().rolling(252, min_periods=200).mean() > 0.9)

def decile(s):
    r = s.rank(method="first"); return (((r - 1) * 10 // len(r)).clip(upper=9)).astype(int)

# ── assemble monthly pool: features (t) → label decile of next-month return ──
print("[pool] assembling ...", flush=True)
pool = {}
for k in range(len(me) - 1):
    dt = me[k]
    rows = pd.DataFrame({c: feat[c].loc[dt] for c in FCOLS})
    fwd = mret.iloc[k + 1]                                            # next-month return (label)
    ok = rows.notna().all(axis=1) & fwd.notna() & elig.loc[dt].fillna(False)
    idx = rows.index[ok.values]
    if len(idx) < 50: continue
    pool[k] = (rows.loc[idx], decile(fwd.reindex(idx)), fwd.reindex(idx))

keys = sorted(pool)
first_pred = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
print(f"[pool] {len(keys)} months, predict from {me[keys[first_pred]].date()}", flush=True)

def invvol_ls(score, dt, fwd):
    n = max(1, int(len(score) * 0.10))                               # top/bottom decile
    s = pd.Series(score, index=fwd.index)
    longs, shorts = s.nlargest(n).index, s.nsmallest(n).index
    iv = (1.0 / (vol_d.loc[dt].reindex(fwd.index) * np.sqrt(21))).clip(upper=50)  # inverse ex-ante monthly vol
    w = pd.Series(0.0, index=fwd.index)
    w[longs] = iv[longs] / iv[longs].sum(); w[shorts] = -iv[shorts] / iv[shorts].sum()
    return float((w * fwd).sum())

# ── walk-forward: LambdaMART (LM) and JT benchmark, annual retrain ──
def run(model_kind):
    rows, store = [], None
    for i in range(first_pred, len(keys)):
        k = keys[i]; dt = me[k]
        if model_kind == "JT":
            score = pool[k][0]["ret12"].values                       # raw 12m return ranking
        else:
            if dt.month == 1 or store is None:                       # annual retrain
                tr = [keys[j] for j in range(i) if me[keys[j]] < dt][-120:]
                if len(tr) >= 36:
                    X = pd.concat([pool[t][0] for t in tr]); y = pd.concat([pool[t][1] for t in tr])
                    qid = np.concatenate([np.full(len(pool[t][0]), j) for j, t in enumerate(tr)])
                    store = [XGBRanker(**RANKER, random_state=s).fit(X, y, qid=qid) for s in range(N_SEEDS)]
            if store is None: continue
            score = np.mean([m.predict(pool[k][0]) for m in store], axis=0)
        rows.append((dt, invvol_ls(score, dt, pool[k][2])))
    r = pd.Series(dict(rows)).dropna()
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return r, ann, (r.mean() * 12) / vol, mdd

print("[JT] benchmark ...", flush=True); rJT, aJT, sJT, dJT = run("JT")
print("[LM] LambdaMART ...", flush=True); rLM, aLM, sLM, dLM = run("LM")
corr = pd.DataFrame({"JT": rJT, "LM": rLM}).dropna().corr().iloc[0, 1]

print("\n" + "=" * 66)
print("=== Poh et al LTR replication (daily ~526 names, GROSS, their features) ===")
print(f"{'model':28}{'ann':>9}{'sharpe':>9}{'maxDD':>9}")
print(f"{'JT (raw 12m, invvol L/S)':28}{aJT:>9.1%}{sJT:>9.2f}{dJT:>9.1%}")
print(f"{'LM (LambdaMART, their feats)':28}{aLM:>9.1%}{sLM:>9.2f}{dLM:>9.1%}")
print(f"\ncorr(JT, LM) = {corr:.2f}   (paper: JT 0.55, LM 2.16 gross)")
print("[done]", flush=True)
