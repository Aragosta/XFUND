#!/usr/bin/env python3
"""Capstone: LM (LambdaMART/Poh features) vs DM (Han bimodal, classifier+RET) on the SAME daily
~526-name universe → corr(LM, DM) and the combined z(LM)+z(DM) L/S book. Both use inverse-vol
decile construction. GROSS + flat-cost (10bp/side) net check. (No volume in daily data → no impact
model; capacity needs volume.)"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRanker, XGBClassifier
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS

N_SEEDS = 3
RANKER = dict(objective="rank:pairwise", eval_metric="ndcg", n_estimators=100, max_depth=6,
              learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

print("[load] daily ...", flush=True)
px = pd.read_parquet("tiingo_daily_checkpoint.parquet").sort_index()
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
pm_ = px.reindex(me); rm = pm_.pct_change(); T = len(me)
size = pd.DataFrame(1.0, index=me, columns=px.columns)                 # no volume → inert size dummies
elig = (pm_ > 1.0) & (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")

# ── Poh features (for LM) ──
print("[feat] Poh/Baz (LM) ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); PF = list(pf.keys())

# ── Han features (for DM) ──
print("[feat] Han (DM) FFD ...", flush=True)
first_feat = max(MOM_WINDOWS) + 1
ffd = d._ffd_from_training_window(pm_, first_feat + 48)
for m in list(ffd): ffd[m] = ffd[m].reindex(me)

# ── shared monthly pool: common eligible names with BOTH feature sets valid ──
print("[pool] ...", flush=True)
pool = {}
for k in range(first_feat, T - 1):
    dt = me[k]
    P = pd.DataFrame({c: pf[c].loc[dt] for c in PF})
    H = make_features(rm, size, k, ffd_scores=ffd)
    fwd = rm.iloc[k + 1]
    ok = P.notna().all(axis=1) & H.notna().all(axis=1) & fwd.notna() & elig.loc[dt].fillna(False)
    idx = H.index[ok.reindex(H.index).fillna(False).values]
    if len(idx) < 50: continue
    pool[k] = dict(P=P.loc[idx], H=H.loc[idx], fwd=fwd.reindex(idx),
                   dec=d._decile_labels(fwd.reindex(idx)), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
print(f"[pool] {len(keys)} months, predict from {pool[keys[fp]]['dt'].date()}", flush=True)

def invvol_book(score, k):
    p = pool[k]; s = pd.Series(score, index=p["fwd"].index); n = max(1, int(len(s) * 0.10))
    iv = (1.0 / (vol_d.loc[p["dt"]].reindex(s.index) * np.sqrt(21))).clip(upper=50)
    w = pd.Series(0.0, index=s.index)
    lo, sh = s.nlargest(n).index, s.nsmallest(n).index
    w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
    return w

# ── walk-forward: LM (ranker) and DM (classifier+RET) ──
print("[fit] LM + DM walk-forward ...", flush=True)
sc_LM, sc_DM = {}, {}
lm_store = dm_store = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month == 1 or lm_store is None:
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt][-120:]
        if len(tr) >= 36:
            # LM
            X = pd.concat([pool[t]["P"] for t in tr]); yrel = pd.concat([(9 - pool[t]["dec"]) for t in tr])
            qid = np.concatenate([np.full(len(pool[t]["P"]), j) for j, t in enumerate(tr)])
            lm_store = [XGBRanker(**RANKER, random_state=s).fit(X, yrel, qid=qid) for s in range(N_SEEDS)]
            # DM
            XH = pd.concat([pool[t]["H"] for t in tr]); yH = pd.concat([pool[t]["dec"] for t in tr])
            fwdH = pd.concat([pool[t]["fwd"] for t in tr]); labH = pd.concat([pool[t]["dec"] for t in tr])
            grp = pd.DataFrame({"l": labH.values, "r": fwdH.values}).groupby("l")["r"]
            mu_k = np.array([grp.get_group(c).mean() if c in grp.groups else 0.0 for c in range(10)])
            dmp = {kk2: vv2 for kk2, vv2 in d.XGB_PARAMS.items() if kk2 != "early_stopping_rounds"}
            dm_store = ([XGBClassifier(**dmp, random_state=s).fit(XH, yH) for s in range(N_SEEDS)], mu_k)
            print(f"  [{dt.year}] train={len(tr)}", flush=True)
    if lm_store is None: continue
    sc_LM[k] = np.mean([m.predict(pool[k]["P"]) for m in lm_store], axis=0)
    probs = np.mean([m.predict_proba(pool[k]["H"]) for m in dm_store[0]], axis=0)
    sc_DM[k] = d.score_ret(probs, dm_store[1])

# ── build return streams: LM, DM, combined z(LM)+z(DM) ──
def z(a): a = pd.Series(a); return (a - a.mean()) / (a.std() + 1e-9)
def stream(which):
    out = {}
    for k in sc_LM:
        if which == "LM": s = sc_LM[k]
        elif which == "DM": s = sc_DM[k]
        else: s = (z(sc_LM[k]).values + z(sc_DM[k]).values)
        w = invvol_book(s, k); out[pool[k]["dt"]] = float((w * pool[k]["fwd"]).sum())
    return pd.Series(out).dropna()

def perf(r, cost_side=0.0):
    r = r.dropna()
    if cost_side: r = r - cost_side * 2 * 0.20        # ~20% one-way turnover-of-book proxy per month
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol, mdd

rLM, rDM, rCB = stream("LM"), stream("DM"), stream("CB")
cc = pd.DataFrame({"LM": rLM, "DM": rDM}).dropna().corr().iloc[0, 1]

print("\n" + "=" * 70)
print("=== LM vs DM vs COMBINED on same daily ~526-name universe ===")
print(f"{'model':30}{'ann':>8}{'sharpe':>8}{'maxDD':>9} | {'net10bp SR':>10}")
for nm, r in [("LM (LambdaMART/Poh)", rLM), ("DM (Han bimodal)", rDM), ("COMBINED z(LM)+z(DM)", rCB)]:
    a, s, m = perf(r); _, sn, _ = perf(r, cost_side=0.001)
    print(f"{nm:30}{a:>8.1%}{s:>8.2f}{m:>9.1%} | {sn:>10.2f}")
print(f"\ncorr(LM, DM) = {cc:.2f}   n={len(pd.DataFrame({'a':rLM,'b':rDM}).dropna())} months")
print("[done]", flush=True)
