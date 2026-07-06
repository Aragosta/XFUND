#!/usr/bin/env python3
"""DM + ranker ENSEMBLE vs RET-only, on DM's tight survivorship-free universe (monthly, Han features).
  RET-only (base DM) : rank by RET = probs @ mu_k  (Han reclassification, law of total expectation)
  z(RET)+z(ranker)   : add LambdaMART ranker (Han features, 30-quantile t+1 relevance), blend z-scores
Does the ranker cut DM's -31% drawdown like it cut MOM's? Equal-weight decile L/S, gross, N_SEEDS=1."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBClassifier, XGBRanker
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS

N_SEEDS = 1
DMP = {k: v for k, v in d.XGB_PARAMS.items() if k != "early_stopping_rounds"}
RNK = dict(objective="rank:pairwise", eval_metric="ndcg", n_estimators=200, max_depth=5, learning_rate=0.1,
           subsample=0.8, colsample_bytree=0.8, verbosity=0)
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)

print("[DM] load + eligibility (tight) + FFD ...", flush=True)
pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6)   # tight ~561
T = len(rm); first_feat = max(MOM_WINDOWS) + 1; first_pred = 120 + first_feat
ffd = d._ffd_from_training_window(pm, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(rm.index)

print("[DM] pool (Han features) ...", flush=True)
pool = {}
for t in range(first_feat, T):
    F = make_features(rm, sm, t, ffd_scores=ffd).dropna()
    fwd = rm.iloc[t]; e = elig.iloc[t - 1]
    idx = F.index.intersection(fwd.dropna().index); idx = idx.intersection(e.index[e.values])
    if len(idx) < 50: continue
    fw = fwd.reindex(idx)
    r30 = ((fw.rank(method="first") - 1) * 30 // len(idx)).clip(upper=29).astype(int).values     # higher ret = higher grade
    pool[t] = dict(F=F.loc[idx], dec=d._decile_labels(fw), fwd=fw, r30=r30,
                   short=short.iloc[t - 1].reindex(idx).fillna(False).values, dt=rm.index[t])
keys = [t for t in sorted(pool) if t >= first_pred]
print(f"[DM] {len(keys)} OOS months from {pool[keys[0]]['dt'].date()}, ~{np.mean([len(pool[t]['F']) for t in keys[:12]]):.0f} names/mo", flush=True)

def run(mode):
    rows, prevw, clf, rk, mu_k = [], pd.Series(dtype=float), None, None, None
    allkeys = sorted(pool)
    for t in keys:
        dt = pool[t]["dt"]
        if dt.month == 1 or clf is None:
            tr = [tt for tt in allkeys if tt < t][-120:]
            if len(tr) >= 36:
                X = pd.concat([pool[tt]["F"] for tt in tr]); y = pd.concat([pool[tt]["dec"] for tt in tr])
                fwdtr = pd.concat([pool[tt]["fwd"] for tt in tr]).values; dectr = pd.concat([pool[tt]["dec"] for tt in tr]).values
                mu_k = np.array([fwdtr[dectr == c].mean() if (dectr == c).any() else 0.0 for c in range(10)])
                clf = [XGBClassifier(**DMP, random_state=s).fit(X, y) for s in range(N_SEEDS)]
                if mode == "ens":
                    yr = np.concatenate([pool[tt]["r30"] for tt in tr]); qid = np.concatenate([np.full(len(pool[tt]["F"]), j) for j, tt in enumerate(tr)])
                    rk = [XGBRanker(**RNK, random_state=s).fit(X, yr, qid=qid) for s in range(N_SEEDS)]
        if clf is None: continue
        Xk = pool[t]["F"]
        probs = np.mean([c.predict_proba(Xk) for c in clf], axis=0)
        ret = probs @ mu_k                                                    # RET reclassification
        sc = ret if mode == "ret" else zc(pd.Series(ret)).values + zc(pd.Series(np.mean([m.predict(Xk) for m in rk], axis=0))).values
        s = pd.Series(sc, index=Xk.index); n = max(1, int(len(s) * 0.10))
        w = pd.Series(0.0, index=s.index); w[s.nlargest(n).index] = 1.0 / n
        ss = s[pool[t]["short"]]; w[ss.nsmallest(n).index] = -1.0 / n         # short only shortable names
        rows.append((dt, float((w * pool[t]["fwd"]).sum()), w.subtract(prevw, fill_value=0).abs().sum())); prevw = w
    df = pd.DataFrame(rows, columns=["dt", "r", "to"]).set_index("dt"); r = df["r"]
    ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return r, ann, (r.mean() * 12) / vol, mdd, df["to"].mean() * 12

print("[run] RET-only ...", flush=True); rR, aR, sR, mR, tR = run("ret")
print("[run] z(RET)+z(ranker) ...", flush=True); rE, aE, sE, mE, tE = run("ens")
print("\n" + "=" * 60)
print("=== DM ensemble on tight universe (gross) ===")
print(f"{'book':26}{'ann':>8}{'sharpe':>8}{'maxDD':>9}{'ann.turn':>10}")
print(f"{'RET-only (base DM)':26}{aR:>8.1%}{sR:>8.2f}{mR:>9.1%}{tR:>10.0%}")
print(f"{'z(RET)+z(ranker)':26}{aE:>8.1%}{sE:>8.2f}{mE:>9.1%}{tE:>10.0%}")
print(f"\ncorr(RET, ensemble) returns = {pd.DataFrame({'a':rR,'b':rE}).dropna().corr().iloc[0,1]:.2f}")
print("[done]", flush=True)
