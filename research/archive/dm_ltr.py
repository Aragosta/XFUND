#!/usr/bin/env python3
"""DM learning-to-rank upgrade (LambdaMART / XGBRanker) vs classification+RET DM.

Same features (make_features), tight ~561 universe, and cost engine as DM — the only change
is the model: an XGBRanker with a listwise NDCG (LambdaMART) objective that directly optimizes
the cross-sectional ordering, instead of 10-class softprob + RET reclassification. Params match
the Quantitativo LTR writeup (max_depth=5, lr=0.1, 200 trees). Relevance = forward-return decile
(higher return = higher grade). One query group per month.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRanker
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

Q, MIN_TRAIN, MAX_TRAIN = 0.05, 120, 120
RANKER = dict(objective="rank:ndcg", ndcg_exp_gain=False,   # linear gain: exp gain (default) inverts OOS
              n_estimators=200, max_depth=5, learning_rate=0.1,
              subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=0)

print("[load] universe ...", flush=True)
pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
T = len(rm); dates = rm.index
elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6, short_min_dollar_vol_abs=25e6)
oos = elig.index >= "2011-01-01"
print(f"[universe] tight eligible/mo={elig.sum(1)[oos].mean():.0f}  short/mo={short.sum(1)[oos].mean():.0f}", flush=True)
pnl = d._build_pnl_prices(pm)
tcost = BACKTEST.tiered_transaction_costs(sm); bfee = BACKTEST.tiered_borrow_fees(sm)

first_feat = max(MOM_WINDOWS) + 1
first_pred = MIN_TRAIN + first_feat
print("[ffd] optimal-d ...", flush=True)
ffd = d._ffd_from_training_window(pm, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(dates)

# ── pool: features + relevance grade (0=lowest .. 9=highest forward return) ──
print("[pool] features + relevance ...", flush=True)
pool = {}
for t in range(first_feat, T - 1):
    F = make_features(rm, sm, t, ffd_scores=ffd).dropna()
    idx = F.index.intersection(rm.iloc[t].dropna().index)
    e = elig.iloc[t - 1]; idx = idx.intersection(e.index[e.values])
    if len(idx) < 20:
        continue
    rel = (9 - d._decile_labels(rm.iloc[t].reindex(idx))).astype(int)   # higher grade = higher return
    pool[t] = (F.loc[idx], rel)

# ── walk-forward LambdaMART ranker (rolling 120m, annual retrain, 3-seed ensemble) ──
print("[LTR] walk-forward XGBRanker ...", flush=True)
rows, store = {}, {}
for year in sorted({dates[t].year for t in range(first_pred, T - 1)}):
    months = [t for t in range(first_pred, T - 1) if dates[t].year == year]
    if not months:
        continue
    t_cut = months[0]
    all_ts = sorted([t for t in pool if t < t_cut])
    if MAX_TRAIN and len(all_ts) > MAX_TRAIN:
        all_ts = all_ts[-MAX_TRAIN:]
    if len(all_ts) >= 18:
        X   = pd.concat([pool[t][0] for t in all_ts])
        y   = pd.concat([pool[t][1] for t in all_ts])
        qid = np.concatenate([np.full(len(pool[t][0]), i) for i, t in enumerate(all_ts)])  # one group per month
        print(f"  [{year}] train_pool={len(all_ts)}", flush=True)
        mdls = [XGBRanker(**RANKER, random_state=s).fit(X, y, qid=qid) for s in range(d.N_SEEDS)]
        store[year] = mdls
    elif store:
        store[year] = list(store.values())[-1]
    else:
        continue
    mdls = store[year]
    for t in months:
        if t not in pool:
            continue
        F = pool[t][0]
        score = np.mean([m.predict(F) for m in mdls], axis=0)       # higher = higher predicted return
        s_ser = pd.Series(score, index=F.index)
        n = max(1, int(len(s_ser) * Q))
        w = pd.Series(0.0, index=F.index)
        w[s_ser.nlargest(n).index] = 1.0 / n
        sh = short.iloc[t - 1]; s_short = s_ser.loc[s_ser.index.intersection(sh.index[sh.values])]
        w[s_short.nsmallest(n).index] = -1.0 / n
        rows[dates[t - 1]] = w
w_ltr = pd.DataFrame(rows).T.fillna(0.0).sort_index()

# ── DM baseline (classification + RET), same universe ──
print("[DM] classification+RET baseline ...", flush=True)
w_dm = d._generate_all_dm_weights(rm, sm, pm, min_train_months=MIN_TRAIN, max_train_months=MAX_TRAIN,
        q=Q, n_seeds=d.N_SEEDS, use_ffd=True, ffd_scores=ffd, portfolio="ls",
        eligible=elig, shortable=short)["ret"]

def run(w, tag):
    first = w.index[0]; px = pnl.loc[first:]
    sigs = [x for x in w.index if x in px.index]
    res = BACKTEST.backtest(w.reindex(columns=px.columns).fillna(0.0), px, freq=12, lag=0,
                            transaction_cost=tcost, borrow_fee=bfee, signal_dates=sigs)
    r = pd.Series(res["returns"]).dropna()
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12); dn = r[r < 0].std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    print(f"{tag:34}{ann:>8.1%}{(r.mean()*12)/vol:>8.2f}{(r.mean()*12)/dn:>9.2f}"
          f"{mdd:>9.1%}{res.get('ann_turnover', float('nan')):>10.1%}", flush=True)
    return r

print("\n" + "=" * 79)
print(f"=== DM Learning-to-Rank (LambdaMART) vs classification+RET (tight ~561, q={Q}) ===")
print(f"{'model':34}{'ann':>8}{'sharpe':>8}{'sortino':>9}{'maxDD':>9}{'ann.turn':>10}")
r_dm = run(w_dm, "DM (classification + RET)")
r_ltr = run(w_ltr, "DM-LTR (LambdaMART ranker)")
both = pd.DataFrame({"dm": r_dm, "ltr": r_ltr}).dropna()
print(f"\ncorr(DM, DM-LTR) = {both.corr().iloc[0,1]:.3f}   n={len(both)} months")
print("[done]", flush=True)
