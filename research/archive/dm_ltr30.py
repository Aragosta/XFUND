#!/usr/bin/env python3
"""LTR retest with 30-quantile relevance (Quantitativo's optimum) — does finer granularity
flip LambdaMART from noise (spearman +0.025, -0.58 SR at 10 deciles) to a real signal?
LTR arm only; DM baseline on this tight universe is known (~18.3% / 1.00 SR)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRanker
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

Q, MIN_TRAIN, MAX_TRAIN, GRADES = 0.05, 120, 120, 30
RANKER = dict(objective="rank:ndcg", ndcg_exp_gain=False, n_estimators=200, max_depth=5,
              learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=0)

print("[load] universe ...", flush=True)
pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
T = len(rm); dates = rm.index
elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6, short_min_dollar_vol_abs=25e6)
pnl = d._build_pnl_prices(pm)
tcost = BACKTEST.tiered_transaction_costs(sm); bfee = BACKTEST.tiered_borrow_fees(sm)
first_feat = max(MOM_WINDOWS) + 1; first_pred = MIN_TRAIN + first_feat
print("[ffd] optimal-d ...", flush=True)
ffd = d._ffd_from_training_window(pm, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(dates)

def relevance(fwd, g=GRADES):
    r = fwd.rank(method="first")                                   # 1..n ascending (1=lowest return)
    return (((r - 1) * g // len(r)).clip(upper=g - 1)).astype(int)  # 0=lowest .. g-1=highest return

print(f"[pool] features + {GRADES}-grade relevance ...", flush=True)
pool = {}
for t in range(first_feat, T - 1):
    F = make_features(rm, sm, t, ffd_scores=ffd).dropna()
    idx = F.index.intersection(rm.iloc[t].dropna().index)
    e = elig.iloc[t - 1]; idx = idx.intersection(e.index[e.values])
    if len(idx) < 20: continue
    pool[t] = (F.loc[idx], relevance(rm.iloc[t].reindex(idx)))

print("[LTR] walk-forward ...", flush=True)
rows, store = {}, {}
for year in sorted({dates[t].year for t in range(first_pred, T - 1)}):
    months = [t for t in range(first_pred, T - 1) if dates[t].year == year]
    if not months: continue
    all_ts = sorted([t for t in pool if t < months[0]])
    if MAX_TRAIN and len(all_ts) > MAX_TRAIN: all_ts = all_ts[-MAX_TRAIN:]
    if len(all_ts) >= 18:
        X = pd.concat([pool[t][0] for t in all_ts]); y = pd.concat([pool[t][1] for t in all_ts])
        qid = np.concatenate([np.full(len(pool[t][0]), i) for i, t in enumerate(all_ts)])
        store[year] = [XGBRanker(**RANKER, random_state=s).fit(X, y, qid=qid) for s in range(d.N_SEEDS)]
    elif store:
        store[year] = list(store.values())[-1]
    else:
        continue
    for t in months:
        if t not in pool: continue
        F = pool[t][0]
        sc = np.mean([m.predict(F) for m in store[year]], axis=0)
        s = pd.Series(sc, index=F.index); n = max(1, int(len(s) * Q))
        w = pd.Series(0.0, index=F.index); w[s.nlargest(n).index] = 1.0 / n
        sh = short.iloc[t - 1]; ss = s.loc[s.index.intersection(sh.index[sh.values])]
        w[ss.nsmallest(n).index] = -1.0 / n
        rows[dates[t - 1]] = w
W = pd.DataFrame(rows).T.fillna(0.0).sort_index()

first = W.index[0]; px = pnl.loc[first:]
sigs = [x for x in W.index if x in px.index]
res = BACKTEST.backtest(W.reindex(columns=px.columns).fillna(0.0), px, freq=12, lag=0,
                        transaction_cost=tcost, borrow_fee=bfee, signal_dates=sigs)
r = pd.Series(res["returns"]).dropna()
ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12); dn = r[r < 0].std() * np.sqrt(12)
eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
print("\n" + "=" * 60)
print(f"=== DM-LTR, {GRADES}-grade relevance (tight ~561) ===")
print(f"  ann {ann:>7.1%}  sharpe {(r.mean()*12)/vol:>5.2f}  sortino {(r.mean()*12)/dn:>5.2f}"
      f"  maxDD {mdd:>6.1%}  turn {res.get('ann_turnover', float('nan')):>7.1%}")
print(f"  (ref: DM=1.00 SR;  10-grade LTR was -0.58 SR)")
W.to_parquet("/tmp/ltr30_weights.parquet")  # save for combine step if positive
print("[done]", flush=True)
