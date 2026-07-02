#!/usr/bin/env python3
"""Reproduce Quantitativo LTR on the BROAD universe (~3729 names, Russell-3000-like breadth):
30-grade relevance LambdaMART, exp_gain=False. Report GROSS (does the edge exist?) and NET at
small AUM via the square-root impact model (does it survive at the scale we'd run it?)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBRanker
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

Q, MIN_TRAIN, MAX_TRAIN, GRADES, ETA = 0.05, 120, 120, 30, 0.5
RANKER = dict(objective="rank:ndcg", ndcg_exp_gain=False, n_estimators=200, max_depth=5,
              learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, verbosity=0)

print("[load] universe ...", flush=True)
pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
T = len(rm); dates = rm.index
elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_abs=1e6, short_min_dollar_vol_abs=10e6)  # BROAD
oos = dates >= "2011-01-01"
print(f"[universe] broad eligible/mo={elig.sum(1)[oos].mean():.0f}", flush=True)
pnl = d._build_pnl_prices(pm); pnl_ret = pnl.pct_change()
dv3 = sm.rolling(3, min_periods=1).mean(); sig_m = rm.rolling(12, min_periods=6).std()
half_spread = BACKTEST.tiered_transaction_costs(sm); bfee = BACKTEST.tiered_borrow_fees(sm)
first_feat = max(MOM_WINDOWS) + 1; first_pred = MIN_TRAIN + first_feat
print("[ffd] optimal-d ...", flush=True)
ffd = d._ffd_from_training_window(pm, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(dates)

def relevance(fwd, g=GRADES):
    r = fwd.rank(method="first")
    return (((r - 1) * g // len(r)).clip(upper=g - 1)).astype(int)

print(f"[pool] features + {GRADES}-grade relevance (broad) ...", flush=True)
pool = {}
for t in range(first_feat, T - 1):
    F = make_features(rm, sm, t, ffd_scores=ffd).dropna()
    idx = F.index.intersection(rm.iloc[t].dropna().index)
    e = elig.iloc[t - 1]; idx = idx.intersection(e.index[e.values])
    if len(idx) < 20: continue
    pool[t] = (F.loc[idx], relevance(rm.iloc[t].reindex(idx)))

print("[LTR] walk-forward (broad) ...", flush=True)
rows, store = {}, {}
for year in sorted({dates[t].year for t in range(first_pred, T - 1)}):
    months = [t for t in range(first_pred, T - 1) if dates[t].year == year]
    if not months: continue
    all_ts = sorted([t for t in pool if t < months[0]])
    if MAX_TRAIN and len(all_ts) > MAX_TRAIN: all_ts = all_ts[-MAX_TRAIN:]
    if len(all_ts) >= 18:
        X = pd.concat([pool[t][0] for t in all_ts]); y = pd.concat([pool[t][1] for t in all_ts])
        qid = np.concatenate([np.full(len(pool[t][0]), i) for i, t in enumerate(all_ts)])
        print(f"  [{year}]", flush=True)
        store[year] = [XGBRanker(**RANKER, random_state=s).fit(X, y, qid=qid) for s in range(d.N_SEEDS)]
    elif store: store[year] = list(store.values())[-1]
    else: continue
    for t in months:
        if t not in pool: continue
        F = pool[t][0]
        sc = np.mean([m.predict(F) for m in store[year]], axis=0)
        s = pd.Series(sc, index=F.index); n = max(1, int(len(s) * Q))
        w = pd.Series(0.0, index=F.index); w[s.nlargest(n).index] = 1.0 / n
        sh = short.iloc[t - 1]; ss = s.loc[s.index.intersection(sh.index[sh.values])]
        w[ss.nsmallest(n).index] = -1.0 / n
        rows[dates[t - 1]] = w
W = pd.DataFrame(rows).T.fillna(0.0).sort_index(); sd = list(W.index)
W.to_parquet("/tmp/ltr_broad_weights.parquet")
print(f"[weights] {len(sd)} rebalances", flush=True)

def net_returns(aum):
    out, prev = [], pd.Series(0.0, index=W.columns)
    for d0 in sd:
        w = W.loc[d0]; nxt = pnl_ret.index[pnl_ret.index > d0]
        if len(nxt) == 0: break
        gross = float((w * pnl_ret.loc[nxt[0]]).sum(skipna=True))
        dw = (w - prev).abs(); tr = dw[dw > 0]
        if len(tr):
            sp = half_spread.loc[d0].reindex(tr.index).fillna(0.015)
            sg = sig_m.loc[d0].reindex(tr.index).fillna(0.15)
            dvm = dv3.loc[d0].reindex(tr.index).replace(0, np.nan).fillna(1e5)
            cost = float(((sp + ETA * sg * np.sqrt((tr * aum / dvm).clip(lower=0))) * tr).sum())
        else: cost = 0.0
        brw = float((w.clip(upper=0).abs() * bfee.loc[d0].reindex(w.index).fillna(0.05)).sum()) / 12.0
        out.append(gross - cost - brw); prev = w
    return pd.Series(out)

def perf(r):
    r = pd.Series(r).dropna(); ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol if vol > 0 else np.nan, mdd

print("\n" + "=" * 62)
print(f"=== LTR (30-grade) on BROAD universe: gross + net vs AUM ===")
print(f"{'AUM':>10}{'ann':>9}{'sharpe':>9}{'maxDD':>9}")
for aum in [0, 1e6, 1e7, 3e7, 1e8]:
    ann, shp, mdd = perf(net_returns(aum))
    print(f"{('gross' if aum == 0 else f'${aum/1e6:,.0f}M'):>10}{ann:>9.1%}{shp:>9.2f}{mdd:>9.1%}", flush=True)
print("[done]", flush=True)
