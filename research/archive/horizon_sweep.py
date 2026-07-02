#!/usr/bin/env python3
"""Horizon sweep: how do standalone t+h forecasts (h=1,2,3,6) and stacked combos behave?
Tests whether longer horizons get progressively more momentum-like (crash-prone) or decay to noise."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBClassifier
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

N_SEEDS, Q, MIN_TRAIN, MAX_TRAIN = 5, 0.05, 120, 120
HORIZONS = (1, 2, 3, 6)
H_MAX = max(HORIZONS)
pd.set_option("display.width", 200)

print("[load] tiingo ...", flush=True)
prices_monthly, rets_monthly, size_monthly = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
T = len(rets_monthly)
eligible, shortable = d.compute_eligibility(prices_monthly, size_monthly, min_dollar_vol_abs=5e6)
pnl_prices = d._build_pnl_prices(prices_monthly)
tcost = BACKTEST.tiered_transaction_costs(size_monthly)
bfee  = BACKTEST.tiered_borrow_fees(size_monthly)
first_feat = max(MOM_WINDOWS) + 1
first_pred = MIN_TRAIN + first_feat
print("[ffd] optimal-d search ...", flush=True)
ffd = d._ffd_from_training_window(prices_monthly, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(rets_monthly.index)

pool = {}
for t in range(first_feat, T - H_MAX):
    F = make_features(rets_monthly, size_monthly, t, ffd_scores=ffd).dropna()
    e = eligible.iloc[t - 1]; idx = F.index.intersection(e.index[e.values])
    if len(idx) < 20: continue
    labels, ok = {}, True
    for h in HORIZONS:
        tgt = rets_monthly.iloc[t + h - 1].reindex(idx).dropna()
        if len(tgt) < 20: ok = False; break
        labels[h] = d._decile_labels(tgt)
    if ok: pool[t] = (F.loc[idx], labels)

# scores[h] : dict signal_date -> pd.Series of expected-return score for horizon h
scores = {h: {} for h in HORIZONS}
feats  = {}   # signal_date -> (features, shortable index)
model_store = {}
pred_years = sorted({rets_monthly.index[t].year for t in range(first_pred, T - H_MAX)})
for year in pred_years:
    months = [t for t in range(first_pred, T - H_MAX) if rets_monthly.index[t].year == year]
    if not months: continue
    all_ts = sorted([t for t in pool if t < months[0]])
    if MAX_TRAIN and len(all_ts) > MAX_TRAIN: all_ts = all_ts[-MAX_TRAIN:]
    n_val = max(6, int(len(all_ts) * 0.2))
    if len(all_ts) >= 18:
        tr, va = all_ts[:len(all_ts) - n_val], all_ts[len(all_ts) - n_val:]
        ens = {}
        for h in HORIZONS:
            tgt_all = pd.concat([rets_monthly.iloc[t + h - 1].reindex(pool[t][1][h].index) for t in all_ts])
            lab_all = pd.concat([pool[t][1][h] for t in all_ts])
            grp = pd.DataFrame({"l": lab_all.values, "r": tgt_all.values}).groupby("l")["r"]
            mu_k = np.array([grp.get_group(k).mean() if k in grp.groups else 0.0 for k in range(10)])
            Xtr = pd.concat([pool[t][0].reindex(pool[t][1][h].index) for t in tr]); ytr = pd.concat([pool[t][1][h] for t in tr])
            Xva = pd.concat([pool[t][0].reindex(pool[t][1][h].index) for t in va]); yva = pd.concat([pool[t][1][h] for t in va])
            ms = [XGBClassifier(**d.XGB_PARAMS, random_state=s).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False) for s in range(N_SEEDS)]
            ens[h] = (ms, mu_k)
        model_store[year] = ens
        print(f"  [{year}] trained", flush=True)
    elif model_store:
        model_store[year] = list(model_store.values())[-1]
    else:
        continue
    ens = model_store[year]
    for t in months:
        if t not in pool: continue
        F, _ = pool[t]; sd = rets_monthly.index[t - 1]
        feats[sd] = (F, shortable.iloc[t - 1])
        for h in HORIZONS:
            p = np.mean([m.predict_proba(F) for m in ens[h][0]], axis=0)
            scores[h][sd] = pd.Series(d.score_ret(p, ens[h][1]), index=F.index)

def weights_from_score(score_by_date):
    rows = {}
    for sd, sc in score_by_date.items():
        F, sh = feats[sd]; shidx = sh.index[sh.values]
        n = max(1, int(len(sc) * Q))
        w = pd.Series(0.0, index=F.index); w[sc.nlargest(n).index] = 1.0 / n
        ssub = sc.loc[sc.index.intersection(shidx)]; w[ssub.nsmallest(n).index] = -1.0 / n
        rows[sd] = w
    return pd.DataFrame(rows).T.fillna(0.0).sort_index()

def combo(hs):
    dates = scores[hs[0]].keys()
    return {sd: sum(scores[h][sd] for h in hs) for sd in dates}

variants = {f"h{h}": scores[h] for h in HORIZONS}
variants["MH12"]   = combo((1, 2))
variants["MH123"]  = combo((1, 2, 3))
variants["MH1236"] = combo((1, 2, 3, 6))
W = {k: weights_from_score(v) for k, v in variants.items()}

bench_w = d._bench_weights(rets_monthly, size_monthly, prices_monthly, min_train_months=MIN_TRAIN,
                           q=Q, use_ffd=False, portfolio="ls", eligible=eligible, shortable=shortable)

def run(weights):
    first = weights.index[0]; px = pnl_prices.loc[first:]
    sigs = [x for x in weights.index if x in px.index]
    return BACKTEST.backtest(weights.reindex(columns=px.columns).fillna(0.0), px,
                             freq=12, lag=0, transaction_cost=tcost, borrow_fee=bfee, signal_dates=sigs)

res = {k: run(w) for k, w in W.items()}
res["bench"] = run(bench_w)
rr = pd.DataFrame({k: pd.Series(res[k]["returns"]) for k in list(W) + ["bench"]}).dropna()
corr = rr.corr()["bench"]

# holdings momentum-tilt per standalone horizon
tilt = {}
for h in HORIZONS:
    longs, shorts = [], []
    for sd, sc in scores[h].items():
        F, sh = feats[sd]; shidx = sh.index[sh.values]; n = max(1, int(len(sc) * Q))
        Ln = list(sc.nlargest(n).index)
        Sn = list(sc.loc[sc.index.intersection(shidx)].nsmallest(n).index)
        longs.append(F.loc[Ln, "zMOM12"].mean()); shorts.append(F.loc[Sn, "zMOM12"].mean())
    tilt[h] = (np.nanmean(longs), np.nanmean(shorts))

def L(name, r, c):
    print(f"{name:8} ann={r['ann_return']:7.2%}  sharpe={r['sharpe']:5.2f}  sortino={r['sortino_ratio']:5.2f}  "
          f"maxDD={r['max_drawdown']:7.2%}  vol={r['ann_vol']:6.2%}  corr→bench={c:5.2f}")

print("\n" + "=" * 86 + "\n=== STANDALONE HORIZON (h-only L/S, q=0.05, 5 seeds) ===")
for h in HORIZONS: L(f"h{h}", res[f"h{h}"], corr[f"h{h}"])
print("\n=== STACKED COMBOS ===")
for k in ("MH12", "MH123", "MH1236"): L(k, res[k], corr[k])
L("bench", res["bench"], 1.0)
print("\n=== MOMENTUM TILT OF HOLDINGS (avg zMOM12) ===")
print(f"{'horizon':8}{'long':>8}{'short':>8}")
for h in HORIZONS: print(f"h{h:<7}{tilt[h][0]:>8.2f}{tilt[h][1]:>8.2f}")
print("\n[done]", flush=True)
