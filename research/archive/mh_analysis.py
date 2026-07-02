#!/usr/bin/env python3
"""Deep analysis: why does DM-MH (multi-horizon t+1&t+2) draw down so much more than base DM?

Decomposes the multi-horizon signal into its h1 (t+1) and h2 (t+2) components, trained in a
SINGLE pass, and tests the hypothesis that the t+2 component re-couples the strategy to raw
momentum (and thus to momentum-crash risk that base DM's t+1 reclassification avoids).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from xgboost import XGBClassifier
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

N_SEEDS, Q, MIN_TRAIN, MAX_TRAIN = 5, 0.05, 120, 120
pd.set_option("display.width", 200, "display.max_columns", 30)

print("[load] tiingo from checkpoint ...", flush=True)
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

# ── pool with both-horizon labels ────────────────────────────────────────────
horizons, h_max = (1, 2), 2
pool = {}
for t in range(first_feat, T - h_max):
    F = make_features(rets_monthly, size_monthly, t, ffd_scores=ffd).dropna()
    e = eligible.iloc[t - 1]; idx = F.index.intersection(e.index[e.values])
    if len(idx) < 20: continue
    labels, ok = {}, True
    for h in horizons:
        tgt = rets_monthly.iloc[t + h - 1].reindex(idx).dropna()
        if len(tgt) < 20: ok = False; break
        labels[h] = d._decile_labels(tgt)
    if ok: pool[t] = (F.loc[idx], labels)

# ── train per-year, emit h1 / h2 / mh weights + per-month diagnostics ─────────
rows = {"h1": {}, "h2": {}, "mh": {}}
diag = []
model_store = {}
pred_years = sorted({rets_monthly.index[t].year for t in range(first_pred, T - h_max)})
for year in pred_years:
    months = [t for t in range(first_pred, T - h_max) if rets_monthly.index[t].year == year]
    if not months: continue
    all_ts = sorted([t for t in pool if t < months[0]])
    if MAX_TRAIN and len(all_ts) > MAX_TRAIN: all_ts = all_ts[-MAX_TRAIN:]
    n_val = max(6, int(len(all_ts) * 0.2))
    if len(all_ts) >= 18:
        tr, va = all_ts[:len(all_ts) - n_val], all_ts[len(all_ts) - n_val:]
        ens = {}
        for h in horizons:
            tgt_all = pd.concat([rets_monthly.iloc[t + h - 1].reindex(pool[t][1][h].index) for t in all_ts])
            lab_all = pd.concat([pool[t][1][h] for t in all_ts])
            grp = pd.DataFrame({"l": lab_all.values, "r": tgt_all.values}).groupby("l")["r"]
            mu_k = np.array([grp.get_group(k).mean() if k in grp.groups else 0.0 for k in range(10)])
            Xtr = pd.concat([pool[t][0].reindex(pool[t][1][h].index) for t in tr]); ytr = pd.concat([pool[t][1][h] for t in tr])
            Xva = pd.concat([pool[t][0].reindex(pool[t][1][h].index) for t in va]); yva = pd.concat([pool[t][1][h] for t in va])
            ms = [XGBClassifier(**d.XGB_PARAMS, random_state=s).fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False) for s in range(N_SEEDS)]
            ens[h] = (ms, mu_k)
        model_store[year] = ens
        print(f"  [{year}] trained pool={len(all_ts)}", flush=True)
    elif model_store:
        model_store[year] = list(model_store.values())[-1]
    else:
        continue
    ens = model_store[year]
    for t in months:
        if t not in pool: continue
        F, _ = pool[t]; sd = rets_monthly.index[t - 1]
        p1 = np.mean([m.predict_proba(F) for m in ens[1][0]], axis=0)
        p2 = np.mean([m.predict_proba(F) for m in ens[2][0]], axis=0)
        s1 = pd.Series(d.score_ret(p1, ens[1][1]), index=F.index)
        s2 = pd.Series(d.score_ret(p2, ens[2][1]), index=F.index)
        smh = s1 + s2
        sh = shortable.iloc[t - 1]; shidx = sh.index[sh.values]
        n = max(1, int(len(F) * Q))
        def mkw(sc):
            w = pd.Series(0.0, index=F.index); w[sc.nlargest(n).index] = 1.0 / n
            ssub = sc.loc[sc.index.intersection(shidx)]; w[ssub.nsmallest(n).index] = -1.0 / n
            return w
        rows["h1"][sd], rows["h2"][sd], rows["mh"][sd] = mkw(s1), mkw(s2), mkw(smh)
        L1, Lm = set(s1.nlargest(n).index), set(smh.nlargest(n).index)
        S1 = set(s1.loc[s1.index.intersection(shidx)].nsmallest(n).index)
        Sm = set(smh.loc[smh.index.intersection(shidx)].nsmallest(n).index)
        feat = pool[t][0]
        def av(names, col):
            names = [x for x in names if x in feat.index]
            return float(feat.loc[names, col].mean()) if names else np.nan
        diag.append(dict(date=sd,
            corr_s1s2=float(np.corrcoef(s1.values, s2.values)[0, 1]),
            long_overlap=len(L1 & Lm) / n, short_overlap=len(S1 & Sm) / n,
            h1L_zMOM12=av(L1, "zMOM12"), mhL_zMOM12=av(Lm, "zMOM12"),
            h1L_zVOL=av(L1, "zVOL"),     mhL_zVOL=av(Lm, "zVOL"),
            h1S_zMOM12=av(S1, "zMOM12"), mhS_zMOM12=av(Sm, "zMOM12"),
            mhOnlyL_zVOL=av(Lm - L1, "zVOL"), mhOnlyL_zMOM12=av(Lm - L1, "zMOM12")))

W = {k: pd.DataFrame(v).T.fillna(0.0).sort_index() for k, v in rows.items()}
diag = pd.DataFrame(diag).set_index("date")

# bench (raw zMOM12) for coupling comparison — cheap, no XGB
bench_w = d._bench_weights(rets_monthly, size_monthly, prices_monthly, min_train_months=MIN_TRAIN,
                           q=Q, use_ffd=False, portfolio="ls", eligible=eligible, shortable=shortable)

def run(weights, tc=tcost):
    first = weights.index[0]; px = pnl_prices.loc[first:]
    sigs = [x for x in weights.index if x in px.index]
    return BACKTEST.backtest(weights.reindex(columns=px.columns).fillna(0.0), px,
                             freq=12, lag=0, transaction_cost=tc, borrow_fee=bfee, signal_dates=sigs)

res = {k: run(W[k]) for k in ("h1", "h2", "mh")}
res["bench"] = run(bench_w)
legs = {}
for k in ("h1", "mh"):
    legs[k + "_long"] = run(W[k].clip(lower=0))
    legs[k + "_short"] = run(W[k].clip(upper=0))

def L(name, r):
    print(f"{name:16} ann={r['ann_return']:7.2%}  sharpe={r['sharpe']:5.2f}  sortino={r['sortino_ratio']:5.2f}  "
          f"maxDD={r['max_drawdown']:7.2%}  vol={r['ann_vol']:6.2%}  turn={r['ann_turnover']:7.1%}")

print("\n" + "=" * 78 + "\n=== STRATEGY SUMMARY (n_seeds=5, q=0.05) ===")
for k, nm in [("bench", "Bench zMOM12"), ("h1", "DM (t+1 only)"), ("h2", "H2-only (t+2)"), ("mh", "DM-MH (t1+t2)")]:
    L(nm, res[k])
print("\n=== LEG DECOMPOSITION (long-only / short-only sub-portfolios) ===")
for k in ("h1_long", "h1_short", "mh_long", "mh_short"):
    L(k, legs[k])

# coupling to raw momentum
rr = pd.DataFrame({k: pd.Series(res[k]["returns"]) for k in ("bench", "h1", "h2", "mh")}).dropna()
print("\n=== MONTHLY-RETURN CORRELATION TO BENCH (raw zMOM12) ===")
print(rr.corr()["bench"].round(3).to_string())

# drawdown windows
print("\n=== DRAWDOWN WINDOW (peak→trough) ===")
for k in ("bench", "h1", "mh"):
    eq = pd.Series(res[k]["equity"]); dd = pd.Series(res[k]["drawdown"])
    trough = dd.idxmin(); peak = eq.loc[:trough].idxmax()
    print(f"{k:6}  maxDD={dd.min():7.2%}  peak={pd.Timestamp(peak).date()}  trough={pd.Timestamp(trough).date()}")

# worst 8 months
print("\n=== WORST 8 MONTHS (each strategy) ===")
for k in ("h1", "mh"):
    s = pd.Series(res[k]["returns"]).sort_values().head(8)
    print(f"-- {k} --"); print("  " + "  ".join(f"{pd.Timestamp(i).date()}:{v:+.1%}" for i, v in s.items()))

# signal diagnostics
print("\n=== SIGNAL / HOLDINGS DIAGNOSTICS (time-averaged) ===")
print(diag.mean(numeric_only=True).round(3).to_string())

# crash-month behaviour: worst-10 MH months vs the rest
worst = pd.Series(res["mh"]["returns"]).sort_values().head(10).index
worst = [pd.Timestamp(x) for x in worst]
dd_idx = diag.index.map(lambda x: pd.Timestamp(x))
mask = pd.Series([x in set(worst) for x in dd_idx], index=diag.index)
print("\n=== DIAGNOSTICS: MH worst-10 months vs rest ===")
print(pd.DataFrame({"worst10": diag[mask].mean(numeric_only=True),
                    "rest": diag[~mask].mean(numeric_only=True)}).round(3).to_string())

# how much of MH's long/short book differs from h1, and bench-overlap of MH short leg
print("\n=== AVG TAIL OVERLAP MH-vs-h1 ===")
print(f"long overlap  : {diag['long_overlap'].mean():.1%}")
print(f"short overlap : {diag['short_overlap'].mean():.1%}")
print("\n[done]", flush=True)
