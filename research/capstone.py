#!/usr/bin/env python3
"""CAPSTONE: combine the two surviving, orthogonal models —
   Model 1 = DM (stock-level, nonlinear, monthly; harvests reversal as a short-book hedge)
   Model 2 = Factor/common momentum (factor/industry level, intermediate horizon; persistent offense)
Run both through the cost engine, align return streams, and blend at equal risk. Test whether the
diversified pair beats either sleeve alone (the whole thesis)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from numpy.linalg import eigh, lstsq
import deep_momentum_xgb as d
from features import MOM_WINDOWS
import BACKTEST

Q, MIN_TRAIN, MAX_TRAIN = 0.05, 120, 120
print(f"[load] n_seeds={d.N_SEEDS} q={Q}", flush=True)
prices_monthly, rets_monthly, size_monthly = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
T = len(rets_monthly); dates = rets_monthly.index
eligible, shortable = d.compute_eligibility(prices_monthly, size_monthly, min_dollar_vol_abs=5e6)
pnl_prices = d._build_pnl_prices(prices_monthly)
tcost = BACKTEST.tiered_transaction_costs(size_monthly)
bfee  = BACKTEST.tiered_borrow_fees(size_monthly)

first_feat = max(MOM_WINDOWS) + 1
first_pred = MIN_TRAIN + first_feat
print("[ffd] optimal-d ...", flush=True)
ffd = d._ffd_from_training_window(prices_monthly, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(rets_monthly.index)

# ── Model 1: DM-RET (q=0.05, eligibility + borrowability) ──
print("[DM] generating weights ...", flush=True)
dm_w = d._generate_all_dm_weights(rets_monthly, size_monthly, prices_monthly,
        min_train_months=MIN_TRAIN, max_train_months=MAX_TRAIN, q=Q, n_seeds=d.N_SEEDS,
        use_ffd=True, ffd_scores=ffd, portfolio="ls", eligible=eligible, shortable=shortable)["ret"]

# ── Model 2: factor/common momentum (asymptotic-PCA, common component, same q & masks) ──
print("[FactorMom] generating weights ...", flush=True)
K, WIN = 5, 12
fm_rows = {}
for t in range(first_pred, T):                         # align start to DM window
    win = rets_monthly.iloc[t - WIN:t]
    e = eligible.iloc[t - 1]
    cols = win.columns[win.notna().all().values & e.reindex(win.columns).fillna(False).values]
    if len(cols) < 50: continue
    Rw = win[cols].to_numpy(); n = Rw.shape[1]
    _, V = eigh(Rw @ Rw.T / n); F = V[:, ::-1][:, :K]
    B, *_ = lstsq(F, Rw, rcond=None)
    common = (F @ B)[:WIN - 1].sum(0)                  # common 12-1 momentum
    s = pd.Series(common, index=cols)
    sd = dates[t - 1]; sh = shortable.iloc[t - 1]; shset = set(sh.index[sh.values])
    nq = max(1, int(n * Q))
    w = pd.Series(0.0, index=cols); w[s.nlargest(nq).index] = 1.0 / nq
    ssub = s.loc[[c for c in s.index if c in shset]]; w[ssub.nsmallest(nq).index] = -1.0 / nq
    fm_rows[sd] = w
fm_w = pd.DataFrame(fm_rows).T.fillna(0.0).sort_index()
print(f"[FactorMom] {len(fm_w)} signal months", flush=True)

def run(weights):
    first = weights.index[0]; px = pnl_prices.loc[first:]
    sigs = [x for x in weights.index if x in px.index]
    return BACKTEST.backtest(weights.reindex(columns=px.columns).fillna(0.0), px,
                             freq=12, lag=0, transaction_cost=tcost, borrow_fee=bfee, signal_dates=sigs)

dm_res, fm_res = run(dm_w), run(fm_w)
r = pd.DataFrame({"DM": pd.Series(dm_res["returns"]), "FactorMom": pd.Series(fm_res["returns"])}).dropna()

# equal-risk (inverse-vol) blend, rescaled to ~DM vol for comparability
iv = 1.0 / r.std()
wts = iv / iv.sum()
blend = (r * wts).sum(axis=1)
blend = blend * (r["DM"].std() / blend.std())          # vol-match to DM for fair Sharpe read

def perf(x):
    x = x.dropna()
    ann = (1 + x).prod() ** (12 / len(x)) - 1
    vol = x.std() * np.sqrt(12); dn = x[x < 0].std() * np.sqrt(12)
    eq = (1 + x).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return ann, (x.mean() * 12) / vol, (x.mean() * 12) / dn, dd

print("\n" + "=" * 80 + f"\n=== CAPSTONE: DM + Factor Momentum  (q={Q}, {r.index[0].date()}–{r.index[-1].date()}) ===")
print(f"{'strategy':22}{'ann':>8}{'sharpe':>8}{'sortino':>8}{'maxDD':>9}")
for nm, x in [("DM alone", r["DM"]), ("FactorMom alone", r["FactorMom"]),
              ("BLEND (equal-risk)", blend)]:
    a, sh, so, dd = perf(x)
    print(f"{nm:22}{a:>8.1%}{sh:>8.2f}{so:>8.2f}{dd:>9.1%}")
print(f"\ncorr(DM, FactorMom) = {r.corr().iloc[0,1]:.3f}   blend weights: DM={wts['DM']:.2f} FM={wts['FactorMom']:.2f}")
print("\n[done]", flush=True)
