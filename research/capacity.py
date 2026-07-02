#!/usr/bin/env python3
"""Capacity / market-impact analysis for DM — the 'can we actually trade this' number.

Replaces the crude tiered cost with a square-root (Almgren) market-impact model whose cost
depends on TRADE SIZE, so we can sweep AUM and see where DM's net Sharpe degrades:

  cost_frac(i) = half_spread(i)  +  ETA * sigma_m(i) * sqrt( |dW_i| * AUM / DV_month(i) )
                 (fixed spread)      (size-dependent impact; participation = trade$ / monthly $vol)

half_spread from the existing tiered model (proxy); sigma_m = trailing 12m monthly-return vol;
DV_month = trailing 3m dollar volume. Borrow fees charged on the short book. Tight ~561 universe.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import deep_momentum_xgb as d
from features import MOM_WINDOWS
import BACKTEST

Q, MIN_TRAIN, MAX_TRAIN, ETA = 0.05, 120, 120, 0.5

print("[load] universe ...", flush=True)
pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6, short_min_dollar_vol_abs=25e6)
pnl = d._build_pnl_prices(pm)
pnl_ret = pnl.pct_change()
dv3   = sm.rolling(3, min_periods=1).mean()                 # monthly $ volume (trailing 3m)
sig_m = rm.rolling(12, min_periods=6).std()                 # monthly return vol (trailing)
half_spread = BACKTEST.tiered_transaction_costs(sm)         # per-name fixed spread proxy (one-way)
bfee = BACKTEST.tiered_borrow_fees(sm)

first_pred = MIN_TRAIN + max(MOM_WINDOWS) + 1
print("[ffd] optimal-d ...", flush=True)
ffd = d._ffd_from_training_window(pm, first_pred)
for m in list(ffd): ffd[m] = ffd[m].reindex(rm.index)

print("[DM] generating weights ...", flush=True)
W = d._generate_all_dm_weights(rm, sm, pm, min_train_months=MIN_TRAIN, max_train_months=MAX_TRAIN,
        q=Q, n_seeds=d.N_SEEDS, use_ffd=True, ffd_scores=ffd, portfolio="ls",
        eligible=elig, shortable=short)["ret"]
sd = list(W.index)
print(f"[weights] {len(sd)} rebalances", flush=True)

def net_returns(aum):
    """Monthly net returns of DM at a given AUM using the square-root impact model."""
    out, prev = [], pd.Series(0.0, index=W.columns)
    for d0 in sd:
        w = W.loc[d0]
        nxt = pnl_ret.index[pnl_ret.index > d0]
        if len(nxt) == 0: break
        hold = pnl_ret.loc[nxt[0]]
        gross = float((w * hold).sum(skipna=True))
        dw = (w - prev).abs()
        traded = dw[dw > 0]
        if len(traded):
            sp  = half_spread.loc[d0].reindex(traded.index).fillna(0.015)
            sg  = sig_m.loc[d0].reindex(traded.index).fillna(0.15)
            dvm = dv3.loc[d0].reindex(traded.index).replace(0, np.nan).fillna(1e5)
            impact = ETA * sg * np.sqrt((traded * aum / dvm).clip(lower=0))
            cost = float(((sp + impact) * traded).sum())
        else:
            cost = 0.0
        brw = float((w.clip(upper=0).abs() * bfee.loc[d0].reindex(w.index).fillna(0.05)).sum()) / 12.0
        out.append(gross - cost - brw)
        prev = w
    return pd.Series(out)

def perf(r):
    r = pd.Series(r).dropna()
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol if vol > 0 else np.nan, mdd

print("\n" + "=" * 62)
print(f"=== DM net-of-impact CAPACITY (square-root model, ETA={ETA}) ===")
print(f"{'AUM':>10}{'ann':>9}{'sharpe':>9}{'maxDD':>9}")
for aum in [0, 1e6, 1e7, 3e7, 1e8, 3e8, 1e9, 3e9]:
    ann, shp, mdd = perf(net_returns(aum))
    label = "gross" if aum == 0 else f"${aum/1e6:,.0f}M"
    print(f"{label:>10}{ann:>9.1%}{shp:>9.2f}{mdd:>9.1%}", flush=True)
print("[done]", flush=True)
