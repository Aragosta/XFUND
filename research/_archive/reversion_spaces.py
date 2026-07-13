#!/usr/bin/env python3
"""reversion_spaces.py — test the 'modern spirit of stat-arb' claim: the ALPHA is in the SIGNAL SPACE, and
returns-reversion is the worst (fastest, most crowded, highest turnover). Bet on cross-sectional displacement
from the basket centroid in DIFFERENT spaces, dollar-neutral, and compare NET Sharpe AND turnover.

Each 'space' = a per-name metric M. Anti-displacement (reversion) book: long the low-M decile, short the
high-M decile, dollar-neutral, daily. Evaluated through BACKTEST.py (lag=1 real execution, tiered trade+borrow
costs). Hypothesis: slower spaces (dollar-volume autocorrelation, illiquidity) turn over far less than
returns-reversion and may net more. Sign convention is 'revert toward centroid'; a negative net SR just means
that space pays in the opposite direction (report both magnitude and turnover — turnover is the article's point).

VERDICT: EXPLORATORY / CUT-adjacent (see [[reversion-catalog-synthesis]]). Confirmed returns-reversion is the
worst space (net -0.58, turn 74x); only net-positive slow space = Amihud illiquidity +0.32 (turn 5.2x). Its
'winner' vol-autocorr did NOT replicate (toy basket). No baseline arm, seeds=1 — DIRECTIONAL only, not a sleeve."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import BACKTEST

MINPX, MINDV, TOPN = 5.0, 5e6, 1000

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
test = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"])
px = px.loc[:, ~test]                                                       # drop NASDAQ test tickers (known leak)
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
dvd = px * vb; dv = dvd.rolling(63, min_periods=40).mean()
elig = (px > MINPX) & (dv > MINDV)
rankdv = dv.where(elig).rank(axis=1, ascending=False); elig = (elig & (rankdv <= TOPN)).fillna(False)
dret = px.pct_change(fill_method=None).where(lambda z: z.abs() < 1.0)
synth = (1 + dret.clip(-0.5, 0.5).fillna(0.0)).cumprod()                    # cleaned price grid for the engine
mdvq = dv * 21.0
TC = BACKTEST.tiered_transaction_costs(mdvq); BF = BACKTEST.tiered_borrow_fees(mdvq)

# --- the signal spaces (per-name metrics) ---
SPACES = {
    "returns (3d)":   px / px.shift(3) - 1,                                 # fast, crowded reversion baseline
    "vol-autocorr":   dvd.rolling(20, min_periods=15).corr(dvd.shift(1)),   # article's winner: $-volume persistence
    "amihud illiq":   (dret.abs() / dvd.replace(0, np.nan)).rolling(63, min_periods=40).mean(),
    "vol-trend":      np.log(dvd.rolling(21, min_periods=15).mean() / dvd.rolling(252, min_periods=150).mean()),
    "abn-volume":     dvd.rolling(3, min_periods=3).mean() / dvd.rolling(63, min_periods=40).mean().shift(3),
}

oos = px.index[px.index.year >= 2011]
def book_weights(M):
    """daily decile anti-displacement book: long low-M / short high-M, dollar-neutral."""
    Me = M.where(elig)
    r = Me.rank(axis=1, pct=True)
    lo = (r <= 0.10); hi = (r >= 0.90)
    nlo = lo.sum(axis=1).replace(0, np.nan); nhi = hi.sum(axis=1).replace(0, np.nan)
    W = lo.div(nlo, axis=0) * 0.5 - hi.div(nhi, axis=0) * 0.5
    return W.reindex(columns=synth.columns).fillna(0.0)

print("=" * 74)
print(f"=== Stat-arb SIGNAL SPACES (top-{TOPN}, daily, lag=1, tiered) — 'modern spirit' test ===")
print(f"{'space':16}{'grossSR':>9}{'netSR':>8}{'netAnn':>8}{'netDD':>8}{'turn(ann)':>11}")
sig = list(oos)
for nm, M in SPACES.items():
    W = book_weights(M)
    g = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig)
    n = BACKTEST.backtest(W, synth, freq=252, lag=1, signal_dates=sig, transaction_cost=TC, borrow_fee=BF)
    print(f"{nm:16}{g['sharpe']:>9.2f}{n['sharpe']:>8.2f}{n['ann_return']:>8.1%}{n['max_drawdown']:>8.1%}{n['ann_turnover']:>11.1f}", flush=True)
print("[done]", flush=True)
