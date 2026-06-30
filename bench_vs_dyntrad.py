"""
bench_vs_dyntrad.py — Standalone comparison: Bench zMOM12 vs Bench+DynTrad.

Uses synthetic cross-sectional equity data with realistic momentum properties:
  - 500 stocks, 180 months (15 years)
  - Monthly prices driven by AR(1) expected-return factors (φ ≈ 0.65)
  - Cross-sectional dispersion calibrated to US equity universe

Compares:
  1. Bench zMOM12 L/S  — raw 12-month momentum, equal-weight top/bottom 5%
  2. Bench-DT L/S      — same signal, routed through DynTrad execution layer

Run:
    python bench_vs_dyntrad.py
"""

import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
import BACKTEST
from execution_layer import run_dyntrad, estimate_signal_decay

# ── Simulation parameters ──────────────────────────────────────────────────────
N_STOCKS   = 500
N_MONTHS   = 240        # 20 years total (120 warmup + 120 OOS)
SEED       = 42
PHI        = 0.65       # signal autocorrelation (measured from DM-RET signal)
ALPHA_VOL  = 0.005      # monthly cross-sectional signal vol (annualised ~1.7%)
NOISE_VOL  = 0.06       # monthly idiosyncratic return vol (~20% ann)
MKT_VOL    = 0.04       # monthly market return vol (~14% ann)
TC_BPS     = 10         # one-way transaction cost in bps
TOP_Q      = 0.05       # long/short tail: top/bottom 5%


def simulate_universe(n_stocks: int, n_months: int, seed: int) -> tuple:
    """
    Simulate a cross-sectional equity universe with AR(1) expected returns.

    Returns (prices_monthly, factor_scores)
    """
    rng = np.random.default_rng(seed)

    # AR(1) expected-return factor per stock: f_{t+1} = phi*f_t + eps
    eps_vol = ALPHA_VOL * np.sqrt(1 - PHI**2)   # stationary variance = ALPHA_VOL^2
    f = np.zeros((n_months + 1, n_stocks))
    f[0] = rng.normal(0, ALPHA_VOL, n_stocks)
    for t in range(n_months):
        f[t + 1] = PHI * f[t] + rng.normal(0, eps_vol, n_stocks)

    # Realised returns: r_{t+1} = f_t + market + idio
    mkt   = rng.normal(0.005, MKT_VOL, n_months)   # ~6% ann drift
    idio  = rng.normal(0, NOISE_VOL, (n_months, n_stocks))
    r     = f[:n_months] + mkt[:, None] + idio      # (n_months, n_stocks)

    # Build price series from returns
    tickers = [f"S{i:03d}" for i in range(n_stocks)]
    dates   = pd.date_range("2005-01", periods=n_months + 1, freq="ME")
    px      = pd.DataFrame(100 * np.cumprod(1 + np.vstack([np.zeros((1, n_stocks)), r]), axis=0),
                           index=dates, columns=tickers)
    return px


def momentum_signal(prices: pd.DataFrame, lookback: int = 12, skip: int = 1) -> pd.DataFrame:
    """
    Cross-sectional z-scored momentum: past [lookback] months return, skipping [skip] months.

    zMOM12 = z-score of (P_{t-skip} / P_{t-lookback-skip} - 1) at each date.
    """
    ret = prices.shift(skip) / prices.shift(lookback + skip) - 1
    # Cross-sectional z-score
    mu  = ret.mean(axis=1)
    sig = ret.std(axis=1).replace(0, np.nan)
    return ret.sub(mu, axis=0).div(sig, axis=0)


def build_weights(signal: pd.DataFrame, q: float = TOP_Q) -> pd.DataFrame:
    """Equal-weight top/bottom q long-short portfolio."""
    n = max(1, int(signal.shape[1] * q))
    rows = {}
    for date, row in signal.iterrows():
        valid = row.dropna()
        if len(valid) < 2 * n:
            continue
        w = pd.Series(0.0, index=signal.columns)
        w[valid.nlargest(n).index]  = +1.0 / n
        w[valid.nsmallest(n).index] = -1.0 / n
        rows[date] = w
    return pd.DataFrame(rows).T.fillna(0.0)


def print_metrics(name: str, res: dict):
    """Print a one-line summary of backtest results."""
    print(
        f"  {name:<25s}  "
        f"Ret={res['ann_return']:+.1%}  "
        f"Sharpe={res['sharpe']:.2f}  "
        f"MaxDD={res['max_drawdown']:.1%}  "
        f"Vol={res['ann_vol']:.1%}  "
        f"Turnover={res.get('ann_turnover', float('nan')):.0%}"
    )


# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Simulating cross-sectional equity universe …")
    print(f"  {N_STOCKS} stocks  |  {N_MONTHS} months  |  φ={PHI}  |  TC={TC_BPS}bps\n")

    px = simulate_universe(N_STOCKS, N_MONTHS, SEED)

    # Signal available from month 13 (need 12m lookback + 1m skip)
    signal = momentum_signal(px, lookback=12, skip=1)

    # Build raw benchmark weights — OOS starts at month 120
    oos_start_idx = 120
    oos_dates     = px.index[oos_start_idx:]
    signal_oos    = signal.loc[oos_dates[:-1]]   # last date has no forward return
    bench_w       = build_weights(signal_oos)

    # ── DynTrad on top of bench signal ────────────────────────────────────────
    phi_est = estimate_signal_decay(bench_w).mean()
    bench_dt, dt_params = run_dyntrad(
        bench_w,
        signal_decay=phi_est,
        cost_multiplier=2.0,
        gross_exposure=2.0,
    )
    print(f"[DynTrad]  φ_estimated={phi_est:.3f}  δ={dt_params['trading_fraction_delta']:.3f}"
          f"  aim_weight={dt_params['aim_weights'][0]:.3f}\n")

    # Prices for PnL (OOS window, matching signal dates)
    px_oos   = px.loc[oos_dates]
    tc       = TC_BPS / 1e4

    results = {}
    for label, w in [("Bench zMOM12 L/S", bench_w), ("Bench-DT L/S", bench_dt)]:
        sigs = [d for d in w.index if d in px_oos.index]
        w_bt = w.reindex(columns=px_oos.columns).fillna(0.0)
        res  = BACKTEST.backtest(
            w_bt, px_oos,
            freq=12, lag=0,
            transaction_cost=tc,
            signal_dates=sigs,
        )
        res["name"] = label
        results[label] = res

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 70)
    print(f"OOS window: {oos_dates[0].date()} – {oos_dates[-1].date()}  "
          f"({len(bench_w)} signal months)")
    print("=" * 70)
    for label, res in results.items():
        print_metrics(label, res)
    print("=" * 70)

    # ── Turnover detail ───────────────────────────────────────────────────────
    to_bench = bench_w.diff().abs().sum(axis=1).mean() * 12
    to_dt    = bench_dt.diff().abs().sum(axis=1).mean() * 12
    print(f"\nTurnover (ann):  Bench={to_bench:.0%}  Bench-DT={to_dt:.0%}"
          f"  reduction={(1-to_dt/to_bench)*100:.0f}%")

    # ── Delta sensitivity ─────────────────────────────────────────────────────
    print("\nDelta sensitivity (cost_multiplier sweep):")
    print(f"  {'λ':>6}  {'δ':>6}  {'Sharpe':>8}  {'Return':>8}  {'Turnover':>10}")
    for lam in [0.5, 1.0, 2.0, 5.0, 10.0]:
        w_sweep, p_sweep = run_dyntrad(bench_w, signal_decay=phi_est,
                                       cost_multiplier=lam, gross_exposure=2.0)
        sigs = [d for d in w_sweep.index if d in px_oos.index]
        w_bt = w_sweep.reindex(columns=px_oos.columns).fillna(0.0)
        r    = BACKTEST.backtest(w_bt, px_oos, freq=12, lag=0,
                                 transaction_cost=tc, signal_dates=sigs)
        to   = w_sweep.diff().abs().sum(axis=1).mean() * 12
        print(f"  {lam:>6.1f}  {p_sweep['trading_fraction_delta']:>6.3f}"
              f"  {r['sharpe']:>8.2f}  {r['ann_return']:>+8.1%}  {to:>10.0%}")
