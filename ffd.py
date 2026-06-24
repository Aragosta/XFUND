#!/usr/bin/env python3
"""
ffd.py  —  Fixed-Width-Window Fractional Differentiation
           López de Prado, "Advances in Financial Machine Learning", Ch. 5

Core idea
---------
Standard returns (d = 1) achieve stationarity but throw away all long-run
memory.  FFD with the *minimum* d that passes an ADF test preserves as much
memory as possible while removing the unit root.

Public API
----------
  ffd_weights(d, thres)                   → np.ndarray  (FIR coefficients)
  ffd_apply(prices_df, d, thres)          → pd.DataFrame (FFD of log-prices)
  find_optimal_d(series, ...)             → float
  find_optimal_d_batch(prices_df, ...)    → pd.Series   (optimal d per ticker)
  build_ffd_scores(prices_monthly,        → dict {window → pd.DataFrame}
                   d_series, windows, thres)

Quick-start
-----------
  from ffd import find_optimal_d_batch, build_ffd_scores
  d_per_ticker   = find_optimal_d_batch(prices_monthly)
  ffd_scores     = build_ffd_scores(prices_monthly, d_per_ticker)
  # ffd_scores[12] is a (T × N) DataFrame of FFD-12 momentum scores
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from statsmodels.tsa.stattools import adfuller

try:
    from joblib import Parallel, delayed
    _HAS_JOBLIB = True
except ImportError:
    _HAS_JOBLIB = False


# ── 1. Weights ─────────────────────────────────────────────────────────────────

def ffd_weights(d: float, thres: float = 1e-4, max_lags: int = 60) -> np.ndarray:
    """
    FIR coefficients for fractional differentiation of order d.

    Recurrence: w[0] = 1,  w[k] = -w[k-1] * (d - k + 1) / k

    Truncation: weights are dropped once |w_k| < thres OR k >= max_lags,
    whichever comes first.  For monthly data the default max_lags=60 (5 yr)
    is a practical upper bound; for daily data raise it to 500+.

    Returns w in *convolution order*: w[0] applies to x[t] (most recent),
    w[-1] applies to x[t - L + 1] (oldest).  Correct order for
    scipy.signal.lfilter(w, 1.0, x).
    """
    if not (0.0 <= d <= 2.0):
        raise ValueError(f"d must be in [0, 2]; got {d}")
    w, k = [1.0], 1
    while k < max_lags:
        w_next = -w[-1] * (d - k + 1) / k
        if abs(w_next) < thres:
            break
        w.append(w_next)
        k += 1
    return np.array(w, dtype=np.float64)


# ── 2. Apply FFD ───────────────────────────────────────────────────────────────

def ffd_apply(
    prices_df: pd.DataFrame | pd.Series,
    d: float | None = None,
    d_per_col: pd.Series | None = None,
    thres: float = 1e-5,
) -> pd.DataFrame:
    """
    Apply fixed-width-window fractional differentiation to a price DataFrame.

    Uses log-prices internally (MLDP recommendation) so the output is a
    stationary fractionally-differenced log-price series.

    Parameters
    ----------
    prices_df  : T × N prices (or a single Series).  NaN-forward-filled before
                 log transform, then any leading NaN columns stay NaN.
    d          : scalar d applied to all columns (ignored if d_per_col given).
    d_per_col  : pd.Series indexed by column name, one d per ticker.
    thres      : weight truncation threshold (lower → longer filter, more memory).

    Returns
    -------
    pd.DataFrame same shape as prices_df.  The first L-1 rows of each column
    are NaN (L = len(ffd_weights(d, thres))).
    """
    if isinstance(prices_df, pd.Series):
        prices_df = prices_df.to_frame()
        squeeze = True
    else:
        squeeze = False

    if d_per_col is None and d is None:
        raise ValueError("Provide either `d` or `d_per_col`.")

    out = pd.DataFrame(np.nan, index=prices_df.index, columns=prices_df.columns)

    for col in prices_df.columns:
        col_d = float(d_per_col[col]) if d_per_col is not None else float(d)
        series = prices_df[col].ffill()
        valid  = series.notna()
        if valid.sum() < 2:
            continue

        log_px = np.log(series[valid].values.astype(np.float64))
        w      = ffd_weights(col_d, thres)
        L      = len(w)

        # lfilter(b, a, x): y[n] = b[0]*x[n] + b[1]*x[n-1] + … + b[L-1]*x[n-L+1]
        filtered = lfilter(w, 1.0, log_px)
        filtered[:L - 1] = np.nan   # startup transient → NaN

        valid_idx = prices_df.index[valid]
        out.loc[valid_idx, col] = filtered

    return out.squeeze() if squeeze else out


# ── 3. Optimal d for a single series ──────────────────────────────────────────

def find_optimal_d(
    series: pd.Series,
    d_grid: list[float] | None = None,
    thres: float = 1e-5,
    adf_alpha: float = 0.05,
    autolag: str = "AIC",
    regression: str = "c",
) -> float:
    """
    Binary-search for the minimum d ∈ d_grid that makes `series` stationary.

    Strategy
    --------
    1. Coarse sweep over d_grid until the first stationary d is found.
    2. Binary-search between [prev_d, first_stationary_d] for a tighter bound.

    Returns 1.0 if the series is non-stationary at all tested d values,
    and 0.0 if it is already stationary without any differencing.
    """
    if d_grid is None:
        d_grid = [round(x, 2) for x in np.arange(0.0, 1.05, 0.05)]

    log_s = np.log(series.ffill().dropna().values.astype(np.float64))
    if len(log_s) < 20:
        return 1.0

    def _is_stationary(d_val: float) -> bool:
        w  = ffd_weights(d_val, thres)
        L  = len(w)
        if L >= len(log_s):
            return False
        y = lfilter(w, 1.0, log_s)[L - 1:]   # drop startup NaN
        if len(y) < 10:
            return False
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                pval = adfuller(y, autolag=autolag, regression=regression)[1]
            except Exception:
                return False
        return pval < adf_alpha

    # 1. Check d=0 first (already stationary?)
    if _is_stationary(0.0):
        return 0.0

    # 2. Coarse sweep
    first_stat = None
    prev_d     = d_grid[0]
    for d_val in d_grid[1:]:
        if _is_stationary(d_val):
            first_stat = d_val
            break
        prev_d = d_val

    if first_stat is None:
        return 1.0   # never stationary on grid → full differentiation

    # 3. Binary search in (prev_d, first_stat) with precision 0.01
    lo, hi = prev_d, first_stat
    while hi - lo > 0.01:
        mid = round((lo + hi) / 2, 3)
        if _is_stationary(mid):
            hi = mid
        else:
            lo = mid

    return round(hi, 3)


# ── 4. Batch optimal d ────────────────────────────────────────────────────────

def find_optimal_d_batch(
    prices_df: pd.DataFrame,
    d_grid: list[float] | None = None,
    thres: float = 1e-5,
    adf_alpha: float = 0.05,
    n_jobs: int = -1,
    verbose: bool = True,
) -> pd.Series:
    """
    Find optimal d for every column of prices_df in parallel.

    Returns pd.Series indexed by ticker with optimal d values.
    """
    tickers = prices_df.columns.tolist()

    if verbose:
        print(f"[FFD] searching optimal d for {len(tickers)} tickers …")

    def _worker(col):
        return find_optimal_d(
            prices_df[col], d_grid=d_grid, thres=thres, adf_alpha=adf_alpha
        )

    if _HAS_JOBLIB and n_jobs != 1:
        d_vals = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_worker)(col) for col in tickers
        )
    else:
        d_vals = [_worker(col) for col in tickers]

    d_series = pd.Series(d_vals, index=tickers, name="optimal_d")

    if verbose:
        print(
            f"[FFD] d stats — min: {d_series.min():.3f}  "
            f"median: {d_series.median():.3f}  "
            f"max: {d_series.max():.3f}"
        )
    return d_series


# ── 5. Multi-window FFD momentum scores ───────────────────────────────────────

def build_ffd_scores(
    prices_monthly: pd.DataFrame,
    d_series: pd.Series,
    windows: list[int] | None = None,
    thres: float = 1e-5,
) -> dict[int, pd.DataFrame]:
    """
    Build FFD-based momentum score DataFrames, one per lookback window.

    The score for window m at time t is the rolling m-period mean of the
    FFD-transformed log-price series.  This is the FFD analog of the
    cumulative-return momentum signal: it captures the recent trend
    direction in the stationary (memory-preserving) domain.

    Parameters
    ----------
    prices_monthly : T × N monthly close prices
    d_series       : optimal d per ticker (output of find_optimal_d_batch)
    windows        : lookback windows in months (default [1, 3, 6, 9, 12])
    thres          : FFD weight truncation threshold

    Returns
    -------
    dict mapping window → pd.DataFrame (T × N)
        Each value has the same index/columns as prices_monthly.
        Values are NaN where insufficient history exists.
    """
    if windows is None:
        windows = [1, 3, 6, 9, 12]

    # Align d_series to columns present in prices_monthly
    common  = prices_monthly.columns.intersection(d_series.index)
    px      = prices_monthly[common]
    d_align = d_series.reindex(common).fillna(d_series.median())

    # Compute FFD once for all tickers
    ffd_df = ffd_apply(px, d_per_col=d_align, thres=thres)

    scores = {}
    for m in windows:
        if m == 1:
            # Single-period FFD score (no rolling mean needed)
            scores[m] = ffd_df.copy()
        else:
            # Rolling mean over m periods — analogous to cumulative return window
            # min_periods=max(1, m//2) so we don't get too many NaN early on
            scores[m] = ffd_df.rolling(window=m, min_periods=max(1, m // 2)).mean()

    return scores


# ── 5b. Improved FFD features: uniform d + memory-preserving momentum ─────────

def build_ffd_scores_v2(
    prices_monthly: pd.DataFrame,
    d_series: pd.Series,
    windows: list[int] | None = None,
    thres: float = 1e-5,
) -> dict[int, pd.DataFrame]:
    """
    Cross-sectionally coherent FFD features (improved over build_ffd_scores).

    Two fixes vs. v1:
      1. UNIFORM d (cross-sectional median of d_series) applied to every ticker, so
         the feature is the same transform for all names — required for the
         cross-sectional z-scoring done downstream.  (v1 used a per-ticker d, which
         makes z-scores across stocks incomparable.)
      2. Memory-preserving MOMENTUM instead of a rolling mean of the level:
           window 1  → FFD log-price LEVEL  (position vs. long-memory equilibrium)
           window m>1 → ΔFFD_m = level_t - level_{t-m}   (slope in the stationary,
                        memory-retaining domain — the FFD analog of MOM_m)
      All series are causal (FIR filter + backward diff) → no look-ahead.

    Default windows {1, 3, 12}: level + short slope + long slope.  Parsimonious to
    avoid collinearity with the raw zMOM features.
    """
    if windows is None:
        windows = [1, 3, 12]

    common = prices_monthly.columns.intersection(d_series.index)
    px     = prices_monthly[common]
    d_unif = float(np.nanmedian(d_series.reindex(common).values))

    level  = ffd_apply(px, d=d_unif, thres=thres)   # single scalar d → coherent

    scores: dict[int, pd.DataFrame] = {}
    for m in windows:
        scores[m] = level.copy() if m == 1 else level.diff(m)
    return scores


# ── 6. CLI: inspect d values and FFD output ───────────────────────────────────

if __name__ == "__main__":
    import sys, os

    sys.path.insert(0, os.path.dirname(__file__))

    CACHE = "sp500_yf_cache.parquet"
    if not os.path.exists(CACHE):
        print(f"[error] cache not found: {CACHE}  — run deep_momentum_xgb.py --compare first")
        sys.exit(1)

    cached = pd.read_parquet(CACHE)
    prices_monthly = cached["close"].resample("ME").last()
    prices_monthly.index = pd.to_datetime(prices_monthly.index)

    print(f"Loaded monthly prices: {prices_monthly.shape}")

    d_series = find_optimal_d_batch(prices_monthly, n_jobs=-1, verbose=True)
    print("\nOptimal d distribution:")
    print(d_series.describe())
    print("\nSample tickers:")
    print(d_series.sort_values().head(10).to_string())
    print("...")
    print(d_series.sort_values().tail(10).to_string())

    # Show FFD scores for AAPL as a sanity check
    ffd_scores = build_ffd_scores(prices_monthly, d_series, windows=[1, 12])
    aapl_d  = d_series.get("AAPL", None)
    if aapl_d is not None:
        print(f"\nAAPL  d = {aapl_d:.3f}")
        print("FFD-1  (last 6):", ffd_scores[1]["AAPL"].dropna().tail(6).values)
        print("FFD-12 (last 6):", ffd_scores[12]["AAPL"].dropna().tail(6).values)
