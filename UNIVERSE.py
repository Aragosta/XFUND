#!/usr/bin/env python3
"""UNIVERSE.py — shared UNIVERSE + FEATURE utilities (extracted from the retired deep_momentum_xgb).

Delisting-injection, cleaned PnL prices, and return-winsorization are DATA ENGINEERING and live GLOBALLY in
DATAHUB (hub.delisted_prices / hub.clean_returns / hub.dollar_size / hub.pnl). This module holds the remaining
point-in-time trading-prep both MOM and DM share:

  eligibility(m_px, size, ...)  -> (eligible, shortable) point-in-time masks (short side has a stricter float)
  ffd_scores(m_px, t_cut)       -> leak-free fractional-difference features (d chosen on <=t_cut, causal filter)
"""
import numpy as np, pandas as pd


def eligibility(prices_monthly, size_monthly=None, min_price=5.0, min_coverage=0.70, window=36,
                min_history=12, min_dollar_vol_pct=0.0, min_dollar_vol_abs=5e6, short_min_dollar_vol_abs=25e6):
    """Point-in-time (causal) tradeability. Returns (eligible, shortable), T×N booleans. Short side clears a
    higher dollar-volume floor (borrowability / locate proxy)."""
    valid   = prices_monthly.notna()
    good_px = (prices_monthly > min_price) & valid
    n_valid = valid.rolling(window, min_periods=min_history).sum()
    n_good  = good_px.rolling(window, min_periods=min_history).sum()
    cov     = n_good / n_valid.clip(lower=1)
    elig    = (cov >= min_coverage) & (prices_monthly > min_price) & (n_valid >= min_history)
    shortable = elig.copy()
    if size_monthly is not None:
        dv = size_monthly.reindex_like(prices_monthly).rolling(3, min_periods=1).mean()
        if min_dollar_vol_pct > 0: elig = elig & (dv.rank(axis=1, pct=True) >= min_dollar_vol_pct)
        if min_dollar_vol_abs > 0: elig = elig & (dv >= min_dollar_vol_abs)
        shortable = elig & (dv >= short_min_dollar_vol_abs)
    return elig.fillna(False), shortable.fillna(False)


def ffd_scores(prices_monthly, t_cut):
    """Leak-free FFD: pick optimal d on training data <= t_cut, apply the causal filter to the full series."""
    from ffd import find_optimal_d_batch, build_ffd_scores_v2
    d_series = find_optimal_d_batch(prices_monthly.iloc[:t_cut], n_jobs=-1, verbose=False)
    return build_ffd_scores_v2(prices_monthly, d_series, windows=[1, 3, 12])
