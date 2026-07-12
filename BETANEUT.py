#!/usr/bin/env python3
"""BETANEUT.py — the beta-neutralization construction layer, reusable across sleeves.

Strips MARKET beta from a long/short weight book: w' = w - beta * (w.beta)/(beta.beta), then renorm gross to 2.0.
This is what gives MOM/DM their +skew and crash-resistance (MOM 0.98->1.17, MH-DM 1.26->1.30). Market beta is a
leak-free rolling-BW-month regression of each stock's return on the eligible-universe equal-weight market,
shifted one period. NOTE: this is the RISK-LAYER 'beta neutralization' (construction) — NOT the vol-managed
'beta sleeve' (a directional index position, parked in BETA_SLEEVE_RESEARCH.md).

API:
  rolling_beta(mret, elig, bw=60)  -> DataFrame of leak-free per-stock market betas (aligned to mret.index)
  neutralize(w, beta_row)          -> beta-neutralize one date's weight Series (renorm gross 2.0)
  betaneut(W, BETA)                -> beta-neutralize a whole (dates x tickers) weight book
"""
import numpy as np, pandas as pd

def rolling_beta(mret, elig, bw=60, min_periods=36):
    """Leak-free rolling market beta per stock (regress stock return on eligible-universe equal-weight market)."""
    mkt = mret.where(elig).mean(axis=1)
    mr  = mret.rolling(bw, min_periods=min_periods).mean(); mm = mkt.rolling(bw, min_periods=min_periods).mean()
    mrm = mret.mul(mkt, axis=0).rolling(bw, min_periods=min_periods).mean()
    vm  = (mkt**2).rolling(bw, min_periods=min_periods).mean() - mm**2
    return mrm.sub(mr.mul(mm, axis=0)).div(vm, axis=0).shift(1)             # shift(1) -> known at signal date

def neutralize(w, beta_row):
    """Beta-neutralize one date's weight Series; renorm gross to 2.0 (missing betas -> 1.0)."""
    b = beta_row.reindex(w.index).fillna(1.0).values
    wv = w.values - b * ((w.values @ b) / (b @ b + 1e-9)); g = np.abs(wv).sum()
    return pd.Series(wv * (2.0 / g) if g > 0 else wv, index=w.index)

def betaneut(W, BETA):
    """Beta-neutralize a whole weight book (dates x tickers)."""
    cols = W.columns
    rows = {dt: neutralize(W.loc[dt].dropna(), BETA.loc[dt]) for dt in W.index if dt in BETA.index}
    return pd.DataFrame(rows).T.reindex(columns=cols).fillna(0.0)
