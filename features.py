#!/usr/bin/env python3
"""
features.py — DM model feature engineering (Han 2022 + dynamics + optional FFD).

Feature groups and column order:

  MOM (10):      zMOM1, MMOM1, zMOM3, MMOM3, zMOM6, MMOM6, zMOM9, MMOM9, zMOM12, MMOM12
  Dynamics (6):  zACCEL, MACCEL, zVOL, MVOL, zPOS, MPOS
  FFD (6, opt):  zFFD1, MFFD1, zFFD3, MFFD3, zFFD12, MFFD12   (when ffd_scores supplied)
  SIZE (10):     SIZE_1 … SIZE_10

Base total (no FFD): 26 features
With FFD:            32 features
"""

import numpy as np
import pandas as pd

# ── Constants ─────────────────────────────────────────────────────────────────
MOM_WINDOWS = [1, 3, 6, 9, 12]   # cumulative-return lookback windows (months)
FFD_WINDOWS = [1, 3, 12]         # FFD score windows (match build_ffd_scores_v2 default)

# Feature name lists — order matches make_features() insertion order
MOM_FEATURE_NAMES = [name for m in MOM_WINDOWS for name in (f"zMOM{m}", f"MMOM{m}")]

DYNAMICS_FEATURE_NAMES = [
    "zACCEL", "MACCEL",   # momentum acceleration: recent 6m minus older 6m cumret
    "zVOL",   "MVOL",     # trailing 11-month realized vol
    "zPOS",   "MPOS",     # fraction of up months over t-12..t-2
]

FFD_FEATURE_NAMES = [name for m in FFD_WINDOWS for name in (f"zFFD{m}", f"MFFD{m}")]

SIZE_FEATURE_NAMES = [f"SIZE_{s}" for s in range(1, 11)]

BASE_FEATURE_NAMES = MOM_FEATURE_NAMES + DYNAMICS_FEATURE_NAMES + SIZE_FEATURE_NAMES
ALL_FEATURE_NAMES  = MOM_FEATURE_NAMES + DYNAMICS_FEATURE_NAMES + FFD_FEATURE_NAMES + SIZE_FEATURE_NAMES


# ── Feature computation ───────────────────────────────────────────────────────

def make_features(
    rets: pd.DataFrame,
    size: pd.DataFrame,
    t: int,
    ffd_scores: dict | None = None,
) -> pd.DataFrame:
    """
    Build the DM feature matrix at time index t.

    Parameters
    ----------
    rets        : T×N monthly returns (clipped ±50%).
    size        : T×N market-cap proxy (Close × Volume), forward-filled.
    t           : current time index; features use only data up to t-1 (no look-ahead).
    ffd_scores  : dict {window → T×N DataFrame} of FFD log-price scores (optional).
                  When supplied, 6 FFD features (zFFD/MFFD for each window) are appended.

    Returns
    -------
    pd.DataFrame of shape (N_stocks, n_features), indexed by stock ticker.
    Column order: MOM → Dynamics → FFD (if any) → SIZE.

    Notes
    -----
    All features are z-scored cross-sectionally; the cross-sectional mean is retained
    as a separate feature (macro-state variable) — Han (2022) Sec. 3.2.3.
    Requires t >= 13 (needs 12 months of history for dynamics + skip t-1 for MOM).
    """
    feats = {}

    def _zc(s: pd.Series, name: str) -> None:
        """Cross-sectional z-score + retain cross-sectional mean as macro feature."""
        mu, sigma = s.mean(), s.std()
        feats[f"z{name}"] = (s - mu) / (sigma + 1e-10)
        feats[f"M{name}"] = mu

    # ── Raw cumulative-return momentum (paper Eq. 6–8) ──────────────────────
    # m=1: skip the most recent month (reversal avoidance); m>1: cumulative prod
    for m in MOM_WINDOWS:
        mom = (
            rets.iloc[t - 1]
            if m == 1
            else (1 + rets.iloc[t - m : t - 1]).prod() - 1
        )
        mu    = mom.mean()
        sigma = mom.std()
        feats[f"zMOM{m}"] = (mom - mu) / (sigma + 1e-10)
        feats[f"MMOM{m}"] = mu

    # ── Momentum-dynamics features (vectorized over cross-section) ───────────
    win        = rets.iloc[t - 12 : t - 1]                    # 11 months: t-12..t-2
    mom_recent = (1 + rets.iloc[t - 6  : t - 1]).prod() - 1   # t-6..t-2  (recent half)
    mom_older  = (1 + rets.iloc[t - 12 : t - 6]).prod() - 1   # t-12..t-7 (older half)
    _zc(mom_recent - mom_older, "ACCEL")   # momentum acceleration: rising vs. fading
    _zc(win.std(),              "VOL")     # trailing realized vol (momentum-risk conditioning)
    _zc((win > 0).mean(),       "POS")     # frog-in-the-pan: fraction of up months

    # ── FFD momentum (López de Prado, AFML Ch. 5) — optional ─────────────────
    # Stationary, memory-preserving trend signal.  Optimal d is frozen on the
    # pre-OOS training window (causal); filter is applied to the full series.
    if ffd_scores is not None:
        for m in sorted(ffd_scores.keys()):
            row   = ffd_scores[m].iloc[t - 1].reindex(rets.columns)
            mu    = row.mean()
            sigma = row.std()
            feats[f"zFFD{m}"] = (row - mu) / (sigma + 1e-10)
            feats[f"MFFD{m}"] = mu

    # ── Size decile dummies (paper Sec. 3.3.1) ───────────────────────────────
    # 10 binary indicators D_s (s=1..10); s=1 smallest cap, s=10 largest cap.
    cap         = size.iloc[t - 1].rank(pct=True, na_option="keep").fillna(0.5)
    size_decile = ((cap * 9.999).astype(int) + 1)   # map [0,1) → {1..10}
    for s in range(1, 11):
        feats[f"SIZE_{s}"] = (size_decile == s).astype(float)

    return pd.DataFrame(feats, index=rets.columns)


def feature_names(with_ffd: bool = False) -> list[str]:
    """Return the ordered list of feature column names produced by make_features()."""
    return ALL_FEATURE_NAMES if with_ffd else BASE_FEATURE_NAMES
