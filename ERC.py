#!/usr/bin/env python3
"""ERC.py — capital allocator (Equal Risk Contribution / risk-parity), the sleeve-level MIX layer.

For a FEW similar-Sharpe sleeves use ERC; it needs only the covariance (estimable), not expected returns.
Weights estimated on an EXPANDING window (leak-free). The MIX (ERC) and the SIZE (vol-target) are separate.
Used by META.py to combine the beta-neut MOM + DM books at the sleeve (capital) level, BEFORE the name-level
meta overlay. API:
  erc_weights(cov, budget=None)        -> ERC weights (equal risk contribution)
  expanding_alloc(rets, method, win)   -> DataFrame of leak-free weights per period
  combine(rets, weights)               -> combined return stream
  vol_target(r, target)                -> size overlay (scale to target annual vol, leak-free)
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd


def erc_weights(cov, budget=None, iters=1000, tol=1e-10):
    """Equal Risk Contribution weights (long-only, sum to 1) via the standard fixed-point iteration.
    At convergence w_i * (cov @ w)_i is proportional to budget_i (default: equal)."""
    n = cov.shape[0]; w = np.ones(n) / n
    b = np.ones(n) / n if budget is None else np.asarray(budget, float) / np.sum(budget)
    for _ in range(iters):
        mrc = cov @ w                                    # marginal risk (unscaled)
        w_new = b / np.maximum(mrc, 1e-12)               # fixed-point: drives w_i*(cov w)_i -> b_i
        w_new = w_new / w_new.sum()
        if np.abs(w_new - w).max() < tol: w = w_new; break
        w = 0.5 * w + 0.5 * w_new                        # damped for stability
    return w


def risk_contributions(cov, w):
    s = cov @ w; port_var = float(w @ s)
    return (w * s) / port_var if port_var > 0 else np.full(len(w), np.nan)


def expanding_alloc(rets, method="erc", win=36, shrink=0.1):
    """Leak-free weights each period from history up to t-1. method in {equal, invvol, erc}."""
    R = rets.values; T, n = R.shape; W = np.full((T, n), 1.0 / n)
    for i in range(T):
        if i < win: continue
        H = R[:i]                                         # strictly past
        if method == "equal":
            W[i] = 1.0 / n
        elif method == "invvol":
            v = H.std(0); iv = 1.0 / np.maximum(v, 1e-9); W[i] = iv / iv.sum()
        else:                                            # erc (shrunk covariance)
            S = np.cov(H.T); S = (1 - shrink) * S + shrink * np.diag(np.diag(S))
            W[i] = erc_weights(S)
    return pd.DataFrame(W, index=rets.index, columns=rets.columns)


def combine(rets, weights):
    return (rets * weights.shift(0)).sum(axis=1)         # weights already known at t (built from <t)


def vol_target(r, target=0.10, win=6, lo=0.25, hi=3.0):
    """Size overlay: scale each period by target/trailing-vol (leak-free). Separate from the mix."""
    fvol = r.rolling(win, min_periods=3).std().shift(1) * np.sqrt(12)
    scale = (target / fvol).clip(lo, hi).fillna(1.0)
    return scale * r


def perf(r):
    r = pd.Series(r).dropna(); ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); return ann, (r.mean() * 12) / vol if vol > 0 else np.nan, (eq / eq.cummax() - 1).min()


if __name__ == "__main__":
    import pickle
    dm_g, dm_n = pickle.load(open("/tmp/dm_streams.pkl", "rb"))
    M = pickle.load(open("/tmp/mom_champ.pkl", "rb"))
    DM = pd.Series(dm_n).dropna(); DM.index = pd.DatetimeIndex(DM.index).to_period("M")
    MOM = pd.Series(M["n1"]).dropna(); MOM.index = pd.DatetimeIndex(MOM.index).to_period("M")
    df = pd.DataFrame({"DM": DM, "MOM": MOM}).dropna()
    print("=" * 58)
    print(f"=== allocation.py demo — DM + MOM ({len(df)} months, net) ===")
    print(f"  corr(DM,MOM) = {df.corr().iloc[0,1]:.2f}")
    for nm, r in [("DM", df["DM"]), ("MOM", df["MOM"])]:
        a, s, m = perf(r); print(f"  standalone {nm:4}: SR {s:.2f}  ann {a:.1%}  maxDD {m:.1%}")
    print(f"\n  {'method':16}{'SR':>6}{'ann':>8}{'maxDD':>8}   avg weights (DM/MOM)")
    for method in ("equal", "invvol", "erc"):
        W = expanding_alloc(df, method=method); c = combine(df, W).dropna()
        a, s, m = perf(c); aw = W.loc[c.index].mean()
        print(f"  {method:16}{s:>6.2f}{a:>8.1%}{m:>8.1%}   {aw['DM']:.2f}/{aw['MOM']:.2f}")
    # size overlay demo on the ERC book
    Werc = expanding_alloc(df, method="erc"); erc = combine(df, Werc).dropna()
    a, s, m = perf(vol_target(erc, 0.10))
    print(f"\n  ERC + vol-target(10%): SR {s:.2f}  ann {a:.1%}  maxDD {m:.1%}")
    print("[done]")
