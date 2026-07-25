#!/usr/bin/env python3
"""shrinkage_combine.py — "RenTec combines millions of signals" — test the combiner THEY would use.

WHAT THE PREVIOUS TEST DID AND DID NOT SHOW. erc_signals.py found that equal-weight, inverse-vol, ERC,
score-blending and position-netting ALL lose to the best single signal. But every one of those is a NAIVE
combiner that ignores the covariance between signals and the difference in their expected returns. That is
NOT how a large multi-signal fund combines anything.

The relevant reference is Kozak, Nagel & Santosh, "Shrinking the Cross-Section" (JFE 2020). Their result is
precisely a two-sided negative: characteristic-SPARSE models fail (you cannot pick 5 winners), AND naive
dense models fail (you cannot equal-weight everything). What works is DENSE + SHRINKAGE — a Bayesian/ridge
estimate of the tangency portfolio over many correlated predictors:

        w(λ) ∝ (Σ + λI)^{-1} μ

    λ → 0    mean-variance optimal in-sample, wildly unstable out-of-sample (Σ is near-singular when the
             signals are 0.7-0.85 correlated, so the inverse amplifies estimation noise)
    λ → ∞    collapses toward equal-weight — the naive combiner we already tested
    λ*       the shrinkage that trades estimation error against diversification. THIS is the arm we skipped.

Also reported: GRINOLD'S FUNDAMENTAL LAW, IR ≈ IC·√breadth, with breadth estimated as the effective number
of INDEPENDENT signals, N_eff = (Σᵢλᵢ)² / Σᵢλᵢ² on the signal-return correlation matrix. That number tells us
whether "more signals" can possibly help here: five signals at 0.85 correlation are NOT five bets.

All weights are estimated on an EXPANDING window (leak-free) and applied out-of-sample.
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST
from horse_race_v2 import S, tc, bf, hub, pnl, spy, adv
from dm_capacity import book


def stats(tag, x):
    D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
    if len(D) < 24: return
    X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
    e = D.r.values - X @ c
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
    eq = (1 + D.r).cumprod()
    print(f"  {tag:32}{D.r.mean()*12:>8.1%}{D.r.std()*np.sqrt(12):>7.1%}"
          f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>6.2f}{(eq/eq.cummax()-1).min():>8.1%}"
          f"{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>6.2f}", flush=True)


if __name__ == "__main__":
    q = adv.rank(axis=1, pct=True); Q3 = (q > 0.4) & (q <= 0.6)
    names = ["hi52", "mom11", "tvalpast", "resmom", "fip"]
    R = {}
    for n in names:
        W = book(S[n], uni=Q3)
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                              transaction_cost=tc, borrow_fee=bf)
        x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
        x.index = pd.PeriodIndex(x.index, freq="M"); R[n] = x
    RET = pd.DataFrame(R).dropna()

    C = RET.corr().values
    ev = np.linalg.eigvalsh(C)
    n_eff = (ev.sum() ** 2) / (ev ** 2).sum()
    print(f"\n[shrinkage_combine] {len(RET)} months · {len(names)} signals")
    print(f"  eigenvalues of the signal correlation matrix: {np.round(sorted(ev)[::-1], 2)}")
    print(f"  EFFECTIVE INDEPENDENT SIGNALS N_eff = {n_eff:.2f}  (of {len(names)} nominal)")
    print(f"  → Grinold IR ≈ IC·sqrt(breadth): combining {len(names)} signals buys sqrt({n_eff:.2f}/1) ="
          f" {np.sqrt(n_eff):.2f}x, not sqrt({len(names)}) = {np.sqrt(len(names)):.2f}x")

    H = f"  {'combiner':32}{'ann':>8}{'vol':>7}{'SR':>6}{'maxDD':>8}{'alpha':>9}{'t':>6}"
    print("\nSHRINKAGE PATH  w(λ) ∝ (Σ+λI)⁻¹μ, expanding-window, leak-free"); print(H)
    for n in names: stats(f"single: {n}", RET[n])
    stats("equal-weight", RET.mean(axis=1))

    for lam in (0.0, 0.01, 0.05, 0.2, 1.0, 5.0, 50.0):
        out = pd.Series(0.0, index=RET.index)
        for i in range(36, len(RET)):
            H_ = RET.iloc[:i]                                   # strictly past
            mu = H_.mean().values
            Sg = np.cov(H_.values, rowvar=False)
            sc = np.trace(Sg) / len(mu)                          # scale λ to the covariance magnitude
            w = np.linalg.solve(Sg + lam * sc * np.eye(len(mu)), mu)
            w = w / (np.abs(w).sum() + 1e-12)
            out.iloc[i] = float(RET.iloc[i].values @ w)
        stats(f"ridge λ={lam:g}", out[out != 0])
