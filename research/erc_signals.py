#!/usr/bin/env python3
"""erc_signals.py — ERC across the raw momentum SIGNAL BOOKS (not a score blend).

WHY THIS IS A DIFFERENT TEST FROM `composite`. The composite that lost in T26 and again today
(+27.32% vs hi52's +37.10%) blends the SCORES and then forms ONE book — averaging five rankings dilutes
the best one. ERC instead lets each signal build its OWN book and allocates CAPITAL across the resulting
return streams by equal risk contribution. Even at 0.9 book correlation that can still lower volatility,
so the right question is not "does it raise alpha" (it will not, it is an average) but "does it raise the
SHARPE / t-stat by killing idiosyncratic noise".

AND IT MEASURES THE NETTING CLAIM. Two ways to combine, which are NOT equivalent once costs are real:
  (a) RETURN-level  — ERC-weight the five net return streams. Every sleeve pays its own costs in full.
                      This is what ERC.combine() does today.
  (b) POSITION-level — ERC-weight the five WEIGHT matrices, sum to one book, then backtest ONCE. Offsetting
                      positions across signals cancel BEFORE trading, so the netted book pays less.
The gap between (a) and (b) is the internalisation benefit, in basis points, on our own data.

Allocators: erc (equal risk contribution) · invvol · equal · best-single · score-composite.
All on Q3 (best net size bucket), hold=6, banded, MEASURED IBKR costs. Leak-free expanding-window weights.
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST, ERC
from horse_race_v2 import S, tc, bf, hub, pnl, spy, adv, parts
from dm_capacity import book

Q3 = None


def netser(W, aum=0):
    r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                          transaction_cost=tc, borrow_fee=bf)
    x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
    x.index = pd.PeriodIndex(x.index, freq="M")
    return x, r["ann_turnover"]


def stats(tag, x, turn=np.nan):
    D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
    if len(D) < 24: return
    X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
    e = D.r.values - X @ c
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
    eq = (1 + D.r).cumprod()
    print(f"  {tag:30}{D.r.mean()*12:>8.1%}{D.r.std()*np.sqrt(12):>7.1%}"
          f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>6.2f}{(eq/eq.cummax()-1).min():>8.1%}"
          f"{c[1]:>6.2f}{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>6.2f}{turn:>6.1f}", flush=True)


if __name__ == "__main__":
    q = adv.rank(axis=1, pct=True); Q3 = (q > 0.4) & (q <= 0.6)
    names = ["hi52", "mom11", "tvalpast", "resmom", "fip"]
    W, R, T = {}, {}, {}
    for n in names:
        W[n] = book(S[n], uni=Q3)
        R[n], T[n] = netser(W[n])
    RET = pd.DataFrame(R).dropna()
    print(f"\n[erc_signals] Q3 · hold=6 · band · measured IBKR costs · {len(RET)} months")
    print("\nNET BOOK-RETURN CORRELATION (are these one bet?)")
    print(RET.corr().round(2).to_string())

    H = (f"  {'book':30}{'ann':>8}{'vol':>7}{'SR':>6}{'maxDD':>8}{'beta':>6}{'alpha':>9}{'t':>6}{'turn':>6}")
    print("\n(a) RETURN-LEVEL combination — each sleeve pays its own costs"); print(H)
    for n in names: stats(n, RET[n], T[n])
    Aw = ERC.expanding_alloc(RET, method="erc", win=36)
    Aw = pd.DataFrame(Aw, index=RET.index, columns=RET.columns) if not isinstance(Aw, pd.DataFrame) else Aw
    stats("ERC (return-level)", (RET * Aw.values).sum(axis=1))
    Iv = ERC.expanding_alloc(RET, method="invvol", win=36)
    Iv = pd.DataFrame(Iv, index=RET.index, columns=RET.columns) if not isinstance(Iv, pd.DataFrame) else Iv
    stats("inv-vol (return-level)", (RET * Iv.values).sum(axis=1))
    stats("equal-weight (return-level)", RET.mean(axis=1))
    stats("score-composite (1 book)", *netser(book(S["composite"], uni=Q3)))

    print("\n(b) POSITION-LEVEL combination — one netted book, costs paid ONCE"); print(H)
    for tag, A in (("ERC", Aw), ("inv-vol", Iv), ("equal", pd.DataFrame(1.0/len(names), index=RET.index, columns=RET.columns))):
        acc = None
        for n in names:
            w = W[n].reindex(index=RET.index.to_timestamp("M"), method="ffill").fillna(0.0)
            a = A[n].reindex(RET.index).values[:, None]
            acc = (w * a) if acc is None else acc.add(w * a, fill_value=0.0)
        x, tn = netser(acc)
        stats(f"{tag} (position-level, NETTED)", x, tn)
    print("\n  The (b)-minus-(a) gap on the same allocator = the internalisation benefit.")
