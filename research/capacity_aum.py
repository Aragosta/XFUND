#!/usr/bin/env python3
"""capacity_aum.py — at what AUM does the momentum alpha actually die?

CONTEXT. Two cost models give opposite answers on the SAME book:
  old dollar-volume tiers (150 bp for microcaps)  -> sharpe L/S alpha -1.37%  (t=-0.54)
  measured IBKR + Corwin-Schultz (20 bp)          -> sharpe L/S alpha +9.68%  (t= 3.78)
Neither is right, because BOTH ignore the variable that actually binds: SIZE. Corwin-Schultz measures the
SPREAD you cross, not the IMPACT you cause, and it under-estimates for names that barely trade (when a stock
does not trade, high = low and the estimator collapses toward zero). So the honest question is not "what does
a trade cost" but "how much can this book hold before its own trading eats the alpha".

MODEL. Square-root market impact (Almgren / Kyle-consistent, the industry standard):
    impact_i = k · sqrt( participation_i ),   participation_i = |Δw_i| · AUM / ADV_i
with k = 0.10 (a common calibration: trading 100% of ADV costs ~10%). Total one-way cost per name is then
    spread/2 (Corwin-Schultz, measured) + IBKR commission (per-share) + impact(AUM)
Impact is the ONLY term that scales with AUM, and it is the term that turns a paper edge into a real capacity.

READ THE OUTPUT AS: the AUM at which net alpha crosses zero is the strategy's honest capacity.
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST
from dm_criteria import scores, hub, pnl, sm, spy
from dm_capacity import book, UNIS
from ibkr_costs import ibkr_cost_panel, borrow_panel

K_IMPACT = 0.10
base_tc = ibkr_cost_panel(hub); bf = borrow_panel(hub)
adv = sm.rolling(3, min_periods=1).mean()          # monthly dollar volume


def cost_at_aum(W, aum):
    """Per-name one-way cost INCLUDING size-dependent square-root impact for a book of `aum` dollars."""
    dW = W.diff().abs().fillna(W.abs())
    part = (dW * aum).div(adv.reindex_like(W).replace(0, np.nan))
    imp = K_IMPACT * np.sqrt(part.clip(lower=0))
    return (base_tc.reindex_like(W).fillna(base_tc.stack().median()) + imp).clip(0.0001, 0.50)


def rep(tag, W, aum):
    tc = cost_at_aum(W, aum) if aum else base_tc
    try:
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                              transaction_cost=tc, borrow_fee=bf)
        x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
        x.index = pd.PeriodIndex(x.index, freq="M")
        D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
        if len(D) < 24: return
        X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
        e = D.r.values - X @ c
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
        eq = (1 + D.r).cumprod()
        med_part = ((W.diff().abs().fillna(W.abs()) * aum).div(adv.reindex_like(W).replace(0, np.nan))
                    ).stack().median() if aum else 0.0
        print(f"  {tag:34}{D.r.mean()*12:>8.1%}{D.r.std()*np.sqrt(12):>7.1%}"
              f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>6.2f}{(eq/eq.cummax()-1).min():>8.1%}"
              f"{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>6.2f}{med_part:>9.1%}", flush=True)
    except Exception as ex:
        print(f"  {tag:34} FAILED {type(ex).__name__}")


if __name__ == "__main__":
    S = {c: scores(c) for c in ("ret", "sharpe")}
    H = (f"  {'book @ AUM':34}{'ann':>8}{'vol':>7}{'SR':>6}{'maxDD':>8}{'alpha':>9}{'t':>6}"
         f"{'med%ADV':>9}")
    for c in ("sharpe", "ret"):
        for un in ("all", "mdv>$100M/mo (~$5M/day)"):
            W = book(S[c], uni=UNIS[un])
            print(f"\n[{c} L/S · {un}] square-root impact k={K_IMPACT}, on top of measured spread+commission")
            print(H)
            rep("no impact (paper edge)", W, 0)
            for aum in (1e6, 1e7, 5e7, 1e8, 5e8, 1e9):
                rep(f"AUM ${aum/1e6:,.0f}M", W, aum)
