#!/usr/bin/env python3
"""size_attribution.py — "does momentum live in a big universe?"

Our capacity test showed the Han-DM alpha dies above ~$10M AUM, which SUGGESTS the edge is a small-cap
phenomenon. But that inference is indirect: restricting the UNIVERSE before selection is not the same
question as asking where the PnL of the UNRESTRICTED book comes from. Han himself claims the opposite
(§4.4.1): after Return-reclassification his long book averages $2,495M vs a $1,709M market average, and
excluding the bottom 20% NYSE size still leaves SR 1.08 — i.e. "the profits are NOT driven by small firms."

This decomposes the actual book:
  A. WHAT IT HOLDS  — the dollar-volume distribution of selected names vs the universe (Han's size-shift test)
  B. WHERE THE PnL IS — split the SAME book's return into size-quintile sub-books, each self-financed, so we
     see which size bucket actually pays. Gross AND net (measured IBKR costs), with alpha vs SPY.
If the alpha is flat across size buckets, momentum lives everywhere and our capacity limit is about the
EQUAL-WEIGHTING, not about the signal. If it is monotone in size, it is a small-cap premium.
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST
from dm_criteria import scores, hub, pnl, sm, spy, shortable
from dm_capacity import book
from ibkr_costs import ibkr_cost_panel, borrow_panel

tc = ibkr_cost_panel(hub); bf = borrow_panel(hub)
adv = sm.rolling(3, min_periods=1).mean()
S = {c: scores(c) for c in ("sharpe", "ret")}


def held_size_profile(Sc):
    """A. Median dollar volume of LONG / SHORT holdings vs the universe median (Han's §4.4.1 test)."""
    rows = []
    for d, s in Sc.items():
        s = s.dropna()
        if len(s) < 50 or d not in adv.index: continue
        n = max(1, int(len(s) * 0.10))
        a = adv.loc[d]
        rows.append(dict(uni=a.reindex(s.index).median(),
                         lng=a.reindex(s.nlargest(n).index).median(),
                         sht=a.reindex(s.nsmallest(n).index).median()))
    R = pd.DataFrame(rows)
    return R.median()


def rep(tag, W):
    out = []
    for lbl, kw in (("g", dict(transaction_cost=0.0, borrow_fee=0.0)),
                    ("n", dict(transaction_cost=tc, borrow_fee=bf))):
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index), **kw)
        x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
        if len(x) < 24: return
        x.index = pd.PeriodIndex(x.index, freq="M")
        D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
        X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
        e = D.r.values - X @ c
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
        out.append((D.r.mean() * 12, D.r.std() * np.sqrt(12), c[0] * 12, c[0] / (se[0] + 1e-12)))
    (ga, gv, gal, gt), (na, nv, nal, nt) = out
    print(f"  {tag:30}{ga:>8.1%}{gal:>+9.2%}{gt:>6.2f} |{na:>9.1%}{na/(nv+1e-9):>6.2f}{nal:>+9.2%}{nt:>6.2f}",
          flush=True)


if __name__ == "__main__":
    print("\nA. WHAT THE BOOK HOLDS — median monthly dollar volume (Han §4.4.1: his book shifts LARGE)")
    for c in ("sharpe", "ret"):
        p = held_size_profile(S[c])
        print(f"  {c:8} universe ${p['uni']/1e6:>9,.1f}M   LONG ${p['lng']/1e6:>9,.1f}M "
              f"({p['lng']/p['uni']:>4.2f}x)   SHORT ${p['sht']/1e6:>9,.1f}M ({p['sht']/p['uni']:>4.2f}x)")

    print("\nB. WHERE THE PnL IS — the same signal, traded ONLY within each size quintile (self-financed)")
    print(f"  {'sub-book':30}{'gANN':>8}{'gALPHA':>9}{'t':>6} |{'nANN':>9}{'nSR':>6}{'nALPHA':>9}{'t':>6}")
    q = adv.rank(axis=1, pct=True)
    for c in ("sharpe", "ret"):
        print(f"  -- {c} --")
        for lo, hi, lbl in ((0.0, 0.2, "Q1 smallest"), (0.2, 0.4, "Q2"), (0.4, 0.6, "Q3"),
                            (0.6, 0.8, "Q4"), (0.8, 1.01, "Q5 largest")):
            uni = (q > lo) & (q <= hi)
            try: rep(f"{c} {lbl}", book(S[c], uni=uni))
            except Exception as ex: print(f"  {c} {lbl}: {type(ex).__name__}")
    q3_capacity()


def q3_capacity():
    """Q3 (mid-cap) had the best NET alpha (+9.67%, t=2.72). What is ITS capacity? Q1 has more gross alpha
    but loses it to cost; Q3 trades better. Capacity is the number that decides if this is a real strategy."""
    from ibkr_costs import cost_at_aum
    q = adv.rank(axis=1, pct=True)
    print("\nC. CAPACITY of the best NET bucket (Q3) vs the best GROSS bucket (Q1)")
    print(f"  {'sub-book @ AUM':30}{'ann':>8}{'vol':>7}{'SR':>6}{'maxDD':>8}{'alpha':>9}{'t':>6}")
    for lbl, (lo, hi) in (("Q1 smallest", (0.0, 0.2)), ("Q3 middle", (0.4, 0.6)),
                          ("Q3+Q4 mid", (0.4, 0.8))):
        W = book(S["sharpe"], uni=(q > lo) & (q <= hi))
        for aum in (0, 1e6, 1e7, 5e7, 1e8, 2.5e8):
            t_ = cost_at_aum(hub, W, aum) if aum else tc
            r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                                  transaction_cost=t_, borrow_fee=bf)
            x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
            if len(x) < 24: continue
            x.index = pd.PeriodIndex(x.index, freq="M")
            D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
            X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
            e = D.r.values - X @ c
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
            eq = (1 + D.r).cumprod()
            tag = f"{lbl} @ ${aum/1e6:,.0f}M" if aum else f"{lbl} (no impact)"
            print(f"  {tag:30}{D.r.mean()*12:>8.1%}{D.r.std()*np.sqrt(12):>7.1%}"
                  f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>6.2f}"
                  f"{(eq/eq.cummax()-1).min():>8.1%}{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>6.2f}", flush=True)
        print()
