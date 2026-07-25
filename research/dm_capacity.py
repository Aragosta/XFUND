#!/usr/bin/env python3
"""dm_capacity.py — "momentum trades all the time, why can't WE get it to work?"

The suspicion: we are not measuring momentum, we are measuring momentum IN AN UNTRADEABLE UNIVERSE.
Facts that motivate this (measured 2026-07-25):
  - our tiered model charges the Han-DM book a mean 93 bp ONE-WAY; the liquid tier 56 bp; full universe 132 bp
  - Frazzini-Israel-Moskowitz (2015) MEASURED median institutional cost = 6.2 bp per rebalance
  - hub.elig('liquid') = mdv > $5M per MONTH ≈ $250k/day. That is a microcap, not an institutional universe.
  - the median name in our 15,502-name survivorship-free panel pays the WORST tier (150 bp)
So the book equal-weights thousands of names an institution would never touch, and the engine correctly
reports that trading them is ruinous. That is a statement about the UNIVERSE, not about momentum.

THIS SCRIPT SEPARATES THE THREE EXPLANATIONS:
  (1) UNIVERSE   — restrict to genuinely institutional liquidity ($100M/mo ≈ $5M/day, and $1B/mo)
  (2) COST LEVEL — our tiers vs a flat institutional 10 bp / 5 bp (FIM-style)
  (3) BORROW     — our tiers (5-25%/yr in the mid buckets!) vs general-collateral 0.5%/yr, and long-only
If momentum "works" for real desks because they trade liquid names at 6 bp with GC borrow, then arms with
those assumptions should show it, and our failure is an artifact of assumptions, not of the signal.
"""
import warnings, sys, os, pickle
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST
from DATAHUB import DataHub
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "research"))
from dm_criteria import scores, PROB, hub, pnl, sm, spy, shortable

tc_ours = BACKTEST.tiered_transaction_costs(sm); bf_ours = BACKTEST.tiered_borrow_fees(sm)
mdv = sm.rolling(3, min_periods=1).mean()
UNIS = {"all": None,
        "mdv>$5M/mo (our 'liquid')":   mdv > 5e6,
        "mdv>$100M/mo (~$5M/day)":     mdv > 1e8,
        "mdv>$1B/mo (~$50M/day)":      mdv > 1e9}


def book(S, uni=None, hold=6, band=(0.10, 0.30), dec=0.10, ls=True):
    dates = sorted(S); tranche, hh, hl, W = [], set(), set(), {}
    for d in dates:
        s = S[d].dropna()
        if uni is not None and d in uni.index:
            k = uni.loc[d]; s = s[s.index.intersection(k[k].index)]
        if len(s) < 30: continue
        rh = s.rank(ascending=False, pct=True)
        cand = s
        if ls and d in shortable.index:
            sh = shortable.loc[d]; cand = s[s.index.isin(sh[sh].index)]
        rl = cand.rank(ascending=True, pct=True) if len(cand) else None
        en, ex = band if band else (dec, dec)
        hi = [t for t in (set(rh[rh <= en].index) | (hh & set(rh[rh <= ex].index))) if t in s.index]
        if not hi: continue
        w = pd.Series(0.0, index=s.index); w[hi] = 1.0 / len(hi); hh = set(hi)
        if ls and rl is not None and len(cand) >= 20:
            lo = [t for t in (set(rl[rl <= en].index) | (hl & set(rl[rl <= ex].index))) if t in cand.index]
            if lo: w[lo] = w[lo] - 1.0 / len(lo); hl = set(lo)
        tranche.append(w); tranche = tranche[-hold:]
        W[d] = pd.concat(tranche, axis=1).fillna(0.0).mean(axis=1)
    return pd.DataFrame(W).T.reindex(columns=pnl.columns).fillna(0.0)


def rep(tag, W, tcost, bfee):
    if W.abs().sum().sum() == 0: print(f"  {tag:44} empty"); return
    try:
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index),
                              transaction_cost=tcost, borrow_fee=bfee)
        x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
        x.index = pd.PeriodIndex(x.index, freq="M")
        D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
        if len(D) < 24: return
        X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
        e = D.r.values - X @ c
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
        eq = (1 + D.r).cumprod()
        print(f"  {tag:44}{D.r.mean()*12:>8.1%}{D.r.std()*np.sqrt(12):>7.1%}"
              f"{D.r.mean()*12/(D.r.std()*np.sqrt(12)+1e-9):>6.2f}{(eq/eq.cummax()-1).min():>8.1%}"
              f"{c[1]:>6.2f}{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>6.2f}{r['ann_turnover']:>6.1f}", flush=True)
    except Exception as ex:
        print(f"  {tag:44} FAILED {type(ex).__name__}")


H = (f"  {'book':44}{'ann':>8}{'vol':>7}{'SR':>6}{'maxDD':>8}{'beta':>6}{'alpha':>9}{'t':>6}{'turn':>6}")
if __name__ == "__main__":
    S = {c: scores(c) for c in ("ret", "sharpe")}
    print("\n(1) UNIVERSE — same book, progressively institutional liquidity  [our costs, hold=6, band]"); print(H)
    for c in ("ret", "sharpe"):
        for un, m in UNIS.items():
            rep(f"{c} L/S · {un}", book(S[c], uni=m), tc_ours, bf_ours)

    print("\n(2) COST LEVEL — mdv>$100M universe, our tiers vs institutional flat"); print(H)
    for c in ("ret", "sharpe"):
        W = book(S[c], uni=UNIS["mdv>$100M/mo (~$5M/day)"])
        rep(f"{c} L/S · our tiers",            W, tc_ours, bf_ours)
        rep(f"{c} L/S · 10bp + our borrow",    W, 0.0010,  bf_ours)
        rep(f"{c} L/S · 10bp + GC borrow 0.5%", W, 0.0010, 0.005)
        rep(f"{c} L/S · 6bp (FIM) + GC borrow", W, 0.0006, 0.005)

    print("\n(3) LONG-ONLY at institutional liquidity (no borrow at all)"); print(H)
    for c in ("ret", "sharpe"):
        for un in ("mdv>$100M/mo (~$5M/day)", "mdv>$1B/mo (~$50M/day)"):
            W = book(S[c], uni=UNIS[un], ls=False)
            rep(f"{c} LONG-ONLY · {un} · our tiers", W, tc_ours, 0.0)
            rep(f"{c} LONG-ONLY · {un} · 10bp",      W, 0.0010, 0.0)
