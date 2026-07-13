#!/usr/bin/env python3
"""mr_daily.py — the REAL reversion construction (QP) on the honest daily engine. Not naive reversal (=0):
  - RARITY: per-stock z-score of the 3-day move vs own trailing history (adaptive oversold, replaces RSI2 fixed thr)
  - TREND GATE (load-bearing per my research): long rare DROPS in UPTREND (>200d SMA); short rare POPS in DOWNTREND
  - EVENT HOLD: enter only on the rare trigger, hold H days -> turnover-efficient (few names fire/day)
Honest execution: signal[d] -> trade close[d+1] -> earn d+2 (lag=2), per-side bps on daily turnover, dollar-neutral
gross 2. Test liquid vs relaxed, param grid. Does honest QP reversion clear a deployable bar (and beat naive ~0)?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import DAILY as D                                                             # reuse engine: px, ret, adv, sma200, universe, summ, sr

px, ret, sma200 = D.px, D.ret, D.sma200
def qp_book(short=3, zwin=63, zthr=1.5, H=5, uni="liquid"):
    elig = D.universe(uni)
    m = px/px.shift(short) - 1.0                                               # short move
    z = (m - m.rolling(zwin, min_periods=30).mean()) / (m.rolling(zwin, min_periods=30).std()+1e-9)
    up = px > sma200                                                           # trend gate
    long_trig  = (z < -zthr) & up & elig                                       # rare drop in uptrend -> buy
    short_trig = (z >  zthr) & (~up) & elig                                    # rare pop in downtrend -> sell
    entry = long_trig.astype(float) - short_trig.astype(float)                 # +1 / -1 / 0 on trigger day
    held = entry.rolling(H, min_periods=1).sum()                              # hold H days (event persistence)
    L = held.clip(lower=0); S = held.clip(upper=0)                            # dollar-neutral gross-2 renorm
    Lw = L.div(L.sum(1).replace(0,np.nan), axis=0); Sw = S.div((-S.sum(1)).replace(0,np.nan), axis=0)
    return (Lw.fillna(0.0) + Sw.fillna(0.0))                                   # book, gross ~2

def bt(book, lag=2, cost_bps=5.0):
    gross = (book.shift(lag) * ret).sum(axis=1)
    turn = (book - book.shift(1)).abs().sum(axis=1)
    net = gross - (cost_bps/1e4)*turn.shift(lag)
    return gross.dropna(), net.dropna(), turn.mean()

print("="*94); print("QP REVERSION on daily engine (honest lag=2) — 3-day move, 200d trend gate, event-hold")
print(f"  {'config':40}{'gross SR':>9}{'net SR':>8}{'ann':>7}{'maxDD':>8}{'turn':>7}")
for uni in ("liquid","relaxed"):
    for zthr in (1.0, 1.5, 2.0):
        for H in (5, 10):
            b = qp_book(zthr=zthr, H=H, uni=uni); g,n,tn = bt(b, cost_bps=5.0)
            s,a,dd = D.summ(n)
            print(f"  {uni+f'  z>{zthr}  H={H}d':40}{D.sr(g):>9.2f}{s:>8.2f}{a:>7.1%}{dd:>8.1%}{tn:>7.2f}")
# best config cost sweep + correlation to our momentum alpha
print("\n  cost sweep (liquid, z>1.5, H=5):")
b = qp_book(zthr=1.5, H=5, uni="liquid")
for c in (0,3,5,10):
    g,n,tn = bt(b, cost_bps=c); s,a,dd = D.summ(n); print(f"    {c:>2}bps  net SR {s:>5.2f}  ann {a:>6.1%}  maxDD {dd:>7.1%}")
