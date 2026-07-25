#!/usr/bin/env python3
"""horse_race_v2.py — "combine all the signals (returns, frog-in-the-pan, ...) and rank" — RE-TESTED.

T26 (2026-07-23) already ran exactly this and it FAILED: 6 raw momentum signals were 0.86-0.99 book-correlated,
the equal-weight composite (0.49 SR) was WORSE than the best single signal (tvalpast, 0.52), and the ML matched
but did not beat one raw signal. Conclusion at the time: "a LIBRARY of momentum variants adds nothing".

SO WHY RE-TEST? Because every number in T26 came off an engine we have since shown to be wrong or suboptimal
in four ways, all of which bear on SLOW signals like frog-in-the-pan specifically:
  1. COSTS were the old dollar-volume tiers — 150 bp for the median name where measurement says ~25 bp (7x too
     punitive). Cost errors penalise signals in proportion to turnover, so they distort the RANKING of signals.
  2. HOLD=1. We now know holding 6-12 months converts -12.45% net alpha into +1.46% at unchanged gross alpha.
  3. UNIVERSE was top-1000. Size attribution now shows the best NET bucket is Q3 (mid), net alpha +9.67% t=2.72.
  4. The ML arm used the RETURN criterion; Han's SHARPE criterion (§3.3.5) is materially better on the long side.
A signal that lost under a 150bp/hold-1 engine can win under a 25bp/hold-6 one. That is a real possibility for
frog-in-the-pan, whose whole premise (Da-Gurun-Warachka) is SLOW, continuous information — a low-turnover signal.

SIGNALS (all long-short decile, same construction, same universe, same costs):
  mom11     11-month price momentum (the plain benchmark)
  tvalpast  slope/SE of the past 6m log price — the champion target computed BACKWARD, free, no ML
  resmom    residual momentum (Blitz), 11-1 on market residuals / their vol
  hi52      52-week-high proximity (George-Hwang anchoring)
  fip       FROG IN THE PAN: z(mom) - z(information discreteness), ID = sgn(mom)*(%neg - %pos)
  composite equal-weight z-blend of all five ("let the portfolio sort it out")
  ML_sharpe the Han-DM DNN with the Sharpe reclassification
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST
from dm_criteria import scores, hub, pnl, sm, spy
from dm_capacity import book
from ibkr_costs import ibkr_cost_panel, borrow_panel, cost_at_aum

tc = ibkr_cost_panel(hub); bf = borrow_panel(hub)
adv = sm.rolling(3, min_periods=1).mean()
pm, rm = hub.delisted_prices("monthly"), hub.clean_returns("monthly")
ML = scores("sharpe")
DATES = sorted(ML)


def z(x):
    x = np.asarray(x, float); return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)


def build_signals():
    MOM = (pm.shift(1) / pm.shift(12) - 1).where(lambda v: v.abs() < 5)
    HI52 = pm / pm.rolling(12, min_periods=8).max() - 1
    mkt = rm.mean(axis=1)
    bcov = rm.mul(mkt, axis=0).rolling(36, min_periods=24).mean() - rm.rolling(36, min_periods=24).mean().mul(
        mkt.rolling(36, min_periods=24).mean(), axis=0)
    bvar = (mkt ** 2).rolling(36, min_periods=24).mean() - mkt.rolling(36, min_periods=24).mean() ** 2
    RES = rm.sub((bcov.div(bvar, axis=0).shift(1)).mul(mkt, axis=0))
    RESM = RES.shift(1).rolling(11, min_periods=8).sum() / (RES.rolling(11, min_periods=8).std().shift(1) + 1e-9)
    lp = np.log(pm.values); nP = 7; xc = np.arange(nP) - (nP - 1) / 2.0; Sxx = (xc ** 2).sum()
    TVP = pd.DataFrame(np.nan, index=pm.index, columns=pm.columns)
    for t in range(nP - 1, len(pm)):
        Y = lp[t - nP + 1:t + 1]; Ym = np.nanmean(Y, 0)
        sl = np.nansum(xc[:, None] * (Y - Ym), 0) / Sxx
        resd = Y - (Ym + xc[:, None] * sl)
        TVP.iloc[t] = sl / (np.sqrt(np.nansum(resd ** 2, 0) / max(nP - 2, 1) / Sxx) + 1e-9)
    fpos = (rm > 0).rolling(11, min_periods=8).mean(); fneg = (rm < 0).rolling(11, min_periods=8).mean()
    ID = np.sign(MOM) * (fneg - fpos)                      # LOW ID = continuous information = frog in the pan
    return dict(mom11=MOM, tvalpast=TVP, resmom=RESM, hi52=HI52), MOM, ID


RAW, MOM, ID = build_signals()
S = {}
for nm, DF in RAW.items():
    S[nm] = {d: DF.shift(1).loc[d].reindex(ML[d].index) for d in DATES if d in DF.index}
S["fip"] = {d: pd.Series(z(MOM.shift(1).loc[d].reindex(ML[d].index).values)
                         - z(ID.shift(1).loc[d].reindex(ML[d].index).values), index=ML[d].index)
            for d in DATES if d in MOM.index}
parts = ["mom11", "tvalpast", "resmom", "hi52", "fip"]
S["composite"] = {d: pd.Series(np.nanmean(np.vstack([z(S[p][d].values) for p in parts if d in S[p]]), axis=0),
                               index=ML[d].index) for d in DATES if all(d in S[p] for p in parts)}
S["ML_sharpe"] = ML


def rep(tag, W, aum=0):
    t_ = cost_at_aum(hub, W, aum) if aum else tc
    out = []
    for kw in (dict(transaction_cost=0.0, borrow_fee=0.0), dict(transaction_cost=t_, borrow_fee=bf)):
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index), **kw)
        x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
        if len(x) < 24: return None
        x.index = pd.PeriodIndex(x.index, freq="M")
        D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
        X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
        e = D.r.values - X @ c
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
        eq = (1 + D.r).cumprod()
        out.append((D.r.mean() * 12, D.r.std() * np.sqrt(12), c[0] * 12, c[0] / (se[0] + 1e-12),
                    (eq / eq.cummax() - 1).min(), r["ann_turnover"], D.r))
    (ga, gv, gal, gt, _, _, _), (na, nv, nal, nt, ndd, turn, ser) = out
    print(f"  {tag:26}{gal:>+9.2%}{gt:>6.2f} |{na:>8.1%}{na/(nv+1e-9):>6.2f}{ndd:>8.1%}"
          f"{nal:>+9.2%}{nt:>6.2f}{turn:>6.1f}", flush=True)
    return ser


if __name__ == "__main__":
    q = adv.rank(axis=1, pct=True); Q3 = (q > 0.4) & (q <= 0.6)
    order = parts + ["composite", "ML_sharpe"]
    H = f"  {'signal':26}{'gALPHA':>9}{'t':>6} |{'nANN':>8}{'nSR':>6}{'nDD':>8}{'nALPHA':>9}{'t':>6}{'turn':>6}"

    print("\n[A] Q3 (mid-cap) · hold=6 · band · MEASURED IBKR costs — the corrected engine"); print(H)
    rets = {}
    for nm in order:
        s = rep(nm, book(S[nm], uni=Q3))
        if s is not None: rets[nm] = s

    print("\n[B] same signals, hold=1 (T26's setting) — does holding period change the RANKING?"); print(H)
    for nm in order:
        rep(nm, book(S[nm], uni=Q3, hold=1, band=None))

    print("\n[C] REDUNDANCY — net book-return correlation (T26 found 0.86-0.99 = one bet)")
    R = pd.DataFrame(rets).dropna()
    print(R.corr().round(2).to_string())

    print("\n[D] CAPACITY of the best arms on Q3"); print(H)
    for nm in ("ML_sharpe", "composite", "tvalpast"):
        for aum in (1e6, 1e7, 5e7, 1e8):
            rep(f"{nm} @ ${aum/1e6:,.0f}M", book(S[nm], uni=Q3), aum=aum)
