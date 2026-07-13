#!/usr/bin/env python3
"""quantile_edge.py — WHERE does a signal's edge live? Standard diagnostic: bucket into 5 quantiles each month,
show mean forward EXCESS-over-market return per quantile, and compare extreme (decile) vs broad (tercile) L/S.
Lesson (value taught us): some signals' edge is BROAD (extreme quantile is poison — distress/data traps);
others (momentum) reward CONVICTION (edge grows toward the extreme). Never assume decile-L/S is the right lens."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub = DataHub(); me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid")
fwd = mret.shift(-1); mkt = mret.where(elig).mean(1)
bm, ep = hub.bm("monthly"), hub.ep("monthly")
def zc(df): z = df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
value = (zc(bm).fillna(0)*bm.notna()+zc(ep).fillna(0)*ep.notna())/(bm.notna().astype(float)+ep.notna().astype(float)).replace(0,np.nan)
mom = m_px.shift(1)/m_px.shift(12) - 1

def quantile_profile(sig, nq=5):
    rows = []
    for d in me:
        s = sig.loc[d].where(elig.loc[d]).dropna()
        if len(s) < 100 or not np.isfinite(fwd.loc[d]).any(): continue
        q = pd.qcut(s.rank(method="first"), nq, labels=False)                  # rank-first avoids duplicate-edge NaN
        ex = fwd.loc[d] - mkt.loc[d]
        for k in range(nq):
            nm = s.index[q.values == k]; v = ex.reindex(nm).dropna()
            if len(v): rows.append((k, v.mean()))
    R = pd.DataFrame(rows, columns=["q","ex"]).groupby("q")["ex"].mean()*12
    return R
def ls_sr(sig, q):                                                             # top-q minus bottom-q L/S, raw, net-of-nothing
    r = []
    for d in me:
        s = sig.loc[d].where(elig.loc[d]).dropna()
        if len(s) < 100: continue
        n = max(1,int(len(s)*q)); r.append((d, fwd.loc[d,s.nlargest(n).index].mean() - fwd.loc[d,s.nsmallest(n).index].mean()))
    x = pd.Series(dict(r)).dropna(); return x.mean()/x.std()*np.sqrt(12), x.mean()*12

for nm, sig in [("VALUE (cheap=high)", value), ("MOMENTUM (12-1)", mom)]:
    print("="*70); print(f"{nm}: mean fwd EXCESS-over-market return by quintile (Q0=low signal .. Q4=high)")
    R = quantile_profile(sig)
    print("  " + "   ".join(f"Q{k}:{v:>+5.1%}" for k,v in R.items()))
    print(f"  monotone spread Q4-Q0: {R.iloc[-1]-R.iloc[0]:+.1%}/yr   | edge location: "
          + ("EXTREME (grows to tail)" if abs(R.iloc[-1])>abs(R.iloc[-2]) and abs(R.iloc[0])>abs(R.iloc[1]) else "BROAD (extreme not richest)"))
    for q,lbl in [(0.10,"decile 10%"),(0.20,"quintile 20%"),(0.33,"tercile 33%")]:
        sr,ann = ls_sr(sig,q); print(f"    L/S {lbl:12} SR {sr:>+5.2f}  ann {ann:>+6.1%}")
