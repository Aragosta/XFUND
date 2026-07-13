#!/usr/bin/env python3
"""decile_map.py — the user's idea: split EVERY strategy into 10 deciles and map WHERE the edge lives, so we
trade the deciles that pay, not blind top-vs-bottom. Clean forward-excess (fixes the period-mismatch bug):
signal at d, forward return d->d+1, benchmark = equal-weight universe forward return that same period.
Show per-decile annualized excess + t-stat for momentum and value. Hypothesis: momentum edge in EXTREME deciles,
value edge in the MIDDLE (extremes = distress/data poison)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub = DataHub(); me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid")
fwd = mret.shift(-1)                                                            # return realized d -> d+1
bm, ep = hub.bm("monthly"), hub.ep("monthly")
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
value = (zc(bm).fillna(0)*bm.notna()+zc(ep).fillna(0)*ep.notna())/(bm.notna().astype(float)+ep.notna().astype(float)).replace(0,np.nan)
mom = m_px.shift(1)/m_px.shift(12) - 1

def decile_map(sig, nd=10):
    rows = []
    for d in me[:-1]:
        u = elig.loc[d] & fwd.loc[d].notna() & sig.loc[d].notna()
        s = sig.loc[d][u]
        if len(s) < 150: continue
        fr = fwd.loc[d][u]; mkt = fr.mean()                                    # equal-weight universe forward return
        dec = pd.qcut(s.rank(method="first"), nd, labels=False)
        for k in range(nd):
            nm = s.index[dec.values == k]; v = fr.reindex(nm)
            if len(v): rows.append((k, v.mean() - mkt))
    R = pd.DataFrame(rows, columns=["d","ex"])
    ann = R.groupby("d")["ex"].mean()*12
    tst = R.groupby("d")["ex"].apply(lambda x: x.mean()/(x.std()/np.sqrt(len(x))+1e-9))
    return ann, tst

for nm, sig in [("MOMENTUM (12-1): D0=losers .. D9=winners", mom), ("VALUE (B/M+E/P): D0=expensive .. D9=cheap", value)]:
    ann, tst = decile_map(sig)
    print("="*94); print(nm)
    print("  decile:  " + " ".join(f"D{k}" for k in range(10)))
    print("  ann ex%: " + " ".join(f"{ann[k]*100:>+5.1f}" for k in range(10)))
    print("  t-stat:  " + " ".join(f"{tst[k]:>+5.1f}" for k in range(10)))
