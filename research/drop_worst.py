#!/usr/bin/env python3
"""drop_worst.py — the user's idea: instead of SOFT tilt, HARD-strip the worst-performing models each month
(drop bottom quantile by trailing return, equal-weight the survivors). Compare static / soft-tilt / drop-worst
on the SOUND ensemble (momentum + short-reversal + low-vol). Does hard selection beat soft tilt?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2004-01-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z.abs()<0.8)
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig = (m_px>5)&(cov>0.9)&(mdv>5e6); vol12 = mret.rolling(12,min_periods=6).std()
def stream(sig, long_high, q=0.1):
    S=sig.where(elig); R=S.rank(axis=1,pct=True); top=(R>=1-q).astype(float); bot=(R<=q).astype(float)
    long_,short=(top,bot) if long_high else (bot,top)
    L=long_.div(long_.sum(1).replace(0,np.nan),axis=0); Sh=short.div(short.sum(1).replace(0,np.nan),axis=0)
    return ((L-Sh).fillna(0.0).shift(1)*mret).sum(axis=1)
d={f"mom{lb}_{s}_{int(q*100)}": stream(m_px.shift(s)/m_px.shift(lb)-1,True,q) for lb in (3,6,9,12) for s in (0,1) for q in (0.1,0.2)}
d.update({f"rev{lb}": stream(m_px.shift(0)/m_px.shift(lb)-1,False) for lb in (1,2,3)})
d.update({f"lowvol{q}": stream(vol12,False,q) for q in (0.1,0.2)})
STR=pd.DataFrame(d).loc["2005-01-01":].dropna(how="all"); n=STR.shape[1]
print(f"[ensemble] {n} sound models (momentum+reversal+lowvol)")
def perf(c):
    x=c[c.index>="2005-06-01"].dropna(); e=(1+x).cumprod(); return x.mean()/x.std()*np.sqrt(12),(e/e.cummax()-1).min()
def alloc(mode, keep=0.5, win=6, lam=10, warm=24):
    W=pd.DataFrame(np.nan,index=STR.index,columns=STR.columns)
    for i in range(warm,len(STR)):
        H=STR.iloc[:i]; cr=(1+H.iloc[-win:]).prod().values-1
        if mode=="equal": w=np.ones(n)/n
        elif mode=="tilt": e=np.exp(lam*(cr-np.nanmean(cr))); w=e/np.nansum(e)
        elif mode=="drop":                                                     # keep top `keep` by trailing return
            thr=np.nanquantile(cr, 1-keep); m=(cr>=thr).astype(float); w=m/m.sum()
        W.iloc[i]=w
    c=(STR*W).sum(axis=1); return c[W.notna().all(axis=1)]
print(f"\n  {'allocation':26}{'SR':>6}{'maxDD':>8}")
print(f"  {'static equal':26}{perf(alloc('equal'))[0]:>6.2f}{perf(alloc('equal'))[1]:>8.1%}")
print(f"  {'soft tilt (lam10)':26}{perf(alloc('tilt'))[0]:>6.2f}{perf(alloc('tilt'))[1]:>8.1%}")
for keep in (0.75,0.5,0.25,0.1):
    s,dd=perf(alloc('drop',keep=keep)); print(f"  {'drop-worst keep top '+str(int(keep*100))+'%':26}{s:>6.2f}{dd:>8.1%}")
