#!/usr/bin/env python3
"""target_test.py — test alternative TARGETS for a value ML: does sector-neutralizing and/or trend-scanning the
target improve value's predictability? Compare IC(value -> target) and the value decile L/S across:
  (1) raw fwd return  (2) SECTOR-NEUTRAL fwd return  (3) trend-scan t-stat  (4) sector-neutral trend-scan.
Value is slow -> use 6mo horizon. Sector = 2-digit SIC. If sector-neutral flips value's IC/spread positive, the
target sector-bet was the problem (confirms the pooled-value = sector-bet finding)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub = DataHub(); me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid")
sic = pd.read_parquet("data/edgar/sic.parquet"); sec = pd.Series(sic.set_index("ticker")["sector2"]).reindex(m_px.columns)
bm, ep = hub.bm(), hub.ep()
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
val = (zc(bm).fillna(0)*bm.notna()+zc(ep).fillna(0)*ep.notna())/(bm.notna().astype(float)+ep.notna().astype(float)).replace(0,np.nan)

fwd6 = (m_px.shift(-6)/m_px - 1)                                                # 6-month forward return (value horizon)
# trend-scan t-stat over forward 6mo
H=6; logp=np.log(m_px); xs=np.arange(H); xm=xs.mean(); sxx=((xs-xm)**2).sum()
Tr=pd.DataFrame(np.nan,index=me,columns=m_px.columns)
for i in range(len(me)-H):
    y=logp.iloc[i+1:i+1+H].values; yb=np.nanmean(y,0); sl=np.nansum((xs[:,None]-xm)*(y-yb),0)/sxx
    r=y-(yb+np.outer(xs-xm,sl)); se=np.sqrt(np.nansum(r**2,0)/(H-2)/sxx); Tr.iloc[i]=sl/(se+1e-9)
def sec_neut(panel):                                                           # subtract 2-digit-SIC sector mean each month
    out=panel.copy()
    for d in me:
        row=panel.loc[d].where(elig.loc[d]);
        out.loc[d]=row - row.groupby(sec).transform("mean")
    return out
targets = {"raw fwd6": fwd6, "SECTOR-NEUT fwd6": sec_neut(fwd6), "trend-scan": Tr, "SECTOR-NEUT trend-scan": sec_neut(Tr)}

def ic(sig, tgt):
    cs=[]
    for d in me:
        x=sig.loc[d].where(elig.loc[d]); z=tgt.loc[d]; df=pd.DataFrame({"x":x,"z":z}).dropna()
        if len(df)<80: continue
        cs.append(df["x"].corr(df["z"],method="spearman"))
    a=np.array(cs); return np.nanmean(a), np.nanmean(a)/(np.nanstd(a)/np.sqrt(len(a))+1e-9)
print("="*72); print("VALUE -> alternative TARGETS: IC (does sector-neut / trend-scan help?)")
print(f"  {'target':24}{'IC':>9}{'t-stat':>8}")
for nm,tg in targets.items():
    i,t = ic(val, tg); print(f"  {nm:24}{i:>+9.4f}{t:>+8.1f}")

# does sector-neutral CONSTRUCTION make value tradeable? decile L/S raw vs within-sector rank
fwd1 = mret.shift(-1)
def spread(within_sector):
    sig = val.copy()
    if within_sector:
        for d in me:
            row=val.loc[d].where(elig.loc[d]); sig.loc[d]=row - row.groupby(sec).transform("mean")
    r=[]
    for d in me[:-1]:
        s=sig.loc[d].where(elig.loc[d]).dropna()
        if len(s)<100: continue
        n=max(1,int(len(s)*0.1)); r.append(fwd1.loc[d,s.nlargest(n).index].mean()-fwd1.loc[d,s.nsmallest(n).index].mean())
    x=pd.Series(r).dropna(); return x.mean()*12, x.mean()/x.std()*np.sqrt(12)
print("\n  value decile L/S (long cheap - short expensive), raw vs within-sector:")
a1,s1=spread(False); a2,s2=spread(True)
print(f"    pooled          ann {a1:>+6.1%}  SR {s1:>+5.2f}")
print(f"    within-sector   ann {a2:>+6.1%}  SR {s2:>+5.2f}")
