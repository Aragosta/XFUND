#!/usr/bin/env python3
"""trendscan_value.py — quick test: is a TREND-SCANNING target (de Prado: t-stat of forward log-price~time slope)
a good target for a VALUE ML? (1) is the trend-scan label correlated with future return? (2) does value predict
it, and does that hold in BULL years (where plain value fails)? If value->trend IC is more bull-robust than
value->raw-return, the trend-scan label is the better (denoised) target for value ML."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub = DataHub(); me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid")
H = 6                                                                            # forward window (months)
logp = np.log(m_px)
xs = np.arange(H); xm = xs.mean(); sxx = ((xs - xm) ** 2).sum()
T = pd.DataFrame(np.nan, index=me, columns=m_px.columns)                         # trend-scan target (signed slope t-stat)
for i in range(len(me) - H):
    y = logp.iloc[i+1:i+1+H].values                                             # forward H months, HxN
    ybar = np.nanmean(y, axis=0); slope = np.nansum((xs[:,None]-xm)*(y-ybar), axis=0)/sxx
    resid = y - (ybar + np.outer(xs-xm, slope)); se = np.sqrt(np.nansum(resid**2,axis=0)/(H-2)/sxx)
    T.iloc[i] = slope/(se+1e-9)
fwdH = (m_px.shift(-H)/m_px - 1)                                                 # forward H-month return
fwd1 = mret.shift(-1)                                                            # forward 1-month return
bm, ep = hub.bm(), hub.ep()
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
val = (zc(bm).fillna(0)*bm.notna()+zc(ep).fillna(0)*ep.notna())/(bm.notna().astype(float)+ep.notna().astype(float)).replace(0,np.nan)

def xs_corr(a, b, mask):                                                        # mean cross-sectional Spearman
    cs=[]
    for d in me:
        x=a.loc[d].where(mask.loc[d]); z=b.loc[d]; df=pd.DataFrame({"x":x,"z":z}).dropna()
        if len(df)<80: continue
        cs.append((d, df["x"].corr(df["z"], method="spearman")))
    s=pd.Series(dict(cs)); return s
# bull/bear years by equal-weight market
mkt = mret.where(elig).mean(1); yr_ret = mkt.groupby(mkt.index.year).apply(lambda x:(1+x).prod()-1)
bull_years = set(yr_ret[yr_ret>0.05].index); bear_years = set(yr_ret[yr_ret<=0.05].index)
print("="*80); print(f"TREND-SCAN TARGET (forward {H}mo slope t-stat) — quick test")
c1 = xs_corr(T, fwdH, elig).mean(); print(f"\n(1) corr(trend-scan target, forward {H}mo return) = {c1:+.2f}   -> valid return proxy? {'YES' if c1>0.5 else 'weak'}")
icT = xs_corr(val, T, elig).dropna(); icR = xs_corr(val, fwd1, elig).dropna()
icT = icT[icT.index.year>=2011]; icR = icR[icR.index.year>=2011]                 # fundamentals era
def yrmean(s, yrs):
    m = s[[pd.Timestamp(d).year in yrs for d in s.index]]; return m.mean() if len(m) else float("nan")
print(f"\n(2) does VALUE predict each target, and in BULL years?")
print(f"  {'target':22}{'IC all':>9}{'IC bull':>9}{'IC bear':>9}")
print(f"  {'value -> trend-scan':22}{icT.mean():>+9.3f}{yrmean(icT,bull_years):>+9.3f}{yrmean(icT,bear_years):>+9.3f}")
print(f"  {'value -> raw return':22}{icR.mean():>+9.3f}{yrmean(icR,bull_years):>+9.3f}{yrmean(icR,bear_years):>+9.3f}")
print(f"\n  bull years: {sorted(bull_years)}")
