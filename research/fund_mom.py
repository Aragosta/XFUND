#!/usr/bin/env python3
"""fund_mom.py — FUNDAMENTAL MOMENTUM ('trend on a different dataset'): trend/acceleration of the BUSINESS
(earnings, revenue) instead of the price. Structurally more orthogonal to PRICE momentum than a return-target
model. Signal = sector-neutral rank of earnings-growth + revenue-growth + growth-ACCELERATION. Decile L/S,
beta-neut, net. Measure corr to price-MOM / DM / FUND — is it the orthogonal diversifier residualization couldn't give?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
from DATAHUB import DataHub
import BACKTEST, BETANEUT
hub = DataHub(); me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
sec = hub.sector; tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60); f = hub.fund
ni, rev = f("ni"), f("rev")
g_ni = ni/ni.shift(12) - 1; g_rev = rev/rev.shift(12) - 1                       # YoY growth (fundamental trend)
acc_ni = g_ni - g_ni.shift(12); acc_rev = g_rev - g_rev.shift(12)              # growth ACCELERATION (2nd derivative)
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
# fundamental-momentum composite (growth + acceleration)
fmom = zc(g_ni).fillna(0)+zc(g_rev).fillna(0)+zc(acc_ni).fillna(0)+zc(acc_rev).fillna(0)
W = pd.DataFrame(0.0, index=me, columns=m_px.columns)
for d in me:
    s = fmom.loc[d].where(elig.loc[d] & ni.loc[d].notna())
    s = (s - s.groupby(sec).transform("mean")).dropna()                        # sector-neutral
    if len(s) < 80: continue
    n=max(1,int(len(s)*0.10)); W.loc[d, s.nlargest(n).index]=1.0/n; W.loc[d, s.nsmallest(n).index]=-1.0/n
W = BETANEUT.betaneut(W, BETA)
r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)
x = pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index)
xx=x[(x.index>="2013-01-01")&(x.index<"2027-01-01")].dropna(); e=(1+xx).cumprod()
print("="*70); print("FUNDAMENTAL MOMENTUM (earnings/revenue trend+acceleration, sector-neut) — net")
print(f"  FUND-MOM  SR {xx.mean()/xx.std()*np.sqrt(12):.2f}  ann {(1+xx).prod()**(12/len(xx))-1:.1%}  maxDD {(e/e.cummax()-1).min():.1%}")
print("    "+" ".join(f"{y}:{(1+xx[[d.year==y for d in xx.index]]).prod()-1:>+4.0%}" for y in range(2013,2027)))
for nm,pth,key in [("price-MOM","/tmp/mom_champ.pkl","n1"),("DM","/tmp/dm_returns.pkl",None),("FUND(value)","/tmp/fund_returns.pkl",None)]:
    if os.path.exists(pth):
        o=pickle.load(open(pth,"rb")); s=pd.Series(o[key] if key else o); s.index=pd.DatetimeIndex(s.index)
        print(f"  corr(FUND-MOM, {nm}) = {pd.DataFrame({'a':x,'b':s}).dropna().corr().iloc[0,1]:+.2f}")
