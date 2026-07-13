#!/usr/bin/env python3
"""value_hub.py — value re-test on the coherent DataHub (correct split-coherent mcap), liquid vs relaxed,
done right this time (no pre-masking bug). Answers: does value work where it's supposed to (small/mid-cap)?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST, BETANEUT
from DATAHUB import DataHub
hub = DataHub()
me, m_px, mret, synth = hub.me, hub.m_px, hub.mret, hub.synth
tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
BETA = BETANEUT.rolling_beta(mret, hub.elig("liquid"), bw=60)
bm, ep = hub.bm("monthly"), hub.ep("monthly")
def zwin(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
val=(zwin(bm).fillna(0)*bm.notna()+zwin(ep).fillna(0)*ep.notna())/(bm.notna().astype(float)+ep.notna().astype(float)).replace(0,np.nan)
def book_for(mask):
    v=val.where(mask); W=pd.DataFrame(0.0,index=me,columns=m_px.columns)
    for d in me:
        s=v.loc[d].dropna()
        if len(s)<50: continue
        n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    return BETANEUT.betaneut(W,BETA)
def stream(W):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    s=pd.Series(r["returns"]); s.index=pd.DatetimeIndex(s.index); return s[s.index>="2011-01-01"].dropna()
def show(name,x):
    x=x.dropna(); e=(1+x).cumprod()
    print(f"  {name:24} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  maxDD {(e/e.cummax()-1).min():>6.1%}")
    print("    "+" ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2011,2027)))
print("="*92); print("VALUE on coherent DataHub (split-correct mcap), beta-neut — LIQUID vs RELAXED, net 2011+")
show("VALUE liquid", stream(book_for(hub.elig("liquid"))))
show("VALUE relaxed(small/mid)", stream(book_for(hub.elig("relaxed"))))
