#!/usr/bin/env python3
"""FINAL.py — the full deployable book: META(MOM+DM name-sized) + VQ(value/quality, dollar-neutral) -> expanding
ERC -> 60/40 net-long tilt. VQ is dollar-neutral (tested: beta-neut doesn't help it). Full year-by-year vs SPY."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from DATAHUB import DataHub
import BACKTEST, BETANEUT
hub=DataHub(); me,m_px,mret,synth,elig=hub.me,hub.m_px,hub.mret,hub.synth,hub.elig("liquid")
tc=BACKTEST.tiered_transaction_costs(hub.mdv); bf=BACKTEST.tiered_borrow_fees(hub.mdv); SPY=hub.spy_m.dropna()
def net(W):
    W=W.reindex(index=me,columns=m_px.columns).fillna(0.0)
    r=BACKTEST.backtest(W,synth,freq=12,lag=0,signal_dates=[d for d in me if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    s=pd.Series(r["returns"]); s.index=pd.DatetimeIndex(s.index); return s
# META alpha (MOM+DM name-sized)
Wmeta=pickle.load(open("/tmp/meta_weights.pkl","rb")); Wmeta.index=pd.DatetimeIndex(pd.to_datetime(Wmeta.index)); Wmeta=Wmeta[~Wmeta.index.duplicated()]
META=net(Wmeta)
# VQ dollar-neutral from saved score
SCORE=pd.read_pickle("/tmp/fundq_score.pkl"); Wvq=pd.DataFrame(0.0,index=me,columns=m_px.columns)
for d in me:
    s=SCORE.loc[d].dropna()
    if len(s)<80: continue
    n=max(1,int(len(s)*0.10)); Wvq.loc[d,s.nlargest(n).index]=1.0/n; Wvq.loc[d,s.nsmallest(n).index]=-1.0/n
VQ=net(Wvq)
R=pd.DataFrame({"META":META,"VQ":VQ}).dropna()
def erc(c):
    n=c.shape[0]; w=np.ones(n)/n
    for _ in range(500):
        m=c@w; wn=1/np.maximum(m,1e-12); wn/=wn.sum()
        if np.abs(wn-w).max()<1e-11: break
        w=0.5*w+0.5*wn
    return w
W=pd.DataFrame(np.nan,index=R.index,columns=R.columns)
for i in range(24,len(R)): W.iloc[i]=erc(np.cov(R.iloc[:i].T.values)+1e-6*np.eye(2))
ens=(R*W).sum(1)[W.notna().all(1)].dropna(); spy=SPY.reindex(ens.index)
def stat(x,lab):
    x=x.dropna(); e=(1+x).cumprod(); csp=pd.DataFrame({"a":x,"s":spy.reindex(x.index)}).dropna().corr().iloc[0,1]
    print(f"  {lab:22} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}  corr-SPY {csp:>+5.2f}")
    return x
print("="*92); print("FULL BOOK — ERC[META(MOM+DM) + VQ] with 60/40 net-long tilt, vs SPY")
stat(ens,"neutral (50/50)"); f6040=stat(ens+0.2*spy,"60/40 net-long"); stat(spy,"SPY buy&hold")
print(f"\n  avg ERC weights: META {W.dropna().mean()['META']:.2f} / VQ {W.dropna().mean()['VQ']:.2f}")
print("\n  PER-YEAR — 60/40 book vs SPY:")
print(f"  {'year':>5}{'BOOK ret':>10}{'BOOK SR':>9}{'BOOK DD':>9}  |{'SPY ret':>9}{'SPY DD':>8}")
for y in range(2013,2027):
    def ys(s):
        x=s[[pd.Timestamp(d).year==y for d in s.index]].dropna();
        if not len(x): return None
        e=(1+x).cumprod(); return (1+x).prod()-1, (x.mean()/x.std()*np.sqrt(12) if x.std()>0 else np.nan),(e/e.cummax()-1).min()
    b=ys(f6040); sp=ys(spy)
    if b is None: continue
    ss=f"{sp[0]:>+9.0%}{sp[2]:>+8.0%}" if sp else " "*17
    print(f"  {y:>5}{b[0]:>+10.0%}{b[1]:>9.2f}{b[2]:>+9.0%}  |{ss}")
print("  NOTE: 2021+ magnitudes are the tail regime; anchor on 2016-20.")
