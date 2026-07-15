"""audit_capacity_all.py — does MOM / DM alpha survive in TRADEABLE names, or (like VQ) does it live in
microcaps? Take each sleeve's decile book, keep only positions in the top-N most liquid names, renormalise
dollar-neutral, and backtest on the HONEST engine (squeezes paid). If SR collapses, it isn't deployable."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from DATAHUB import DataHub
import BACKTEST
hub=DataHub(); me,m_px,synth,elig=hub.me,hub.m_px,hub.synth,hub.elig("liquid")
tc=BACKTEST.tiered_transaction_costs(hub.mdv); bf=BACKTEST.tiered_borrow_fees(hub.mdv); mdv=hub.mdv.where(elig)
def load(p):
    W=pickle.load(open(p,"rb")); W.index=pd.DatetimeIndex(W.index); return W.reindex(index=me,columns=m_px.columns).fillna(0.0)
def restrict(W,topN):
    if topN is None: return W
    out=pd.DataFrame(0.0,index=me,columns=m_px.columns)
    for d in me:
        w=W.loc[d]
        if w.abs().sum()<1e-9: continue
        liq=set(mdv.loc[d].dropna().nlargest(topN).index)
        w=w[[c in liq for c in w.index]] if False else w.where(pd.Series([c in liq for c in w.index],index=w.index),0.0)
        L=w.clip(lower=0); S=w.clip(upper=0); gl,gs=L.sum(),-S.sum()
        if gl<1e-9 or gs<1e-9: continue
        out.loc[d]=L/gl + S/gs                                   # renormalise dollar-neutral, gross 2
    return out
def run(W,lab):
    sd=[d for d in me if W.loc[d].abs().sum()>1e-9]
    if not sd: print(f"  {lab:26} (empty)"); return
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=1,signal_dates=sd,transaction_cost=tc,borrow_fee=bf)
    x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); x=x[(x.index>="2016-01-01")&(x.index<"2027-01-01")].dropna()
    e=(1+x).cumprod()
    print(f"  {lab:26} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>7.1%}  maxDD {(e/e.cummax()-1).min():>7.1%}")
print("="*92); print("CAPACITY AUDIT — MOM & DM in tradeable names (honest engine, squeezes paid, 2016-26)")
for nm,pth in [("MOM","/tmp/mom_weights.pkl"),("DM","/tmp/dm_weights.pkl")]:
    print(f"\n  {nm}:")
    W=load(pth)
    for N,lab in [(None,"all eligible"),(1500,"top-1500 liquid"),(1000,"top-1000 liquid"),(500,"top-500 liquid")]:
        run(restrict(W,N), f"{nm} {lab}")
