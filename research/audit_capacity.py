"""audit_capacity.py — THE production question: does the alpha survive in TRADEABLE names, or does it live in
illiquid microcaps? Restrict the VQ book to the top-N most liquid names each month and watch SR/ann decay.
Also: what fraction of the alpha comes from the smallest decile of the eligible universe?
Run on the HONEST engine (squeezes paid for)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
import BACKTEST
hub=DataHub(); me,m_px,mret,synth,elig=hub.me,hub.m_px,hub.mret,hub.synth,hub.elig("liquid")
tc=BACKTEST.tiered_transaction_costs(hub.mdv); bf=BACKTEST.tiered_borrow_fees(hub.mdv)
SCORE=pd.read_pickle("/tmp/fundq_score.pkl"); mdv=hub.mdv.where(elig)
def book(topN):
    W=pd.DataFrame(0.0,index=me,columns=m_px.columns)
    for d in me:
        s=SCORE.loc[d].dropna()
        if topN is not None:
            liq=mdv.loc[d].dropna().nlargest(topN).index; s=s[s.index.isin(liq)]
        if len(s)<60: continue
        n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    return W
def run(W,lab):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=1,signal_dates=[d for d in me if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); x=x[(x.index>="2016-01-01")&(x.index<"2027-01-01")].dropna()
    e=(1+x).cumprod()
    print(f"  {lab:26} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>7.1%}  maxDD {(e/e.cummax()-1).min():>7.1%}  turn {r['ann_turnover']:>4.1f}")
    return x
print("="*94); print("CAPACITY AUDIT — does the alpha survive in TRADEABLE names? (VQ, honest engine, 2016-26)")
print(f"  avg eligible universe: {int(elig.sum(1).mean())} names/mo")
for N,lab in [(None,"all eligible (~3700)"),(1500,"top-1500 liquid"),(1000,"top-1000 liquid"),(500,"top-500 liquid"),(250,"top-250 liquid")]:
    run(book(N), lab)
print("\n  -> If SR collapses as we restrict to liquid names, the edge lives in microcaps = NOT deployable.")
