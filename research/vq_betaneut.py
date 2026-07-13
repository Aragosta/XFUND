"""vq_betaneut.py — does the VQ (value/quality) book benefit from BETA-NEUTRALIZATION vs plain dollar-neutral?
Build the decile L/S book from the saved VQ score both ways; compare net SR / maxDD / skew / corr-to-market."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import skew
from DATAHUB import DataHub
import BACKTEST, BETANEUT
hub=DataHub(); me,m_px,mret,synth,elig=hub.me,hub.m_px,hub.mret,hub.synth,hub.elig("liquid")
tc=BACKTEST.tiered_transaction_costs(hub.mdv); bf=BACKTEST.tiered_borrow_fees(hub.mdv)
BETA=BETANEUT.rolling_beta(mret,elig,bw=60); mkt=mret.where(elig).mean(1)
SCORE=pd.read_pickle("/tmp/fundq_score.pkl")
def decile(SCORE):
    W=pd.DataFrame(0.0,index=me,columns=m_px.columns)
    for d in me:
        s=SCORE.loc[d].dropna()
        if len(s)<80: continue
        n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    return W
Wdn=decile(SCORE); Wbn=BETANEUT.betaneut(Wdn,BETA)
def stat(W,lab):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); x=x[(x.index>="2016-01-01")&(x.index<"2027-01-01")].dropna()
    e=(1+x).cumprod(); cm=pd.DataFrame({"a":x,"m":mkt.reindex(x.index)}).dropna().corr().iloc[0,1]
    print(f"  {lab:22} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}  skew {skew(x):>+5.2f}  corr-mkt {cm:>+5.2f}")
    return x
print("="*84); print("VQ: DOLLAR-NEUTRAL vs BETA-NEUTRAL (2016-2026)")
a=stat(Wdn,"dollar-neutral only"); b=stat(Wbn,"beta-neutralized")
