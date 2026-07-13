"""netlong.py — tilt the market-neutral 3-sleeve ERC net-LONG to capture the equity premium. A 60/40 long book =
neutral alpha + a net-long market position (beta). Test net exposures (50/50 neutral, 60/40, 67/33, 75/25) and a
VOL-MANAGED beta version (size the market leg by inverse vol). Stats + year-by-year vs SPY buy-and-hold."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from DATAHUB import DataHub
hub=DataHub()
def S(p,k=None):
    o=pickle.load(open(p,"rb")); s=pd.Series(dict(o) if isinstance(o,list) else (o[k] if k else o)).dropna(); s.index=pd.DatetimeIndex(s.index); return s
MOM=S("/tmp/mom_champ.pkl","n1"); DM=S("/tmp/dm_returns.pkl"); VQ=S("/tmp/fundq_returns.pkl"); SPY=hub.spy_m.dropna()
R=pd.DataFrame({"MOM":MOM,"DM":DM,"VQ":VQ}).dropna()
def erc(c):
    n=c.shape[0]; w=np.ones(n)/n
    for _ in range(500):
        m=c@w; wn=1/np.maximum(m,1e-12); wn/=wn.sum()
        if np.abs(wn-w).max()<1e-11: break
        w=0.5*w+0.5*wn
    return w
W=pd.DataFrame(np.nan,index=R.index,columns=R.columns)
for i in range(24,len(R)): W.iloc[i]=erc(np.cov(R.iloc[:i].T.values)+1e-6*np.eye(3))
ens=(R*W).sum(1)[W.notna().all(1)].dropna()
spy=SPY.reindex(ens.index)
# vol-managed market leg: target 10% ann vol, leak-free trailing
fv=spy.rolling(6,min_periods=3).std().shift(1)*np.sqrt(12); vmspy=(0.10/fv).clip(0.2,3.0).fillna(1.0)*spy
def stat(x,lab):
    x=x[(x.index>="2013-06-01")].dropna(); e=(1+x).cumprod(); dd=(e/e.cummax()-1).min()
    sr=x.mean()/x.std()*np.sqrt(12); ann=(1+x).prod()**(12/len(x))-1; csp=pd.DataFrame({"a":x,"s":spy.reindex(x.index)}).dropna().corr().iloc[0,1]
    print(f"  {lab:22} SR {sr:>5.2f}  ann {ann:>6.1%}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {dd:>6.1%}  corr-SPY {csp:>+5.2f}")
    return x
print("="*90); print("NET-LONG TILT of the 3-sleeve ERC (neutral alpha + market beta) vs SPY")
variants={"50/50 neutral":0.0,"60/40 (net+0.2)":0.2,"67/33 (net+0.35)":0.35,"75/25 (net+0.5)":0.5}
cols={}
for lab,b in variants.items(): cols[lab]=stat(ens+b*spy,lab)
cols["60/40 VOL-MANAGED"]=stat(ens+0.2*(vmspy/spy.std()*spy.std()) if False else ens+0.4*vmspy,"60/40 vol-managed")
stat(spy,"SPY buy&hold")
print("\n  PER-YEAR RETURNS (%): neutral / 60-40 / 75-25 / SPY")
for y in range(2013,2027):
    def yr(s): x=s[[pd.Timestamp(d).year==y for d in s.index]].dropna(); return f"{(1+x).prod()-1:>+5.0%}" if len(x) else "   —"
    print(f"    {y}   {yr(ens)} / {yr(ens+0.2*spy)} / {yr(ens+0.5*spy)} / {yr(spy)}")
