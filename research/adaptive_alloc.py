#!/usr/bin/env python3
"""adaptive_alloc.py — test the user's idea: dynamically re-weight sleeves by a parameter solved in a meta layer,
optimizing TRAILING Sharpe (a quarter / few months). This is strategy/factor momentum, NOT regime switching.
Leak-free: weights at t come from data strictly < t. Rules: static ERC (baseline), rolling ERC, rolling max-Sharpe
(mean-variance), Sharpe-tilt, return-momentum tilt. Lookbacks 3/6/12/18 mo. Does adaptation beat static OOS?"""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd

M = pickle.load(open("/tmp/mom_champ.pkl","rb")); MOM=pd.Series(M["n1"]).dropna(); MOM.index=pd.DatetimeIndex(MOM.index)
D = pickle.load(open("/tmp/dm_returns.pkl","rb")); DM=pd.Series(D).dropna(); DM.index=pd.DatetimeIndex(DM.index)
R = pd.DataFrame({"MOM":MOM,"DM":DM}).dropna(); n = R.shape[1]
print(f"[streams] {len(R)} months {R.index[0].date()}..{R.index[-1].date()}   corr(MOM,DM)={R.corr().iloc[0,1]:+.2f}")

def erc(cov):
    w=np.ones(n)/n
    for _ in range(500):
        mrc=cov@w; wn=(1.0/np.maximum(mrc,1e-12)); wn/=wn.sum()
        if np.abs(wn-w).max()<1e-10: break
        w=0.5*w+0.5*wn
    return w
def alloc(rule, win):
    W=pd.DataFrame(np.nan,index=R.index,columns=R.columns)
    warm=max(win,24)
    for i in range(warm,len(R)):
        H=R.iloc[i-win:i] if rule!="static" else R.iloc[:i]                  # rolling vs expanding
        mu=H.mean().values; cov=np.cov(H.T.values); sd=H.std().values
        if rule=="static" or rule=="ercroll":
            w=erc(cov)
        elif rule=="maxsharpe":
            w=np.linalg.pinv(cov+1e-6*np.eye(n))@mu; w=np.clip(w,0,None); w=w/w.sum() if w.sum()>0 else np.ones(n)/n
        elif rule=="sharpetilt":
            sh=np.clip(mu/ (sd+1e-9),0,None); w=sh/sh.sum() if sh.sum()>0 else np.ones(n)/n
        elif rule=="retmom":
            cr=(1+H).prod().values-1; e=np.exp(3*(cr-cr.mean())); w=e/e.sum()
        W.iloc[i]=w
    return W
def combine(W):
    c=(R*W).sum(axis=1); c=c[W.notna().all(axis=1)]; return c
def perf(c,lab):
    for w0,w1,tag in [("2000-01-01","2023-01-01","full-2000-2022"),("2011-06-01","2023-01-01","2011-2022")]:
        x=c[(c.index>=w0)&(c.index<w1)].dropna()
        sr=x.mean()/x.std()*np.sqrt(12); ann=(1+x).prod()**(12/len(x))-1
        e=(1+x).cumprod(); dd=(e/e.cummax()-1).min()
        print(f"    {lab:26} [{tag:14}] SR {sr:>5.2f}  ann {ann:>6.1%}  maxDD {dd:>6.1%}")

print("\nSTATIC ERC baseline:"); perf(combine(alloc("static",0)),"static ERC (expanding)")
for rule in ["ercroll","sharpetilt","retmom","maxsharpe"]:
    print(f"\n{rule}:")
    for win in ([3,6,12,18] if rule!="ercroll" else [12,24,36]):
        perf(combine(alloc(rule,win)), f"{rule} win={win}")
