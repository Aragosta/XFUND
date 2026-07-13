#!/usr/bin/env python3
"""erc_return_tilt.py — tilt by trailing RETURN instead of Sharpe. w_i ∝ ERC_i * exp(λ*(ret_i - mean ret)).
Return-tilt is NOT vol-normalized -> leans harder into high-raw-return sleeves (e.g. BETA) than Sharpe-tilt.
Sweep λ, 2- and 3-sleeve, vs pure ERC (λ=0). Also a pure return-proportional (non-ERC) softmax for contrast."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from adapt import load_streams, erc_w

R3 = load_streams(); R2 = R3[["MOM","DM"]].dropna()
def tilt(R, lam, by="ret", anchor=True, win=12, warm=24):
    W=pd.DataFrame(np.nan,index=R.index,columns=R.columns); n=R.shape[1]
    for i in range(warm,len(R)):
        H=R.iloc[:i]; base=erc_w(np.cov(H.T.values)+1e-6*np.eye(n)) if anchor else np.ones(n)/n
        Hw=H.iloc[-win:]
        m=(1+Hw).prod().values-1 if by=="ret" else Hw.mean().values/(Hw.std().values+1e-9)
        w=base*np.exp(lam*(m-m.mean())); w=w/w.sum(); W.iloc[i]=w
    c=(R*W).sum(axis=1); return c[W.notna().all(axis=1)]
def perf(c):
    x=c[(c.index>="2011-06-01")&(c.index<"2027-01-01")].dropna(); e=(1+x).cumprod()
    return x.mean()/x.std()*np.sqrt(12),(e/e.cummax()-1).min()

for tag,R in [("2-sleeve MOM+DM",R2),("3-sleeve +BETA",R3)]:
    print(f"\n{tag} (2011-2026):    {'λ':>6} {'RET-tilt SR':>12} {'(maxDD)':>9}   {'SHARPE-tilt SR':>14}")
    for lam in [0.0,0.25,0.5,1.0,2.0,4.0]:
        rs,rd=perf(tilt(R,lam,"ret")); ss,_=perf(tilt(R,lam,"sharpe"))
        tag2=" <- pure ERC" if lam==0 else ""
        print(f"  {'':22}{lam:>6.2f} {rs:>12.2f} {rd:>9.1%}   {ss:>14.2f}{tag2}")
