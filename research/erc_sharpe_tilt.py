#!/usr/bin/env python3
"""erc_sharpe_tilt.py — the user's ERC x Sharpe-tilt: keep ERC's risk-parity base, tilt MULTIPLICATIVELY by
trailing Sharpe. w_i ∝ ERC_i * exp(λ*(SR_i - mean SR)). λ=0 -> pure static ERC; λ>0 -> lean toward recent winners
but ANCHORED to risk-parity (unlike raw tilt which abandons it). Sweep λ, leak-free, on 2-sleeve (MOM+DM) AND
3-sleeve (+ orthogonal vol-managed BETA). Does an ERC-anchored tilt beat pure ERC where the raw tilt didn't?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from adapt import load_streams, erc_w

R3 = load_streams(); R2 = R3[["MOM","DM"]].dropna()

def erc_sharpe(R, lam, win=12, warm=24):
    W=pd.DataFrame(np.nan,index=R.index,columns=R.columns); n=R.shape[1]
    for i in range(warm,len(R)):
        H=R.iloc[:i]; base=erc_w(np.cov(H.T.values)+1e-6*np.eye(n))                  # ERC on full history
        Hw=H.iloc[-win:]; sr=Hw.mean().values/(Hw.std().values+1e-9); sr=sr-sr.mean()
        w=base*np.exp(lam*sr); w=w/w.sum()                                           # multiplicative Sharpe tilt, renorm
        W.iloc[i]=w
    c=(R*W).sum(axis=1); return c[W.notna().all(axis=1)]
def perf(c):
    x=c[(c.index>="2011-06-01")&(c.index<"2027-01-01")].dropna(); e=(1+x).cumprod()
    return x.mean()/x.std()*np.sqrt(12), (e/e.cummax()-1).min()

for tag,R in [("2-sleeve MOM+DM",R2),("3-sleeve +BETA",R3)]:
    print(f"\n{tag}  (2011-2026):")
    print(f"  {'lambda':>8}{'SR':>8}{'maxDD':>9}")
    for lam in [0.0,0.25,0.5,1.0,2.0,4.0]:
        sr,dd=perf(erc_sharpe(R,lam)); star=" <- pure ERC" if lam==0 else ""
        print(f"  {lam:>8.2f}{sr:>8.2f}{dd:>9.1%}{star}")
