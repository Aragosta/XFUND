#!/usr/bin/env python3
"""[SHELVED 2026-07 — kept for reference, NOT in the deployable stack.]
VERDICT: on the real 3-sleeve book (MOM+DM+orthogonal vol-managed BETA), STATIC ERC beats every adaptive variant
on BOTH return and drawdown (static SR 1.00/1.62 maxDD -11.9% vs best-adapt 0.95/1.59 maxDD -20.6%). The +0.11
SR seen on the 2-sleeve (MOM+DM, corr 0.47) POC was a mirage: with a genuinely ORTHOGONAL third sleeve,
diversification beats trailing-Sharpe return-chasing — adaptive under-weights beta right when it diversifies.
'Improving' it (ensemble/floor/shrink/sortino) just walks it back toward static. Allocation stays STATIC.

adapt.py — ADAPTIVE sleeve allocation (meta-layer). Re-weight sleeves by a parameter solved on trailing
performance, harvesting STRATEGY/FACTOR MOMENTUM (Ehsani-Linnainmaa) — the one 1st-moment that IS forecastable
(a sleeve's own return persists at 6-12mo, unlike the market's direction). NOT regime switching.

Robust design (POC found raw Sharpe-tilt(12) beats static ERC +0.11 SR; win=3 fails = error-maximization):
  - SHARPE/SORTINO TILT, not mean-variance (no unstable covariance inversion -> no DD blowups)
  - LOOKBACK ENSEMBLE (avg over 6/9/12mo) -> robust to any single horizon
  - SHRINK toward ERC (risk-parity prior) -> controls aggression/turnover
  - weight FLOOR -> never zero a sleeve (keep diversification)
Streams: MOM, DM (net) + vol-managed SPY BETA as a 3rd, orthogonal sleeve. Leak-free throughout."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd

# ---------- sleeve return streams ----------
def load_streams():
    M = pickle.load(open("/tmp/mom_champ.pkl","rb")); MOM=pd.Series(M["n1"]).dropna(); MOM.index=pd.DatetimeIndex(MOM.index)
    D = pickle.load(open("/tmp/dm_returns.pkl","rb")); DM=pd.Series(D).dropna(); DM.index=pd.DatetimeIndex(DM.index)
    spy = pd.read_parquet("/tmp/spy.parquet")["SPY"]
    spm = spy.resample("ME").last().pct_change()                                      # proper MONTHLY SPY return
    spm.index = pd.DatetimeIndex(spm.index); spm = spm.reindex(MOM.index)             # align to month-end grid
    fvol = spm.rolling(6, min_periods=3).std().shift(1)*np.sqrt(12)                   # leak-free trailing vol
    beta = (0.15/fvol).clip(0.25, 3.0).fillna(1.0) * spm                              # vol-managed SPY (target 15%)
    R = pd.DataFrame({"MOM":MOM, "DM":DM, "BETA":beta.rename("BETA")}).dropna()
    return R

# ---------- allocators (all leak-free: weights at t from data < t) ----------
def erc_w(cov):
    n=cov.shape[0]; w=np.ones(n)/n
    for _ in range(500):
        mrc=cov@w; wn=1.0/np.maximum(mrc,1e-12); wn/=wn.sum()
        if np.abs(wn-w).max()<1e-10: break
        w=0.5*w+0.5*wn
    return w

def tilt_weights(H, wins=(6,9,12), shrink=0.5, floor=0.10, sortino=True):
    """Ensemble Sharpe/Sortino tilt, shrunk toward ERC, floored. H = trailing return frame (rows=months)."""
    n=H.shape[1]; parts=[]
    for w in wins:
        Hw=H.iloc[-w:] if len(H)>=w else H
        mu=Hw.mean().values
        dn=Hw.clip(upper=0).std().values if sortino else Hw.std().values
        sh=np.clip(mu/(dn+1e-9),0,None)
        parts.append(sh/sh.sum() if sh.sum()>0 else np.ones(n)/n)
    wt=np.mean(parts,axis=0)                                                          # lookback ensemble
    werc=erc_w(np.cov(H.T.values)+1e-6*np.eye(n))
    w=shrink*werc + (1-shrink)*wt                                                     # shrink toward risk-parity
    w=np.clip(w,floor,None); w=w/w.sum()                                             # floor + renorm
    return w

def run(R, rule, warm=24, **kw):
    W=pd.DataFrame(np.nan,index=R.index,columns=R.columns)
    for i in range(warm,len(R)):
        H=R.iloc[:i]
        if rule=="static":   W.iloc[i]=erc_w(np.cov(H.T.values)+1e-6*np.eye(R.shape[1]))
        elif rule=="adapt":  W.iloc[i]=tilt_weights(H, **kw)
    c=(R*W).sum(axis=1); return c[W.notna().all(axis=1)]

def perf(c, lab):
    out=[]
    for w0,w1,tag in [("2011-06-01","2023-01-01","2011-2022"),("2011-06-01","2027-01-01","2011-2026")]:
        x=c[(c.index>=w0)&(c.index<w1)].dropna(); e=(1+x).cumprod()
        sr=x.mean()/x.std()*np.sqrt(12); out.append(sr)
        print(f"    {lab:34}[{tag}] SR {sr:>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}")
    return out

if __name__ == "__main__":
    R=load_streams(); print(f"[streams] {len(R)} mo {R.index[0].date()}..{R.index[-1].date()}")
    print("  corr:\n"+R.corr().round(2).to_string().replace("\n","\n  "))
    print("\nSTATIC ERC (3-sleeve baseline):"); perf(run(R,"static"),"static ERC")
    print("\nADAPT — improvements stacked:")
    perf(run(R,"adapt", wins=(12,), shrink=0.0, floor=0.0, sortino=False),"raw sharpe-tilt(12)")
    perf(run(R,"adapt", wins=(6,9,12), shrink=0.0, floor=0.0, sortino=False),"+ lookback ensemble")
    perf(run(R,"adapt", wins=(6,9,12), shrink=0.0, floor=0.10, sortino=False),"+ floor")
    perf(run(R,"adapt", wins=(6,9,12), shrink=0.5, floor=0.10, sortino=False),"+ shrink-to-ERC")
    perf(run(R,"adapt", wins=(6,9,12), shrink=0.5, floor=0.10, sortino=True),"+ sortino (FULL adapt)")
