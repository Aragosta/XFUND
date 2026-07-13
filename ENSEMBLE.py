#!/usr/bin/env python3
"""ENSEMBLE.py — the 3-sleeve book: MOM + DM + VQ (value/quality), combined by leak-free expanding ERC
(risk-parity). Full per-year statistics vs SPY buy-and-hold. All net-of-cost, all from the DataHub sleeves.
Sleeve return streams: /tmp/mom_champ.pkl (n1), /tmp/dm_returns.pkl, /tmp/fundq_returns.pkl; SPY from DataHub."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from DATAHUB import DataHub
hub = DataHub()
def S(p, k=None):
    o = pickle.load(open(p, "rb")); s = pd.Series(dict(o) if isinstance(o, list) else (o[k] if k else o)).dropna()
    s.index = pd.DatetimeIndex(s.index); return s
MOM = S("/tmp/mom_champ.pkl", "n1"); DM = S("/tmp/dm_returns.pkl"); VQ = S("/tmp/fundq_returns.pkl")
SPY = hub.spy_m.dropna()
R = pd.DataFrame({"MOM": MOM, "DM": DM, "VQ": VQ}).dropna()

def erc(cov):
    n = cov.shape[0]; w = np.ones(n)/n
    for _ in range(500):
        m = cov@w; wn = 1/np.maximum(m,1e-12); wn/=wn.sum()
        if np.abs(wn-w).max()<1e-11: break
        w = 0.5*w+0.5*wn
    return w
W = pd.DataFrame(np.nan, index=R.index, columns=R.columns)
for i in range(24, len(R)): W.iloc[i] = erc(np.cov(R.iloc[:i].T.values)+1e-6*np.eye(3))   # leak-free expanding ERC
ens = (R*W).sum(1)[W.notna().all(1)].dropna()
def full(x, lab, w0="2013-06-01", w1="2027-01-01"):
    x = x[(x.index>=w0)&(x.index<w1)].dropna(); e=(1+x).cumprod(); dd=(e/e.cummax()-1)
    ann=(1+x).prod()**(12/len(x))-1; vol=x.std()*np.sqrt(12); sr=x.mean()/x.std()*np.sqrt(12)
    dn=x[x<0].std()*np.sqrt(12); sor=x.mean()*12/dn if dn>0 else np.nan
    csp=pd.DataFrame({"a":x,"s":SPY.reindex(x.index)}).dropna().corr().iloc[0,1]
    print(f"  {lab:16} SR {sr:>5.2f}  Sortino {sor:>4.1f}  ann {ann:>6.1%}  vol {vol:>5.1%}  maxDD {dd.min():>6.1%}  corr-SPY {csp:>+5.2f}")
    return x

print("="*94); print("3-SLEEVE ENSEMBLE  (MOM + DM + VQ, expanding-ERC, net)  vs  SPY buy-and-hold")
e = full(ens, "ENSEMBLE"); full(SPY, "SPY buy&hold")
print("\n  per-sleeve:"); [full(R[c], c) for c in R.columns]
print("\n  PER-YEAR FULL STATISTICS — ENSEMBLE vs SPY buy-and-hold:")
print(f"  {'year':>5}{'ENS ret':>9}{'ENS SR':>8}{'ENS DD':>8}{'ENS vol':>9}  |{'SPY ret':>9}{'SPY SR':>8}{'SPY DD':>8}  | {'MOM/DM/VQ ret'}")
def ystats(s, y):
    x = s[[pd.Timestamp(d).year==y for d in s.index]].dropna()
    if len(x)==0: return None
    e=(1+x).cumprod(); return dict(ret=(1+x).prod()-1, sr=(x.mean()/x.std()*np.sqrt(12) if x.std()>0 else np.nan),
                                   dd=(e/e.cummax()-1).min(), vol=x.std()*np.sqrt(12))
for y in range(2013,2027):
    en=ystats(ens,y); sp=ystats(SPY,y)
    if en is None: continue
    mv=[ystats(R[c],y) for c in ["MOM","DM","VQ"]]
    mvs="/".join(f"{m['ret']:>+4.0%}" if m else "  —" for m in mv)
    sps=f"{sp['ret']:>+9.0%}{sp['sr']:>8.2f}{sp['dd']:>+8.0%}" if sp else " "*25
    print(f"  {y:>5}{en['ret']:>+9.0%}{en['sr']:>8.2f}{en['dd']:>+8.0%}{en['vol']:>9.0%}  |{sps}  | {mvs}")
avgw=W.dropna().mean(); print(f"\n  avg ERC weights: MOM {avgw['MOM']:.2f} / DM {avgw['DM']:.2f} / VQ {avgw['VQ']:.2f}")
print("  NOTE: 2021+ is a value/quality revival regime (VQ SR ~3 there); anchor forward expectations on 2016-20.")
