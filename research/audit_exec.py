"""audit_exec.py — EXECUTION-TIMING blindspot. Sleeves build the signal from the month-end CLOSE(t) and then
trade AT close(t) (BACKTEST lag=0), earning t->t+1. You cannot know close(t) and trade at it. Realistically you
submit after the close and fill at the NEXT day's close. Measure how much of the monthly L/S alpha is earned on
that first (untradeable) day — the same trap that killed our daily reversion numbers."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub=DataHub(); me,m_px,px_d,elig=hub.me,hub.m_px,hub.px_d,hub.elig("liquid")
SCORE=pd.read_pickle("/tmp/fundq_score.pkl")
days=px_d.index
full=[]; delayed=[]; day1=[]
for i,d in enumerate(me[:-1]):
    s=SCORE.loc[d].dropna()
    if len(s)<80: continue
    n=max(1,int(len(s)*0.10)); L=s.nlargest(n).index; S=s.nsmallest(n).index
    nxt=me[i+1]
    pos=days.searchsorted(d)
    if pos+1>=len(days): continue
    d1=days[pos+1]                                              # first trading day AFTER the signal close
    p0=px_d.loc[d]; p1=px_d.loc[d1]; p2=px_d.loc[nxt]
    def spread(a,b):                                            # equal-weight L/S return between two price rows
        rl=(b[L]/a[L]-1).replace([np.inf,-np.inf],np.nan); rs=(b[S]/a[S]-1).replace([np.inf,-np.inf],np.nan)
        return np.nanmean(rl)-np.nanmean(rs)
    f=spread(p0,p2); dl=spread(p1,p2); d1r=spread(p0,p1)
    if np.isfinite(f) and np.isfinite(dl): full.append(f); delayed.append(dl); day1.append(d1r)
F=pd.Series(full); D=pd.Series(delayed); D1=pd.Series(day1)
def sr(x): return x.mean()/x.std()*np.sqrt(12)
print("="*84); print("EXECUTION-TIMING AUDIT (VQ decile L/S, gross, no costs)")
print(f"  trade AT signal close (current, lag=0):   ann {F.mean()*12:>+7.1%}   SR {sr(F):>5.2f}")
print(f"  trade NEXT day's close (realistic):       ann {D.mean()*12:>+7.1%}   SR {sr(D):>5.2f}")
print(f"  --> alpha earned on the FIRST (untradeable) day: ann {D1.mean()*12:>+7.1%}  ({D1.mean()/F.mean()*100 if F.mean() else 0:.0f}% of total)")
print(f"  --> SR lost to realistic execution: {sr(F)-sr(D):>+5.2f}")
