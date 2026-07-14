"""reliability.py — WHY are recent-year returns implausible? Prime suspect: `mret = pct_change().where(z<1.0)`
silently ZEROES any monthly return >= +100%. Our alpha is SHORT-side-heavy, so every short SQUEEZE (the exact
thing that kills a short book) is erased. Quantify: how many >100% moves per year, how many sit in our SHORT
book, and what P&L we never paid. This is the structural reliability question, not a patch."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub=DataHub(); me,m_px,elig=hub.me,hub.m_px,hub.elig("liquid")
raw = m_px.pct_change(fill_method=None)                       # TRUE returns (nothing zeroed)
used = raw.where(raw < 1.0)                                   # what the sleeves actually use (>=100% -> NaN -> 0)
big  = (raw >= 1.0) & elig                                    # eligible names that MORE THAN DOUBLED in a month
SCORE=pd.read_pickle("/tmp/fundq_score.pkl")
print("="*88); print("SQUEEZE BLIND-SPOT: monthly moves >= +100% that the backtest ZEROES")
print(f"  {'year':>5}{'#>100% moves':>14}{'in SHORT book':>15}{'avg true ret':>14}{'P&L never paid':>16}")
tot=0.0
for y in range(2013,2027):
    ds=[d for d in me if d.year==y]
    n=0; nshort=0; rets=[]; pnl=0.0
    for d in ds:
        b=big.loc[d]; names=b[b].index
        n+=len(names)
        s=SCORE.loc[d].dropna()
        if len(s)>=80:
            k=max(1,int(len(s)*0.10)); shorts=set(s.nsmallest(k).index)
            hit=[t for t in names if t in shorts]
            nshort+=len(hit)
            for t in hit:
                r=raw.loc[d,t]; rets.append(r)
                pnl += -(1.0/k)*r                              # short position eats the true (unclipped) move
    tot+=pnl
    ar=np.mean(rets) if rets else np.nan
    print(f"  {y:>5}{n:>14}{nshort:>15}{ar:>+13.0%}{pnl:>+15.1%}")
print(f"\n  TOTAL P&L the SHORT book never paid for (VQ decile): {tot:+.1%}")
print("  (each unit is a full-book monthly return the short leg would have LOST but booked as 0)")
