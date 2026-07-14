"""honest_backtest.py — STRUCTURAL reliability fix, measured. Two changes vs the current backtest:
  (1) PAY FOR SQUEEZES: stop zeroing monthly returns >= +100% (prices are split-adjusted, so those are REAL
      moves, not artifacts). Only >+1000% is treated as a data error.
  (2) BORROWABLE SHORTS: the short leg may only hold genuinely shortable names (mdv > $25M) — a real desk
      cannot short illiquid squeeze-prone microcaps.
Re-run the VQ book under each, isolating how much each assumption was inflating the result."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
import BACKTEST
hub=DataHub(); me,m_px,elig=hub.me,hub.m_px,hub.elig("liquid")
tc=BACKTEST.tiered_transaction_costs(hub.mdv); bf=BACKTEST.tiered_borrow_fees(hub.mdv)
SCORE=pd.read_pickle("/tmp/fundq_score.pkl")
shortable = elig & (hub.mdv.rolling(3,min_periods=1).mean() > 25e6)             # borrowable short universe
raw = m_px.pct_change(fill_method=None)
rets = {
  "current (>=100% zeroed)": raw.where(raw < 1.0),                              # what we've been using
  "HONEST (pay squeezes)":   raw.where(raw < 10.0),                             # only >1000% = data error
}
def book(short_universe):
    W=pd.DataFrame(0.0,index=me,columns=m_px.columns)
    for d in me:
        s=SCORE.loc[d].dropna()
        if len(s)<80: continue
        n=max(1,int(len(s)*0.10))
        longs=s.nlargest(n).index
        sc=s[s.index.isin(short_universe.loc[d][short_universe.loc[d]].index)] if short_universe is not None else s
        if len(sc)<n: continue
        shorts=sc.nsmallest(n).index
        W.loc[d,longs]=1.0/n; W.loc[d,shorts]=-1.0/n
    return W
def run(W, mret, lab):
    synth=(1+mret.fillna(0.0)).cumprod()
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in me if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); x=x[(x.index>="2016-01-01")&(x.index<"2027-01-01")].dropna()
    e=(1+x).cumprod()
    print(f"  {lab:44} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>7.1%}  maxDD {(e/e.cummax()-1).min():>7.1%}")
    return x
print("="*100); print("HONEST BACKTEST — isolating each unrealistic assumption (VQ book, 2016-2026)")
Wall=book(None); Wsh=book(shortable)
a=run(Wall, rets["current (>=100% zeroed)"], "1. current: all-universe shorts, squeezes zeroed")
b=run(Wall, rets["HONEST (pay squeezes)"],   "2. + PAY FOR SQUEEZES (real returns)")
c=run(Wsh,  rets["current (>=100% zeroed)"], "3. + borrowable shorts only (squeezes still zeroed)")
d=run(Wsh,  rets["HONEST (pay squeezes)"],   "4. HONEST: borrowable shorts + pay squeezes")
print("\n  per-year (honest vs current):")
for y in range(2016,2027):
    ya=a[[pd.Timestamp(x).year==y for x in a.index]]; yd=d[[pd.Timestamp(x).year==y for x in d.index]]
    if not len(ya): continue
    print(f"    {y}   current {(1+ya).prod()-1:>+7.0%}   HONEST {(1+yd).prod()-1:>+7.0%}")
