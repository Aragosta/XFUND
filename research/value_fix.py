#!/usr/bin/env python3
"""value_fix.py — fix the split-basis bug in VALUE. Price is split-ADJUSTED (back-adjusted) but EDGAR shares are
AS-REPORTED (pre-split) -> mcap wrong for any stock that split after date t -> growth splitters (NVDA/AAPL/TSLA)
look like value. Fix: reconstruct cumulative split factor from JUMPS in the shares series, put shares on the
current (adjusted) split basis: adj_shares(t) = shares(t) * prod(splits after t). Re-test value year-by-year."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST, BETANEUT

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig = (m_px>5)&(cov>0.9)&(mdv>5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)

f = pd.read_parquet("data/edgar/facts.parquet").dropna(subset=["val"]).copy(); f["end"]=pd.to_datetime(f["end"])
LAG={"book":90,"shares":90,"ni":120}; f["avail"]=f["end"]+pd.to_timedelta(f["concept"].map(LAG).fillna(90),unit="D")
def pit(sub):
    out=pd.DataFrame(np.nan,index=me,columns=px.columns)
    for tk,g in sub.groupby("ticker"):
        if tk not in px.columns: continue
        g=g.sort_values(["avail","end"]).drop_duplicates("avail",keep="last")
        idx=np.searchsorted(g["avail"].values,me.values,side="right")-1
        out[tk]=np.where(idx>=0,g["val"].values[idx.clip(min=0)],np.nan)
    return out
book=pit(f[f.concept=="book"]); shares_raw=pit(f[f.concept=="shares"]); earn=pit(f[f.concept=="ni"])

CLEAN=np.array([1.5,2,3,4,5,6,7,8,10,15,20,25,30]); CLEAN=np.concatenate([CLEAN,1/CLEAN])
def snap_split(r):
    if 0.72 < r < 1.4: return 1.0                                              # organic buyback/issuance
    j=np.argmin(np.abs(CLEAN-r)); return CLEAN[j] if abs(CLEAN[j]-r)/CLEAN[j] < 0.12 else 1.0
def split_adjust(S):
    adj=S.copy()
    for tk in S.columns:
        s=S[tk].dropna()
        if len(s)<2: continue
        fac=pd.Series(1.0,index=s.index)
        rr=(s/s.shift(1)).values
        for i in range(1,len(s)): fac.iloc[i]=snap_split(rr[i])
        future=fac[::-1].cumprod()[::-1].shift(-1).fillna(1.0)                 # product of splits AFTER t
        adj.loc[s.index,tk]=s.values*future.values
    return adj
shares=split_adjust(shares_raw)

# sanity: NVDA B/M before vs after
d=me[me<="2023-12-31"][-1]
for tk in ["NVDA","AAPL","TSLA"]:
    if tk in shares.columns and np.isfinite(shares_raw.loc[d,tk]):
        bm0=book.loc[d,tk]/(shares_raw.loc[d,tk]*m_px.loc[d,tk]); bm1=book.loc[d,tk]/(shares.loc[d,tk]*m_px.loc[d,tk])
        print(f"  {tk}: B/M raw {bm0:.3f} -> fixed {bm1:.3f}  (shares {shares_raw.loc[d,tk]:,.0f} -> {shares.loc[d,tk]:,.0f})")

mcap=shares*m_px; bm=(book/mcap).where(book>0); ep=(earn/mcap)
def zwin(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
val=(zwin(bm).fillna(0)*bm.notna()+zwin(ep).fillna(0)*ep.notna())/(bm.notna().astype(float)+ep.notna().astype(float)).replace(0,np.nan)
cover=book.notna()&shares.notna(); val=val.where(elig&cover)
Wv=pd.DataFrame(0.0,index=me,columns=px.columns)
for d in me:
    s=val.loc[d].dropna()
    if len(s)<50: continue
    n=max(1,int(len(s)*0.10)); Wv.loc[d,s.nlargest(n).index]=1.0/n; Wv.loc[d,s.nsmallest(n).index]=-1.0/n
Wv=BETANEUT.betaneut(Wv,BETA)
def stream(W):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    s=pd.Series(r["returns"]); s.index=pd.DatetimeIndex(s.index); return s[s.index>="2011-01-01"].dropna()
def show(name,x):
    x=x.dropna(); e=(1+x).cumprod()
    print(f"  {name:20} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  maxDD {(e/e.cummax()-1).min():>6.1%}")
    print("    "+" ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2011,2027)))
def value_book(universe_mask):
    v = val.where(universe_mask & cover); W=pd.DataFrame(0.0,index=me,columns=px.columns)
    for d in me:
        s=v.loc[d].dropna()
        if len(s)<50: continue
        n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    return BETANEUT.betaneut(W,BETA)
relaxed=(m_px>3)&(cov>0.8)&(mdv>5e5)                                            # small/mid-cap inclusive
print("="*90); print("VALUE (split-FIXED, B/M + E/P, beta-neut) — LIQUID vs RELAXED universe, net, 2011+")
show("VALUE liquid", stream(value_book(elig)))
show("VALUE relaxed(small/mid)", stream(value_book(relaxed)))
