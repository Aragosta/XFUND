#!/usr/bin/env python3
"""peer_decomp.py — the user's refined peer idea: decompose momentum into (a) does the GROUP trend, and (b) the
stock's POSITION within its group. Raw peer-return momentum failed; this is different — a CONDITIONAL signal.
Groups = 13F co-holding clusters (PIT). own_mom = 12-1 momentum; group_mom = cluster mean; within = own - group.
Hypotheses: (1) group_mom predicts members (industry-info-diffusion, Hou 2007); (2) laggards in trending groups
catch up (long low-within in high-group-mom clusters). Test IC + L/S net + corr to our MOM alpha (orthogonal?)."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from scipy.stats import skew
import BACKTEST, BETANEUT

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2012-06-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
fwd = mret.shift(-1)

# 12-1 momentum (skip most recent month), per stock, monthly
i12 = m_px.shift(12); i1 = m_px.shift(1); own = (i1/i12 - 1.0).where(np.isfinite(i1/i12))

# PIT cluster map: latest snapshot >=60d before d
CL = pd.read_parquet("data/13f/clusters_ts_focused.parquet"); CL["period"]=pd.to_datetime(CL["period"])
snaps = sorted(CL["period"].unique())
def clmap(d):
    a=[s for s in snaps if s <= d - pd.Timedelta(days=60)]
    if not a: return None
    return CL[CL["period"]==a[-1]].set_index("ticker")["cluster"]

# build group_mom and within per month
GM = pd.DataFrame(np.nan,index=me,columns=px.columns); WI = pd.DataFrame(np.nan,index=me,columns=px.columns)
for d in me:
    cm = clmap(d)
    if cm is None: continue
    o = own.loc[d]; live = elig.loc[d]
    df = pd.DataFrame({"own":o,"cl":cm.reindex(px.columns)})[live].dropna()
    if len(df)<50: continue
    gm = df.groupby("cl")["own"].transform("mean")
    GM.loc[d, df.index] = gm.values; WI.loc[d, df.index] = (df["own"]-gm).values

def xs_ic(sig):
    ics=[]
    for d in me:
        s=sig.loc[d].where(elig.loc[d]); y=fwd.loc[d]; df=pd.DataFrame({"s":s,"y":y}).dropna()
        if len(df)<40: continue
        ics.append(df["s"].corr(df["y"],method="spearman"))
    ics=np.array(ics); return np.nanmean(ics), np.nanmean(ics)/(np.nanstd(ics)/np.sqrt(len(ics))+1e-9)
def book(sig, sign=1):
    W=pd.DataFrame(0.0,index=me,columns=px.columns)
    for d in me:
        s=(sign*sig.loc[d]).where(elig.loc[d]).dropna()
        if len(s)<40: continue
        n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    return BETANEUT.betaneut(W,BETA)
def stream(W):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    s=pd.Series(r["returns"]); s.index=pd.DatetimeIndex(s.index); return s[(s.index>="2014-01-01")&(s.index<"2023-01-01")].dropna()
def show(nm,x):
    e=(1+x).cumprod(); print(f"  {nm:34} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}  skew {skew(x):>+5.2f}")

# laggard-in-trending-group: high group_mom, low within -> catch up. score = group_mom - within (rank)
catchup = GM.rank(axis=1) - WI.rank(axis=1)
al=pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index=pd.DatetimeIndex(pd.to_datetime(al.index)); al=al[~al.index.duplicated()].reindex(index=me,columns=px.columns).fillna(0.0)
ar=stream(al)
print("="*100); print("PEER DECOMPOSITION (13F group momentum x within-group position) — 2014-2022")
for nm,sig in [("group_mom",GM),("within (own-group)",WI),("catchup (hi group, lo within)",catchup),("own 12-1 mom (ref)",own)]:
    ic,tt=xs_ic(sig); x=stream(book(sig)); show(f"{nm}  [IC {ic:+.3f} t{tt:+.1f}]",x)
    c=pd.DataFrame({"s":x,"a":ar}).dropna(); print(f"      corr to MOM+DM alpha = {c.corr().iloc[0,1]:+.2f}")
