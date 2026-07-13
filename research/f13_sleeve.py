#!/usr/bin/env python3
"""f13_sleeve.py — actually BACKTEST the 13F breadth/value LEVEL signal (IC +0.02, t+5) as a sleeve.
It's SLOW (quarterly, held monthly) -> low turnover -> costs barely bite (unlike peer-momentum). Test net SR,
turnover, corr to MOM+DM alpha, AND whether it survives controlling for size (is +0.02 just the size premium?).
Also fix the incremental vol-forecast test (guard infs)."""
import warnings; warnings.filterwarnings("ignore")
import glob, json, numpy as np, pandas as pd
from scipy.stats import skew
import BACKTEST, BETANEUT

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2012-01-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
size = np.log(mdv.replace(0, np.nan))
dvol = px.pct_change().rolling(21).std().resample("ME").last().reindex(me)*np.sqrt(21); fvol = dvol.shift(-1)

c2t = json.load(open("data/13f/cusip_ticker.json")); up = {c.upper(): c for c in px.columns}
rows = []
for f in sorted(glob.glob("data/13f/parsed/*.parquet")):
    d = pd.read_parquet(f, columns=["cik","period","cusip","value"]); d["tic"]=d["cusip"].str.slice(0,9).map(c2t)
    d = d.dropna(subset=["tic"]); d["tic"]=d["tic"].str.upper().map(up); d = d.dropna(subset=["tic"])
    rows.append(d.groupby(["period","tic"]).agg(hold=("cik","nunique"), val=("value","sum")).reset_index())
P = pd.concat(rows, ignore_index=True); P["pend"]=pd.to_datetime(P["period"], format="%d-%b-%Y", errors="coerce"); P=P.dropna(subset=["pend"])
HOLD=P.pivot_table(index="pend",columns="tic",values="hold").reindex(columns=px.columns)
VAL =P.pivot_table(index="pend",columns="tic",values="val").reindex(columns=px.columns)
def mpit(Q): Qs=Q.copy(); Qs.index=[pd.Timestamp(p)+pd.Timedelta(days=50) for p in Q.index]; return Qs.reindex(me,method="ffill")
breadth=mpit(np.log1p(HOLD)); value=mpit(np.log1p(VAL))

def decile_book(sig, resid_of=None):
    S = sig.copy()
    if resid_of is not None:                                                  # neutralize signal to a control (size) cross-sectionally
        for d in me:
            m=elig.loc[d]; X=pd.DataFrame({"s":S.loc[d],"c":resid_of.loc[d]})[m].dropna()
            if len(X)<40: continue
            A=np.column_stack([np.ones(len(X)),X["c"]]); b,*_=np.linalg.lstsq(A,X["s"],rcond=None); S.loc[d]=np.nan; S.loc[d,X.index]=X["s"]-A@b
    W=pd.DataFrame(0.0,index=me,columns=px.columns)
    for d in me:
        s=S.loc[d].where(elig.loc[d]).dropna()
        if len(s)<40: continue
        n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    return BETANEUT.betaneut(W,BETA)
def stream(W):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    s=pd.Series(r["returns"]); s.index=pd.DatetimeIndex(s.index); return s.dropna(), r["ann_turnover"]
def show(name,x,turn):
    x=x[(x.index>="2013-06-01")&(x.index<"2023-01-01")].dropna(); g=stream  # honest window
    e=(1+x).cumprod()
    print(f"  {name:30} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  maxDD {(e/e.cummax()-1).min():>6.1%}  turn {turn:>4.1f}  skew {skew(x):>+5.2f}")
    return x
al=__import__("pickle").load(open("/tmp/meta_weights.pkl","rb")); al.index=pd.DatetimeIndex(pd.to_datetime(al.index)); al=al[~al.index.duplicated()].reindex(index=me,columns=px.columns).fillna(0.0)
ar,_=stream(al)
print("="*104); print("13F BREADTH/VALUE LEVEL as a SLEEVE (slow signal, beta-neut, net costs) — 2013H2-2022")
for nm,sig in [("breadth level",breadth),("value level",value),("breadth+value",breadth.rank(axis=1)+value.rank(axis=1))]:
    x,turn=stream(decile_book(sig)); xx=show(nm,x,turn)
    c=pd.DataFrame({"s":xx,"a":ar}).dropna(); print(f"      corr to MOM+DM alpha = {c.corr().iloc[0,1]:+.2f}")
print("  --- size-neutralized (is +0.02 IC just the size premium?) ---")
for nm,sig in [("breadth ⟂ size",breadth),("value ⟂ size",value)]:
    x,turn=stream(decile_book(sig,resid_of=size)); show(nm,x,turn)

# ---- incremental vol forecast (guarded) ----
def zc(df): d=df.replace([np.inf,-np.inf],np.nan); return d.sub(d.mean(axis=1),axis=0).div(d.std(axis=1)+1e-9,axis=0)
Zv,Zs,Zb,Zval,Zy=zc(dvol),zc(size),zc(breadth),zc(value),zc(fvol)
resid={}
for d in me:
    m=elig.loc[d]; X=pd.DataFrame({'v':Zv.loc[d],'s':Zs.loc[d],'y':Zy.loc[d]})[m].dropna()
    if len(X)<40: continue
    A=np.column_stack([np.ones(len(X)),X['v'],X['s']]); b,*_=np.linalg.lstsq(A,X['y'],rcond=None); resid[d]=X['y']-A@b
ics=[pd.Series(resid[d]).corr(Zb.loc[d].reindex(resid[d].index),method='spearman') for d in resid if Zb.loc[d].reindex(resid[d].index).notna().sum()>40]
ics=np.array(ics); ics=ics[np.isfinite(ics)]
print(f"\n  incremental IC(13F breadth -> vol | trailing-vol + size) = {np.nanmean(ics):+.3f}  t {np.nanmean(ics)/(np.nanstd(ics)/np.sqrt(len(ics))):+.1f}  ({len(ics)} mo)")
