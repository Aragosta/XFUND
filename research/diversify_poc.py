#!/usr/bin/env python3
"""diversify_poc.py — three tests: (1) VALUE sleeve PoC (long-term reversal, the price-based value proxy — no
fundamentals in the panel), (2) full model with a 60/40 net-LONG split, (3) dispersion-conditional net-long tilt.
Alpha book = META final (/tmp/meta_weights.pkl). All with year-by-year + full stats (ann/SR/vol/maxDD/skew)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST, BETANEUT

px=pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t=px.columns.str.match(r"^Z[A-Z]ZZT$")|px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px=px.loc[:,~t]
vb=pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me=pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me=me[me>=pd.Timestamp("2010-12-01")]
m_px=px.reindex(me); mret=m_px.pct_change(fill_method=None).where(lambda z:z<1.0); synth=(1+mret.fillna(0.0)).cumprod()
mdv=(px*vb).resample("ME").sum().reindex(me,method="ffill"); cov=px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig=(m_px>5)&(cov>0.9)&(mdv>5e6); tc=BACKTEST.tiered_transaction_costs(mdv); bf=BACKTEST.tiered_borrow_fees(mdv); meP=pd.PeriodIndex(me,freq="M")
BETA=BETANEUT.rolling_beta(mret,elig,bw=60)
spy=pd.read_parquet("/tmp/spy.parquet")["SPY"].reindex(me,method="ffill").pct_change()

def net_bt(W,borrow=True):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=(bf if borrow else mdv*0.0))
    s=pd.Series(r["returns"]); s.index=pd.DatetimeIndex(s.index); return s[s.index>="2011-01-01"].dropna()
def full(x):
    x=x.dropna(); e=(1+x).cumprod(); return (1+x).prod()**(12/len(x))-1,x.mean()/x.std()*np.sqrt(12),x.std()*np.sqrt(12),(e/e.cummax()-1).min(),skew(x)
def yby(name,x):
    print(f"\n{name}:  ann {full(x)[0]:.1%}  SR {full(x)[1]:.2f}  vol {full(x)[2]:.1%}  maxDD {full(x)[3]:.1%}  skew {full(x)[4]:+.2f}  corrSPY {pd.DataFrame({'a':x,'s':spy.reindex(x.index)}).dropna().corr().iloc[0,1]:+.2f}")
    print("  "+" ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+6.0%}" for y in range(2011,2027)))

# ---------- (1) VALUE sleeve = long-term reversal (long past-losers over yrs 5..1, skip recent 12m), beta-neut decile L/S ----------
val=-(m_px.shift(12)/m_px.shift(60)-1).where(lambda z:z.abs()<10)           # value proxy: cheap = long-term loser
Wv=pd.DataFrame(0.0,index=me,columns=px.columns)
for d in me:
    s=val.loc[d].where(elig.loc[d]).dropna()
    if len(s)<50: continue
    n=max(1,int(len(s)*0.10)); Wv.loc[d,s.nlargest(n).index]=1.0/n; Wv.loc[d,s.nsmallest(n).index]=-1.0/n
Wv=BETANEUT.betaneut(Wv,BETA); vr=net_bt(Wv)
al=pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index=pd.DatetimeIndex(pd.to_datetime(al.index)); al=al[~al.index.duplicated()].reindex(index=me,columns=px.columns).fillna(0.0)
ar=net_bt(al)
print("="*96); print("(1) VALUE SLEEVE PoC (long-term reversal, beta-neut) + correlation to the alpha")
yby("VALUE (LT-reversal)",vr); yby("ALPHA (META)",ar)
print(f"\n  corr(VALUE, ALPHA) = {pd.DataFrame({'v':vr,'a':ar}).dropna().corr().iloc[0,1]:+.2f}")
# value + alpha ERC
both=pd.DataFrame({"a":ar,"v":vr}).dropna(); wv=(1/both['v'].expanding(24).std())/((1/both['a'].expanding(24).std())+(1/both['v'].expanding(24).std()))
yby("ALPHA + VALUE (ERC)",((1-wv)*both['a']+wv*both['v']).dropna())

# ---------- (2) full model 60/40 net-long split; (3) dispersion-conditional net-long ----------
def tilt(W,net):  # net = net-long fraction: long gross=(1+net), short gross=(1-net), total 2.0
    out=pd.DataFrame(0.0,index=W.index,columns=W.columns)
    for d in W.index:
        w=W.loc[d]; L=w.clip(lower=0); S=w.clip(upper=0); gl,gs=L.sum(),-S.sum()
        if gl>1e-9 and gs>1e-9: out.loc[d]=L*((1+net)/gl)+S*((1-net)/gs)
        else: out.loc[d]=w
    return out
disp=mret.where(elig).std(axis=1); dpct=pd.Series(disp.shift(1).expanding(24).apply(lambda a:(a.iloc[-1]>a).mean()).values,index=me)
print("\n"+"="*96); print("(2) 60/40 net-LONG split  &  (3) dispersion-conditional net-long — full model (META book)")
yby("neutral 50/50 (base)",ar)
yby("static 60/40 net-long",net_bt(tilt(al,0.20)))
# dispersion-conditional: low dispersion -> more net long (up to 0.35), high dispersion -> neutral
netdyn=pd.Series((0.35*(0.5-dpct).clip(lower=0)/0.5).values,index=me).fillna(0.0)   # 0..0.35
Wd=pd.DataFrame(0.0,index=me,columns=px.columns)
for d in me:
    w=al.loc[d]; L=w.clip(lower=0); S=w.clip(upper=0); gl,gs=L.sum(),-S.sum(); nn=netdyn.loc[d]
    if gl>1e-9 and gs>1e-9: Wd.loc[d]=L*((1+nn)/gl)+S*((1-nn)/gs)
    else: Wd.loc[d]=w
yby("dispersion net-long (dyn)",net_bt(Wd))
print(f"\n  avg dynamic net-long: {netdyn[netdyn.index>='2011'].mean():.2f}  (2011-17 {netdyn[(netdyn.index>='2011')&(netdyn.index<'2018')].mean():.2f}, 2018+ {netdyn[netdyn.index>='2018'].mean():.2f})")
print("  SPY reference:"); yby("SPY buy&hold",spy[spy.index>='2011-01-01'].dropna())
