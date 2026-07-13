#!/usr/bin/env python3
"""regime_beta.py — upgraded BETA sleeve + supervised STATE-MACHINE alpha/beta allocation.

ALPHA = META final book (market-neutral, /tmp/meta_weights.pkl). BETA sleeve variants:
  (1) vol-managed static-long  : expo = median(vol)/vol  (Moreira-Muir; the current rule)
  (2) TREND-SCAN-timed          : backward Lopez de Prado trend-scan on SPY (point-in-time) gates the vol-managed
                                  exposure long/flat by trend significance — a NEW directional beta sleeve.
SUPERVISED STATE MACHINE for the alpha-vs-beta mix (NOT HMM): observable states = (dispersion regime) x (vol
regime) from EXPANDING-percentile thresholds; per-state beta weight = expanding Sharpe-share of beta vs alpha
WITHIN that state (leak-free, supervised on realized state returns); hysteresis to avoid whipsaw. Compare to
static ERC. Net, 2011+."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST, sys; sys.path.insert(0,".")

px=pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t=px.columns.str.match(r"^Z[A-Z]ZZT$")|px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px=px.loc[:,~t]
vb=pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me=pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me=me[me>=pd.Timestamp("2010-12-01")]
mret=px.reindex(me).pct_change(fill_method=None).where(lambda z:z<1.0); synth=(1+mret.fillna(0.0)).cumprod()
mdv=(px*vb).resample("ME").sum().reindex(me,method="ffill"); cov=px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig=(px.reindex(me)>5)&(cov>0.9)&(mdv>5e6); tc=BACKTEST.tiered_transaction_costs(mdv); bf=BACKTEST.tiered_borrow_fees(mdv)

# ALPHA = META book net monthly returns
W=pickle.load(open("/tmp/meta_weights.pkl","rb")); W.index=pd.DatetimeIndex(pd.to_datetime(W.index)); W=W[~W.index.duplicated()].reindex(index=me,columns=px.columns).fillna(0.0)
ar=BACKTEST.backtest(W,synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)["returns"]
alpha=pd.Series(ar); alpha.index=pd.DatetimeIndex(alpha.index)

# SPY + realized vol + backward trend-scan
spy=pd.read_parquet("/tmp/spy_long.parquet")["SPY"].dropna()
sret=spy.reindex(me,method="ffill").pct_change()
rv=(spy.pct_change().pow(2).groupby(spy.index.to_period("M")).sum()**0.5)*np.sqrt(21); rv.index=[p.to_timestamp("M") for p in rv.index]; rv=rv.reindex(me,method="ffill")
def tstat(y):
    n=len(y); x=np.arange(n)-(n-1)/2.0; sxx=(x*x).sum(); b=(x*(y-y.mean())).sum()/sxx; res=y-(y.mean()+b*x)
    se=np.sqrt((res**2).sum()/max(n-2,1)/sxx); return b/se if se>0 else 0.0
dlog=np.log(spy.values); posd=np.searchsorted(spy.index.values,me.values,side="right")-1
tsv=pd.Series(0.0,index=me)
for k,p in enumerate(posd):
    best=0.0
    for L in range(21,253,20):
        if p-L<0: continue
        v=tstat(dlog[p-L:p+1])                                # BACKWARD scan (point-in-time)
        if abs(v)>abs(best): best=v
    tsv.iloc[k]=best

tgt=rv.expanding(min_periods=24).median(); vm=(tgt/rv).clip(upper=2.5)       # vol-managed multiplier
beta_static=pd.Series((vm.shift(1)*sret).values,index=me)                    # (1) vol-managed static-long
gate=(tsv>0).astype(float).clip(0.2,1.0)                                     # long in uptrend, 0.2 floor in downtrend
beta_trend=pd.Series((vm.shift(1)*gate.shift(1)*sret).values,index=me)      # (2) trend-scan-timed

# dispersion + vol regime state (expanding percentiles, leak-free) + hysteresis
disp=mret.where(elig).std(axis=1)
dpct=disp.shift(1).expanding(min_periods=24).apply(lambda a:(a.iloc[-1]>a).mean())
vpct=rv.shift(1).expanding(min_periods=24).apply(lambda a:(a.iloc[-1]>a).mean())
def statemachine(beta):
    both=pd.DataFrame({"a":alpha,"b":beta}).dropna(); idx=both.index
    dp=dpct.reindex(idx); vp=vpct.reindex(idx); w=pd.Series(0.5,index=idx); state_prev=None; H=0.10
    hist={}  # state -> list of (a,b) realized
    for i,dt in enumerate(idx):
        d,v=dp.loc[dt],vp.loc[dt]
        if not np.isfinite(d) or not np.isfinite(v): w.loc[dt]=0.5; continue
        s=(int(d>0.5),int(v>0.5))
        if state_prev is not None and abs(d-0.5)<H and abs(v-0.5)<H: s=state_prev  # hysteresis near boundary
        # supervised per-state beta share from PAST occurrences of this state (expanding, leak-free)
        h=hist.get(s,[])
        if len(h)>=6:
            A=np.array([x[0] for x in h]); B=np.array([x[1] for x in h])
            sa=A.mean()/(A.std()+1e-9); sb=B.mean()/(B.std()+1e-9)
            w.loc[dt]=float(np.clip(sb/(abs(sa)+abs(sb)+1e-9),0.1,0.9))
        else: w.loc[dt]=0.4
        hist.setdefault(s,[]).append((both.loc[dt,"a"],both.loc[dt,"b"])); state_prev=s
    comb=(1-w)*both["a"]+w*both["b"]; return comb.dropna()

def st(x,lo=2011,hi=2026):
    x=x[[lo<=d.year<=hi for d in x.index]].dropna()
    if len(x)<3: return (np.nan,)*4
    e=(1+x).cumprod(); return (1+x).prod()**(12/len(x))-1,x.mean()/x.std()*np.sqrt(12),(e/e.cummax()-1).min(),skew(x)
def row(nm,x):
    f=st(x); n=st(x,2011,2022); print(f"  {nm:26}{f[0]:>7.1%}{f[1]:>6.2f}{f[2]:>8.1%}{f[3]:>+6.2f}   |{n[0]:>7.1%}{n[1]:>6.2f}{n[2]:>8.1%}")
# static ERC baseline (2-asset ERC = inverse-vol)
def erc2(b):
    both=pd.DataFrame({"a":alpha,"b":b}).dropna(); va=both["a"].expanding(min_periods=24).std(); vbb=both["b"].expanding(min_periods=24).std()
    wb=(1/vbb)/((1/va)+(1/vbb)); return ((1-wb)*both["a"]+wb*both["b"]).dropna()
print("="*94); print("BETA SLEEVE UPGRADE — full 2011-26  | 2011-22 (normal regime)")
print(f"  {'variant':26}{'ann':>7}{'SR':>6}{'maxDD':>8}{'skew':>6}   |{'ann':>7}{'SR':>6}{'maxDD':>8}")
row("ALPHA only (META)",alpha)
row("BETA vol-managed",beta_static); row("BETA trend-scan",beta_trend)
print("  "+"-"*88)
row("alpha + beta-static ERC",erc2(beta_static))
row("alpha + beta-static STATE-M",statemachine(beta_static))
row("alpha + beta-trend ERC",erc2(beta_trend))
row("alpha + beta-trend STATE-M",statemachine(beta_trend))

# --- YEAR-BY-YEAR to see what the state machine does vs static ERC ---
cfg={"alpha":alpha,"beta-vm":beta_static,"ERC(stat)":erc2(beta_static),"STATE-M":statemachine(beta_static),"betaW(SM)":None}
# recover the state-machine beta weight path for transparency
both=pd.DataFrame({"a":alpha,"b":beta_static}).dropna(); idx=both.index; dp=dpct.reindex(idx); vp=vpct.reindex(idx)
wsm=pd.Series(np.nan,index=idx); sp=None; hist={}
for dt in idx:
    d,v=dp.loc[dt],vp.loc[dt]
    if not np.isfinite(d) or not np.isfinite(v): wsm.loc[dt]=0.5; continue
    s=(int(d>0.5),int(v>0.5))
    if sp is not None and abs(d-0.5)<0.10 and abs(v-0.5)<0.10: s=sp
    h=hist.get(s,[])
    if len(h)>=6:
        A=np.array([x[0] for x in h]);B=np.array([x[1] for x in h]);sa=A.mean()/(A.std()+1e-9);sb=B.mean()/(B.std()+1e-9)
        wsm.loc[dt]=float(np.clip(sb/(abs(sa)+abs(sb)+1e-9),0.1,0.9))
    else: wsm.loc[dt]=0.4
    hist.setdefault(s,[]).append((both.loc[dt,"a"],both.loc[dt,"b"])); sp=s
cfg["betaW(SM)"]=wsm
print("\nYEAR-BY-YEAR (returns; last col = avg state-machine BETA weight that year):")
ks=["alpha","beta-vm","ERC(stat)","STATE-M"]
print("  "+"yr".ljust(6)+"".join(f"{k:>11}" for k in ks)+f"{'wBeta(SM)':>11}")
for y in range(2011,2027):
    def yr(x): xx=x[[d.year==y for d in x.index]].dropna(); return (1+xx).prod()-1 if len(xx) else np.nan
    wb=wsm[[d.year==y for d in wsm.index]].mean()
    print(f"  {y:<6}"+"".join(f"{yr(cfg[k]):>11.1%}" for k in ks)+f"{wb:>11.2f}")
