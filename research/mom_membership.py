#!/usr/bin/env python3
"""mom_membership.py — POC: MOM-style rank-space REGRESSION but with a TOP/BOTTOM-MEMBERSHIP target (DM's target
concept, MOM's regressor). Decile-map showed momentum's edge is in the EXTREMES, so a tail-focused target may beat
the smooth Gaussian-rank-of-returns target. Same features/engine/walk; only the TARGET differs:
  A) return-rank  : grank(fwd_h return)            (MOM's current target)
  B) membership   : +1 top-decile / -1 bottom / 0  (tail-focused)
Light POC: top-1000 liquid, HZ=(4,5,6), retrain yearly, seeds=1. Compare net SR."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor
from DATAHUB import DataHub
import BACKTEST, BETANEUT
hub = DataHub(); me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
vol6 = mret.rolling(6, min_periods=3).std()
feat = {"r3": m_px/m_px.shift(3)-1, "r6": m_px/m_px.shift(6)-1, "r12": m_px.shift(1)/m_px.shift(12)-1,
        "nr6": (m_px/m_px.shift(6)-1)/(vol6+1e-6), "hi52": m_px/m_px.rolling(12,min_periods=6).max()}
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
F = {k: zc(v.replace([np.inf,-np.inf],np.nan)) for k,v in feat.items()}; COLS=list(F)
HZ=(4,5,6); EMB=max(HZ)
def grank(a): r=pd.Series(a).rank(method="average"); return norm.ppf((r-0.5)/max(len(r),2))
def member(a):                                                                 # +1 top decile / -1 bottom / 0 else
    r=pd.Series(a).rank(pct=True); return np.where(r>=0.9,1.0,np.where(r<=0.1,-1.0,0.0))
def trendscan(Wd):                                                             # forward Wd-month trend t-stat (embargo=EMB below)
    logp=np.log(m_px); xs=np.arange(Wd); xm=xs.mean(); sxx=((xs-xm)**2).sum(); T=pd.DataFrame(np.nan,index=me,columns=m_px.columns)
    for i in range(len(me)-Wd):
        y=logp.iloc[i+1:i+1+Wd].values; yb=np.nanmean(y,0); sl=np.nansum((xs[:,None]-xm)*(y-yb),0)/sxx
        r=y-(yb+np.outer(xs-xm,sl)); se=np.sqrt(np.nansum(r**2,0)/(Wd-2)/sxx); T.iloc[i]=sl/(se+1e-9)
    return T
TS={h:trendscan(h) for h in HZ}
# top-1000 liquid each month
liq = hub.mdv.where(elig)
pool={}
for i,d in enumerate(me):
    if i+EMB>=len(me): continue
    idx = liq.loc[d].dropna().nlargest(1000).index
    if len(idx)<200: continue
    X=np.column_stack([F[c].loc[d].reindex(idx).fillna(0.0).values for c in COLS])
    fwd={h: mret.iloc[i+h].reindex(idx).values for h in HZ}
    pool[d]=dict(X=X.astype(np.float32), idx=idx,
                 A={h:grank(fwd[h]) for h in HZ}, B={h:member(fwd[h]) for h in HZ},
                 C={h:grank(TS[h].loc[d].reindex(idx).values) for h in HZ})     # trend-scan target
dates=[d for d in me if d in pool]
REG=dict(n_estimators=150,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method="hist",multi_strategy="multi_output_tree",verbosity=0)
def run(tgt):
    W=pd.DataFrame(0.0,index=me,columns=m_px.columns); mdl=None
    for j,d in enumerate(dates):
        if j<60: continue
        if mdl is None or j%12==0:
            tr=[dates[k] for k in range(j) if dates[k]<=me[max(0,me.get_loc(d)-EMB)]]
            if len(tr)<48: continue
            Xtr=np.vstack([pool[t]["X"] for t in tr]); Ytr=np.column_stack([np.concatenate([pool[t][tgt][h] for t in tr]) for h in HZ])
            ok=np.isfinite(Ytr).all(1); mdl=XGBRegressor(**REG); mdl.fit(Xtr[ok],Ytr[ok])
        p=mdl.predict(pool[d]["X"]).mean(1); s=pd.Series(p,index=pool[d]["idx"])
        n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    W=BETANEUT.betaneut(W,BETA)
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); return x
def show(nm,x):
    x=x[(x.index>="2013-01-01")&(x.index<"2023-01-01")].dropna(); e=(1+x).cumprod()
    print(f"  {nm:26} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}")
print("="*74); print("MOM TARGET POC — return-rank vs top/bottom-membership (top-1000, 2013-2022)")
show("A) return-rank (MOM now)", run("A")); show("B) membership (tail-focus)", run("B")); show("C) trend-scan target", run("C"))
