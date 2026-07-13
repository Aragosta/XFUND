#!/usr/bin/env python3
"""fund_trend.py — value/fundamental ML with a TREND-SCANNING target (vs FUND's sector-neutral rank RETURN target).
Target = de Prado trend-scan: signed t-stat of forward log-price~time slope over windows 6/9/12mo, sector-neutral,
Gaussian-ranked. FORWARD-LOOKING label, so EMBARGO = max trend window (12mo): training excludes any sample whose
label window reaches the prediction date -> NO look-ahead. Same features/engine/construction as FUND -> clean A/B
of the TARGET only (return-rank vs trend-scan)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import norm
from xgboost import XGBRegressor
from DATAHUB import DataHub
import BACKTEST, BETANEUT
hub = DataHub(); me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
sec = hub.sector; tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60); f = hub.fund; mcap = hub.mcap()
def d12(x): return x/x.shift(12)-1
feat = {"bm":f("book")/mcap,"ep":f("ni")/mcap,"sp":f("rev")/mcap,"cfp":f("ocf")/mcap,"dp":f("dividends")/mcap,
 "gpa":(f("gross_profit")/f("assets")).where(f("assets")>0),"roe":(f("ni")/f("book")).where(f("book")>0),
 "roa":f("ni")/f("assets"),"ocfa":f("ocf")/f("assets"),"gmar":f("gross_profit")/f("rev"),
 "ag":d12(f("assets")),"capexa":f("capex")/f("assets"),"accr":(f("ni")-f("ocf"))/f("assets"),
 "drec":f("receivables").diff(12)/f("assets"),"dinv":f("inventory").diff(12)/f("assets"),
 "lev":f("debt_lt")/f("assets"),"curr":f("assets_cur")/f("liab_cur"),"casha":f("cash")/f("assets"),
 "revg":d12(f("rev")),"nig":d12(f("ni")),"size":np.log(mcap)}
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
F={k:zc(v.replace([np.inf,-np.inf],np.nan)) for k,v in feat.items()}; COLS=list(F)

def trendscan(Wd):                                                             # forward Wd-month trend t-stat (LOOK-FORWARD label)
    logp=np.log(m_px); xs=np.arange(Wd); xm=xs.mean(); sxx=((xs-xm)**2).sum(); T=pd.DataFrame(np.nan,index=me,columns=m_px.columns)
    for i in range(len(me)-Wd):
        y=logp.iloc[i+1:i+1+Wd].values; yb=np.nanmean(y,0); sl=np.nansum((xs[:,None]-xm)*(y-yb),0)/sxx
        r=y-(yb+np.outer(xs-xm,sl)); se=np.sqrt(np.nansum(r**2,0)/(Wd-2)/sxx); T.iloc[i]=sl/(se+1e-9)
    return T
HZ=(6,9,12); EMB=max(HZ); TS={w:trendscan(w) for w in HZ}                       # embargo = max window -> no leak
def grank(a): r=pd.Series(a).rank(method="average"); return norm.ppf((r-0.5)/max(len(r),2))
def secn(row,idx):                                                             # sector-neutralize + rank a target row
    r=row.where(elig.loc[d0]); r=r-r.groupby(sec).transform("mean"); return grank(r.reindex(idx).values)
pool={}
for i,d in enumerate(me):
    live=elig.loc[d]&mcap.loc[d].notna(); idx=live[live].index
    if len(idx)<100 or i+EMB>=len(me): continue
    d0=d; X=np.column_stack([F[c].loc[d].reindex(idx).fillna(0.0).values for c in COLS])
    Y={w: secn(TS[w].loc[d], idx) for w in HZ}
    pool[d]=dict(X=X.astype(np.float32),idx=idx,Y=Y)
dates=[d for d in me if d in pool]
REG=dict(n_estimators=200,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method="hist",multi_strategy="multi_output_tree",verbosity=0)
W=pd.DataFrame(0.0,index=me,columns=m_px.columns); mdl=None
for j,d in enumerate(dates):
    if j<60: continue
    if mdl is None or j%3==0:
        tr=[dates[k] for k in range(j) if dates[k]<=me[max(0,me.get_loc(d)-EMB)]]      # EMBARGO: exclude labels reaching d
        if len(tr)<48: continue
        Xtr=np.vstack([pool[t]["X"] for t in tr]); Ytr=np.column_stack([np.concatenate([pool[t]["Y"][w] for t in tr]) for w in HZ])
        ok=np.isfinite(Ytr).all(1); mdl=XGBRegressor(**REG); mdl.fit(Xtr[ok],Ytr[ok])
    p=mdl.predict(pool[d]["X"]).mean(1); s=pd.Series(p,index=pool[d]["idx"]); s=s-s.groupby(sec.reindex(s.index)).transform("mean")
    n=max(1,int(len(s)*0.10)); W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
W=BETANEUT.betaneut(W,BETA)
r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); pickle.dump(r["returns"],open("/tmp/fundtrend_returns.pkl","wb"))
xx=x[(x.index>="2013-01-01")&(x.index<"2027-01-01")].dropna(); e=(1+xx).cumprod()
print("="*72); print("VALUE ML with TREND-SCAN target (embargo=12mo, no look-ahead) — net")
print(f"  FUND-trend  SR {xx.mean()/xx.std()*np.sqrt(12):.2f}  ann {(1+xx).prod()**(12/len(xx))-1:.1%}  maxDD {(e/e.cummax()-1).min():.1%}")
print("    "+" ".join(f"{y}:{(1+xx[[d.year==y for d in xx.index]]).prod()-1:>+4.0%}" for y in range(2013,2027)))
import os
for nm,pth,key in [("FUND-return","/tmp/fund_returns.pkl",None),("MOM","/tmp/mom_champ.pkl","n1"),("DM","/tmp/dm_returns.pkl",None)]:
    if os.path.exists(pth):
        o=pickle.load(open(pth,"rb")); s=pd.Series(o[key] if key else o); s.index=pd.DatetimeIndex(s.index)
        print(f"  corr(FUND-trend, {nm}) = {pd.DataFrame({'a':x,'b':s}).dropna().corr().iloc[0,1]:+.2f}")
