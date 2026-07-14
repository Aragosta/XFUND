#!/usr/bin/env python3
"""FUNDQ.py — VALUE x QUALITY ML. Target = FUTURE PROFITABILITY (GP/A + ROE, +12mo, sector-neutral, rank-
transformed) — the denoised/persistent target our stats picked (learn R2 0.3-0.57 vs 0.009 for returns). Features
= fundamentals. Model predicts which firms will be MORE profitable; we trade CHEAP x high-predicted-quality (the
mispricing lever). Orthogonal to price momentum by construction (no price in features OR target). EMBARGO=12mo
(future fundamentals) -> no look-ahead. Year-by-year vs SPY buy-and-hold; corr to MOM/DM/FUND."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
from scipy.stats import norm
from xgboost import XGBRegressor
from DATAHUB import DataHub
import BACKTEST, BETANEUT
hub = DataHub(); me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
sec = hub.sector; tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60); f = hub.fund; mcap = hub.mcap()
gpa = ((f("rev")-f("cogs"))/f("assets")).where(f("assets")>0); roe = (f("ni")/f("book")).where(f("book")>0)
bm = f("book")/mcap                                                             # cheapness (for the value tilt)
def d12(x): return x/x.shift(12)-1
feat = {"bm":bm,"ep":f("ni")/mcap,"sp":f("rev")/mcap,"cfp":f("ocf")/mcap,"gpa":gpa,"roe":roe,
 "roa":f("ni")/f("assets"),"ocfa":f("ocf")/f("assets"),"gmar":f("gross_profit")/f("rev"),
 "ag":d12(f("assets")),"capexa":f("capex")/f("assets"),"accr":(f("ni")-f("ocf"))/f("assets"),
 "lev":f("debt_lt")/f("assets"),"curr":f("assets_cur")/f("liab_cur"),"casha":f("cash")/f("assets"),
 "revg":d12(f("rev")),"nig":d12(f("ni")),"size":np.log(mcap)}
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
F={k:zc(v.replace([np.inf,-np.inf],np.nan)) for k,v in feat.items()}; COLS=list(F)
def grank(a): r=pd.Series(a).rank(method="average"); return norm.ppf((r-0.5)/max(len(r),2))
EMB=12                                                                         # future fundamentals -> 12mo embargo
fut_gpa=gpa.shift(-12); fut_roe=roe.shift(-12)                                 # FORWARD profitability labels
def secr(row,idx):
    r=row.where(elig.loc[d0]); r=r-r.groupby(sec).transform("mean"); return grank(r.reindex(idx).values)
pool={}
for i,d in enumerate(me):
    live=elig.loc[d]&mcap.loc[d].notna(); idx=live[live].index
    if len(idx)<100 or i+EMB>=len(me): continue
    d0=d; X=np.column_stack([F[c].loc[d].reindex(idx).fillna(0.0).values for c in COLS])
    Y=np.column_stack([secr(fut_gpa.loc[d],idx), secr(fut_roe.loc[d],idx)])     # predict future profitability
    pool[d]=dict(X=X.astype(np.float32),idx=idx,Y=Y,bm=bm.loc[d].reindex(idx))
dates=[d for d in me if d in pool]
REG=dict(n_estimators=200,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method="hist",multi_strategy="multi_output_tree",verbosity=0)
def book(vq):                                                                  # vq=True -> tilt by cheapness (value x quality)
    W=pd.DataFrame(0.0,index=me,columns=m_px.columns); mdl=None
    for j,d in enumerate(dates):
        if j<60: continue
        if mdl is None or j%3==0:
            tr=[dates[k] for k in range(j) if dates[k]<=me[max(0,me.get_loc(d)-EMB)]]
            if len(tr)<48: continue
            Xtr=np.vstack([pool[t]["X"] for t in tr]); Ytr=np.vstack([pool[t]["Y"] for t in tr])
            ok=np.isfinite(Ytr).all(1); mdl=XGBRegressor(**REG).fit(Xtr[ok],Ytr[ok])
        q=mdl.predict(pool[d]["X"]).mean(1); s=pd.Series(q,index=pool[d]["idx"])       # predicted future profitability
        if vq:                                                                 # value x quality: add cheapness rank
            b=pool[d]["bm"]; s=(s-s.mean())/(s.std()+1e-9) + (b.rank()-b.rank().mean())/(b.rank().std()+1e-9)
        s=s-s.groupby(sec.reindex(s.index)).transform("mean")                  # sector-neutral
        n=max(1,int(len(s.dropna())*0.10)); s=s.dropna()
        W.loc[d,s.nlargest(n).index]=1.0/n; W.loc[d,s.nsmallest(n).index]=-1.0/n
    W=BETANEUT.betaneut(W,BETA)
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=1,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); return x
spy=hub.spy_m
def show(nm,x):
    x=x[(x.index>="2013-01-01")&(x.index<"2027-01-01")].dropna(); e=(1+x).cumprod()
    print(f"  {nm:24} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}")
    print("    "+" ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+4.0%}" for y in range(2013,2027)))
print("="*84); print("VALUE x QUALITY ML (predict future profitability, trade cheap x quality) — net")
q=book(False); vq=book(True)
show("quality only", q); show("VALUE x QUALITY", vq)
sp=spy[(spy.index>="2013-01-01")&(spy.index<"2027-01-01")].dropna()
print("  SPY buy-and-hold (benchmark):")
print("    "+" ".join(f"{y}:{(1+sp[[d.year==y for d in sp.index]]).prod()-1:>+4.0%}" for y in range(2013,2027)))
print(f"    SPY SR {sp.mean()/sp.std()*np.sqrt(12):.2f}  ann {(1+sp).prod()**(12/len(sp))-1:.1%}")
pickle.dump(vq.rename(None).to_dict() if False else list(vq.items()), open("/tmp/fundq_returns.pkl","wb"))
for nm,pth,key in [("MOM","/tmp/mom_champ.pkl","n1"),("DM","/tmp/dm_returns.pkl",None),("FUND(ret-target)","/tmp/fund_returns.pkl",None)]:
    if os.path.exists(pth):
        o=pickle.load(open(pth,"rb")); s=pd.Series(o[key] if key else o); s.index=pd.DatetimeIndex(s.index)
        print(f"  corr(VxQ, {nm}) = {pd.DataFrame({'a':vq,'b':s}).dropna().corr().iloc[0,1]:+.2f}")
