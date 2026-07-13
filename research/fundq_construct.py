"""fundq_construct.py — how to CONSTRUCT the value/quality book? Decile map says value edge is asymmetric
(short concentrated, long diffuse). Run the VxQ walk once, save the score, then sweep construction:
symmetric decile/quintile/tercile/half, and ASYMMETRIC (broad long / narrow short). Report SR/DD/turnover."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor
from DATAHUB import DataHub
import BACKTEST, BETANEUT
hub = DataHub(); me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
sec = hub.sector; tc=BACKTEST.tiered_transaction_costs(hub.mdv); bf=BACKTEST.tiered_borrow_fees(hub.mdv)
BETA=BETANEUT.rolling_beta(mret,elig,bw=60); f=hub.fund; mcap=hub.mcap()
gpa=((f("rev")-f("cogs"))/f("assets")).where(f("assets")>0); roe=(f("ni")/f("book")).where(f("book")>0); bm=f("book")/mcap
def d12(x): return x/x.shift(12)-1
feat={"bm":bm,"ep":f("ni")/mcap,"sp":f("rev")/mcap,"cfp":f("ocf")/mcap,"gpa":gpa,"roe":roe,"roa":f("ni")/f("assets"),
 "ocfa":f("ocf")/f("assets"),"gmar":f("gross_profit")/f("rev"),"ag":d12(f("assets")),"capexa":f("capex")/f("assets"),
 "accr":(f("ni")-f("ocf"))/f("assets"),"lev":f("debt_lt")/f("assets"),"curr":f("assets_cur")/f("liab_cur"),
 "casha":f("cash")/f("assets"),"revg":d12(f("rev")),"nig":d12(f("ni")),"size":np.log(mcap)}
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
F={k:zc(v.replace([np.inf,-np.inf],np.nan)) for k,v in feat.items()}; COLS=list(F)
def grank(a): r=pd.Series(a).rank(method="average"); return norm.ppf((r-0.5)/max(len(r),2))
EMB=12; fut_gpa=gpa.shift(-12); fut_roe=roe.shift(-12)
def secr(row,idx,d0): r=row.where(elig.loc[d0]); r=r-r.groupby(sec).transform("mean"); return grank(r.reindex(idx).values)
pool={}
for i,d in enumerate(me):
    live=elig.loc[d]&mcap.loc[d].notna(); idx=live[live].index
    if len(idx)<100 or i+EMB>=len(me): continue
    X=np.column_stack([F[c].loc[d].reindex(idx).fillna(0.0).values for c in COLS])
    Y=np.column_stack([secr(fut_gpa.loc[d],idx,d),secr(fut_roe.loc[d],idx,d)])
    pool[d]=dict(X=X.astype(np.float32),idx=idx,Y=Y,bm=bm.loc[d].reindex(idx))
dates=[d for d in me if d in pool]
REG=dict(n_estimators=200,max_depth=5,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,tree_method="hist",multi_strategy="multi_output_tree",verbosity=0)
SCORE=pd.DataFrame(np.nan,index=me,columns=m_px.columns); mdl=None
for j,d in enumerate(dates):
    if j<60: continue
    if mdl is None or j%3==0:
        tr=[dates[k] for k in range(j) if dates[k]<=me[max(0,me.get_loc(d)-EMB)]]
        if len(tr)<48: continue
        Xtr=np.vstack([pool[t]["X"] for t in tr]); Ytr=np.vstack([pool[t]["Y"] for t in tr]); ok=np.isfinite(Ytr).all(1)
        mdl=XGBRegressor(**REG).fit(Xtr[ok],Ytr[ok])
    q=mdl.predict(pool[d]["X"]).mean(1); s=pd.Series(q,index=pool[d]["idx"]); b=pool[d]["bm"]
    s=(s-s.mean())/(s.std()+1e-9)+(b.rank()-b.rank().mean())/(b.rank().std()+1e-9)   # value x quality
    s=s-s.groupby(sec.reindex(s.index)).transform("mean"); SCORE.loc[d,s.index]=s.values
SCORE.to_pickle("/tmp/fundq_score.pkl")
def build(ql,qs):                                                              # long top ql, short bottom qs
    W=pd.DataFrame(0.0,index=me,columns=m_px.columns)
    for d in me:
        s=SCORE.loc[d].dropna()
        if len(s)<80: continue
        nl=max(1,int(len(s)*ql)); ns=max(1,int(len(s)*qs))
        W.loc[d,s.nlargest(nl).index]+=1.0/nl; W.loc[d,s.nsmallest(ns).index]+=-1.0/ns
    return BETANEUT.betaneut(W,BETA)
def stat(W):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)
    x=pd.Series(r["returns"]); x.index=pd.DatetimeIndex(x.index); x=x[(x.index>="2016-01-01")&(x.index<"2027-01-01")].dropna(); e=(1+x).cumprod()
    return x.mean()/x.std()*np.sqrt(12),(e/e.cummax()-1).min(),r["ann_turnover"]
print("="*72); print("VxQ CONSTRUCTION SWEEP (2016-2026, sector-neut, beta-neut)")
print(f"  {'construction':26}{'SR':>6}{'maxDD':>8}{'turn':>7}")
for nm,ql,qs in [("decile 10/10",0.10,0.10),("quintile 20/20",0.20,0.20),("tercile 33/33",0.33,0.33),
                 ("top-half 50/50",0.50,0.50),("ASYM broad-long 40/10",0.40,0.10),("ASYM 30-long/10-short",0.30,0.10)]:
    sr,dd,tn=stat(build(ql,qs)); print(f"  {nm:26}{sr:>6.2f}{dd:>8.1%}{tn:>7.1f}")
