#!/usr/bin/env python3
"""trendscan_long.py — LONG-ONLY trend-scan XGBoost sleeve (a directional 'smart-beta' successor to the beta sleeve).

TARGET (Lopez de Prado trend-scan, forward): at each month-end, scan FORWARD windows L in [21..252]d, OLS logP~time,
take L* with max |t-stat|; label = 1 if that best forward trend is UP & significant (t>THR), else 0. Predict it with
XGBoost from TREND-QUALITY features (the family that fits a trend target): momentum (ret/nret 3/6/12), trend R²,
MACD, 52wk-high, current backward trend-scan t-stat, volume trend/abnormal volume, last-month reversal control, vol.
Rank P(up-trend) -> LONG-ONLY top-25%, vol-managed (directional). Walk-forward, embargo 12mo, net via BACKTEST.py."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
from xgboost import XGBRegressor
from scipy.stats import norm
import BACKTEST

THR=2.0
px=pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t=px.columns.str.match(r"^Z[A-Z]ZZT$")|px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px=px.loc[:,~t]
vb=pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me=pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me=me[me>=pd.Timestamp("2010-12-01")]
m_px=px.reindex(me); mret=m_px.pct_change(fill_method=None).where(lambda z:z<1.0); synth=(1+mret.fillna(0.0)).cumprod()
mdv=(px*vb).resample("ME").sum().reindex(me,method="ffill"); cov=px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig=(m_px>5)&(cov>0.9)&(mdv>5e6); tc=BACKTEST.tiered_transaction_costs(mdv); Z=mdv*0.0
dret=px.pct_change(fill_method=None); vol_d=dret.ewm(span=63,min_periods=20).std(); dvd=px*vb
def at_me(f): return f.reindex(me,method="ffill")
def hl(s): return np.log(0.5)/np.log(1-1/s)

# ---- trend-scan t-stat helper (per window, vectorized across names) ----
logpx=np.log(px.values); posd=np.searchsorted(px.index.values,me.values,side="right")-1; N=px.shape[1]; WIN=[21,63,126,189,252]
def scan(fwd):
    best=np.zeros((len(me),N)); ba=np.zeros((len(me),N))
    for L in WIN:
        x=np.arange(L+1)-L/2.0; sxx=(x*x).sum(); xc=x[:,None]
        for k,p in enumerate(posd):
            a,b=(p,p+L) if fwd else (p-L,p)
            if a<0 or b>=len(logpx): continue
            Y=logpx[a:b+1,:]; sl=(xc*(Y-Y.mean(0))).sum(0)/sxx; res=Y-(Y.mean(0)+xc*sl)
            se=np.sqrt((res**2).sum(0)/max(L-1,1)/sxx); tv=np.where(se>0,sl/se,0.0)
            u=np.abs(tv)>ba[k]; best[k]=np.where(u,tv,best[k]); ba[k]=np.where(u,np.abs(tv),ba[k])
    return best
print("[ts] backward trend feature + multi-horizon forward-return targets ...", flush=True)
bwd_t=scan(False); bwd=pd.DataFrame(bwd_t,index=me,columns=px.columns)       # current (backward) trend strength = feature
def grank(a): r=pd.Series(a).rank(method="average"); return norm.ppf((r-0.5)/max(int(r.notna().sum()),1))
HZ=[3,6,12]; FWD={h:(m_px.shift(-h)/m_px-1).where(lambda z:z.abs()<5) for h in HZ}   # forward h-month returns (multi-horizon)

# ---- TREND-QUALITY feature panel ----
logpm=np.log(m_px); mi=pd.Series(np.arange(len(m_px)),index=m_px.index); xd=mi-mi.rolling(6,min_periods=4).mean()
yd=logpm.sub(logpm.rolling(6,min_periods=4).mean(),axis=0); cxy=(yd.mul(xd,axis=0)).rolling(6,min_periods=4).mean()
r2=(cxy**2).div((xd**2).rolling(6,min_periods=4).mean(),axis=0).div(yd.pow(2).rolling(6,min_periods=4).mean()+1e-12)
comp=0.0
for Sh,Lg in [(8,24),(16,48),(32,96)]:
    q=(px.ewm(halflife=hl(Sh)).mean()-px.ewm(halflife=hl(Lg)).mean())/(px.rolling(63,min_periods=20).std()+1e-9)
    y=q/(q.rolling(252,min_periods=60).std()+1e-9); comp=comp+y*np.exp(-y**2/4)/0.89
PF={"ret3":at_me((px/px.shift(63)-1).where(lambda z:z.abs()<5)),"ret6":at_me((px/px.shift(126)-1).where(lambda z:z.abs()<5)),
    "ret12":at_me((px/px.shift(252)-1).where(lambda z:z.abs()<5)),"nret6":at_me((px/px.shift(126)-1)/(vol_d*np.sqrt(126)+1e-9)),
    "nret12":at_me((px/px.shift(252)-1)/(vol_d*np.sqrt(252)+1e-9)),"macd":at_me(comp),"trendR2":r2,
    "hi52":m_px/m_px.rolling(12,min_periods=8).max()-1,"bwdT":bwd,"vol":at_me(vol_d*np.sqrt(21)),
    "dvtrend":at_me(np.log(dvd.rolling(63,min_periods=40).mean()/dvd.rolling(252,min_periods=150).mean()+1e-12)),
    "abnvol":at_me(dvd.rolling(21,min_periods=15).mean()/dvd.rolling(252,min_periods=150).mean()),"ret1":mret,
    "accel":(m_px/m_px.shift(6)-1)-(m_px.shift(6)/m_px.shift(12)-1)}
FN=list(PF)
print(f"[ts] {len(FN)} trend-quality features: {FN}", flush=True)

# ---- walk-forward classifier -> P(up-trend) ----
T=len(me); EMB=12; store={}; mdl=None
rows=[]
for k in range(14,T-13):
    e=elig.iloc[k]; idx=px.columns[e.fillna(False).values]
    F=pd.DataFrame({c:PF[c].iloc[k].reindex(idx) for c in FN})
    Y=np.column_stack([grank(FWD[h].iloc[k].reindex(idx).values) for h in HZ])
    ok=F.notna().all(axis=1)&np.isfinite(Y).all(1)&mret.iloc[k+1].reindex(idx).notna().values
    rows.append((k,F[ok.values],Y[ok.values]))
pool={k:(F,Y) for k,F,Y in rows}; ks=sorted(pool); fp=next(k for k in ks if me[k].year>=2013)
P={}
for k in ks:
    if k<fp: continue
    if mdl is None or me[k].month in (1,7):
        tr=[j for j in ks if j<=k-EMB]
        if len(tr)>=24:
            X=pd.concat([pool[j][0] for j in tr]).values; Y=np.vstack([pool[j][1] for j in tr])
            mdl=[XGBRegressor(n_estimators=250,max_depth=4,learning_rate=0.05,subsample=0.8,colsample_bytree=0.8,multi_strategy="multi_output_tree",verbosity=0,random_state=s).fit(X,Y) for s in range(2)]
    if mdl is None: continue
    Fk=pool[k][0]; p=np.mean([m.predict(Fk.values).mean(1) for m in mdl],axis=0); P[k]=pd.Series(p,index=Fk.index)
imp=pd.Series(np.mean([m.feature_importances_ for m in mdl],axis=0),index=FN).sort_values(ascending=False)
print("[ts] feature importance: "+", ".join(f"{k} {v:.2f}" for k,v in imp.head(7).items()), flush=True)

# ---- LONG-ONLY top-25% by P(up-trend), vol-managed ----
spy=pd.read_parquet("/tmp/spy_long.parquet")["SPY"].dropna(); sret=spy.reindex(me,method="ffill").pct_change()
rv=(spy.pct_change().pow(2).groupby(spy.index.to_period("M")).sum()**0.5)*np.sqrt(21); rv.index=[pp.to_timestamp("M") for pp in rv.index]; rv=rv.reindex(me,method="ffill")
vm=(rv.expanding(min_periods=24).median()/rv).clip(upper=2.0)
W=pd.DataFrame(0.0,index=me,columns=px.columns)
for k,s in P.items():
    top=s.nlargest(max(1,int(len(s)*0.25))).index; W.loc[me[k],top]=1.0/len(top)
Wv=W.mul(vm.shift(1),axis=0)
def bt(Wx):
    r=BACKTEST.backtest(Wx.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in Wx.index if Wx.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=Z)
    s=pd.Series(r["returns"]); s.index=pd.DatetimeIndex(s.index); return s[s.index>="2011-01-01"].dropna()
lo=bt(Wv); spym=sret[sret.index>="2011-01-01"].dropna()
def st(x,a=2011,b=2026):
    x=x[[a<=d.year<=b for d in x.index]].dropna(); e=(1+x).cumprod(); return (1+x).prod()**(12/len(x))-1,x.mean()/x.std()*np.sqrt(12),(e/e.cummax()-1).min(),skew(x)
def row(n,x):
    f=st(x); nm=st(x,2011,2022); c=pd.DataFrame({'a':x,'s':spym}).dropna().corr().iloc[0,1]
    print(f"  {n:26}{f[0]:>7.1%}{f[1]:>6.2f}{f[2]:>8.1%}{f[3]:>+6.2f}{c:>+7.2f} |{nm[0]:>7.1%}{nm[1]:>6.2f}{nm[2]:>8.1%}")
print("\nLONG-ONLY MULTI-HORIZON REGRESSOR (trend-quality feats, top-25%, vol-mgd) — full 2011-26 | 2011-22")
print(f"  {'variant':26}{'ann':>7}{'SR':>6}{'maxDD':>8}{'skew':>6}{'cSPY':>7} |{'ann':>7}{'SR':>6}{'maxDD':>8}")
row("LO trend-MH-regressor",lo); row("SPY buy&hold",spym)
al=pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index=pd.DatetimeIndex(pd.to_datetime(al.index)); al=al[~al.index.duplicated()].reindex(index=me,columns=px.columns).fillna(0.0)
ar=BACKTEST.backtest(al,synth,freq=12,lag=0,signal_dates=[d for d in al.index if al.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=BACKTEST.tiered_borrow_fees(mdv))["returns"]
alpha=pd.Series(ar); alpha.index=pd.DatetimeIndex(alpha.index); alpha=alpha[alpha.index>="2011-01-01"].dropna()
both=pd.DataFrame({"a":alpha,"b":lo}).dropna(); va=both["a"].expanding(24).std(); vbb=both["b"].expanding(24).std(); wb=(1/vbb)/((1/va)+(1/vbb))
row("ALPHA(META) only",alpha); row("alpha + LO-trendscan ERC",((1-wb)*both["a"]+wb*both["b"]).dropna())
print(f"\n  corr(alpha, LO-trendscan)={pd.DataFrame({'a':alpha,'b':lo}).dropna().corr().iloc[0,1]:+.2f}")
for y in range(2011,2027):
    x=lo[[d.year==y for d in lo.index]]; s=spym[[d.year==y for d in spym.index]]
    print(f"    {y}: LO-TS {(1+x).prod()-1:>7.1%}   SPY {(1+s).prod()-1:>7.1%}")
