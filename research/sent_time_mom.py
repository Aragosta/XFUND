#!/usr/bin/env python3
"""sent_time_mom.py — does AGGREGATE (time-series) news sentiment TIME the momentum sleeve?
NOT market-direction timing (dead). Timing the momentum FACTOR's payoff — a documented effect:
Antoniou-Doukas-Subrahmanyam (momentum stronger after HIGH sentiment), Cooper et al (momentum only after UP
markets), Daniel-Moskowitz (crashes forecastable by bear+high-vol). Features known at t: aggregate sentiment
level & change, fraction-negative, news volume, past-12m market return (bull/bear), market vol. Target = next-month
MOM sleeve return. Report predictive t-stats + an OOS timing overlay (scale MOM exposure by predicted state) vs
static MOM. Also test the SAME overlay on DM and on the combined alpha."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
mkt = mret.mean(axis=1); mkt12 = (1+mkt).rolling(12).apply(np.prod)-1                                    # past-12m market
spy = pd.read_parquet("/tmp/spy.parquet")["SPY"].reindex(me, method="ffill"); mvol = spy.pct_change().rolling(21).std().reindex(me,method="ffill")*np.sqrt(21)

# ---- aggregate news features (from cache) ----
g = pd.read_parquet("data/fnspid/sent_monthly.parquet"); g["d"]=g["per"].map({d.to_period("M").__str__():d for d in me}); g=g.dropna(subset=["d"])
sent=g.pivot_table(index="d",columns="sym",values="mean").reindex(index=me); cnt=g.pivot_table(index="d",columns="sym",values="count").reindex(index=me)
agg_sent = (sent*cnt).sum(1)/cnt.sum(1)                                                                  # volume-weighted market sentiment
frac_neg = (sent<-0.1).sum(1)/sent.notna().sum(1); news_vol=np.log1p(cnt.sum(1))
sent_chg = agg_sent - agg_sent.rolling(3).mean()                                                         # sentiment momentum/shock

# ---- sleeve return streams (net) ----
M = pickle.load(open("/tmp/mom_champ.pkl","rb")); MOM=pd.Series(M["n1"]); MOM.index=pd.DatetimeIndex(MOM.index)
DMr = pickle.load(open("/tmp/dm_returns.pkl","rb")); DM=pd.Series(DMr); DM.index=pd.DatetimeIndex(DM.index)
def align(s): s=s.copy(); s.index=s.index.to_period("M"); return s
streams={"MOM":MOM,"DM":DM,"MOM+DM":(align(MOM)+align(DM)).dropna()}
for k in streams:
    s=streams[k]; s.index=(s.index.to_timestamp("M") if isinstance(s.index,pd.PeriodIndex) else pd.DatetimeIndex(s.index)); s.index=pd.DatetimeIndex([pd.Timestamp(x).to_period("M").to_timestamp("M") for x in s.index]); streams[k]=s

F = pd.DataFrame({"sent":agg_sent,"sent_chg":sent_chg,"frac_neg":frac_neg,"news_vol":news_vol,"mkt12":mkt12,"mvol":mvol}, index=me)
Fm = F.copy(); Fm.index=[d.to_period("M").to_timestamp("M") for d in Fm.index]

print("="*96); print("DOES AGGREGATE NEWS SENTIMENT TIME MOMENTUM? predictive t-stats (feat_t -> sleeve_ret_{t+1})")
def tstat(x,y):
    d=pd.DataFrame({"x":x,"y":y}).dropna();
    if len(d)<40: return np.nan,0
    b=np.polyfit(d["x"],d["y"],1)[0]; r=d["x"].corr(d["y"]); return r, r/ (np.sqrt((1-r**2)/(len(d)-2))+1e-9)
for k,s in streams.items():
    print(f"\n  target = {k} return_(t+1):")
    for f in F.columns:
        r,tt=tstat(Fm[f].reindex(s.index), s.shift(-1).reindex(s.index)); print(f"    {f:10} corr {r:>+6.2f}  t {tt:>+5.1f}")

# ---- OOS timing overlay: predict next-month sleeve return from features, scale exposure ----
def timing(s, feats, win=48):
    idx=[d for d in s.index if d in Fm.index]; s=s.reindex(idx); X=Fm.reindex(idx)[feats]
    scale=pd.Series(1.0,index=idx)
    for i in range(win,len(idx)):
        tr=pd.DataFrame(X.iloc[:i]); tr["y"]=s.shift(-1).iloc[:i].values; tr=tr.dropna()
        if len(tr)<36: continue
        A=np.column_stack([np.ones(len(tr))]+[tr[c].values for c in feats]); b,*_=np.linalg.lstsq(A,tr["y"].values,rcond=None)
        pred=np.concatenate([[1.0],X.iloc[i].values])@b
        scale.iloc[i]=np.clip(0.5+ (pred/ (s.iloc[:i].std()+1e-9)), 0.0, 1.5)                            # more exposure when predicted-good
    return (scale.shift(1).fillna(1.0)*s)
def perf(x,lab):
    x=x.dropna(); x=x[(x.index>="2011-06-01")&(x.index<"2023-01-01")]; sr=x.mean()/x.std()*np.sqrt(12)
    print(f"    {lab:28} SR {sr:>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}")
    return sr
print("\n"+"="*96); print("OOS SENTIMENT-TIMING OVERLAY vs STATIC (net, 2011-2022):")
for k,s in streams.items():
    print(f"  {k}:")
    perf(s,"static")
    perf(timing(s,["sent","sent_chg","frac_neg"]),"news-sentiment timing")
    perf(timing(s,["mkt12","mvol"]),"market-state timing (Cooper/DM)")
    perf(timing(s,["sent","sent_chg","frac_neg","mkt12","mvol"]),"news + market-state timing")
