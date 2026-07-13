#!/usr/bin/env python3
"""ALLTEST.py — the MASTER TEST LOOP. Pulls every sleeve we have, all from the ONE DataHub frame, and runs the
whole portfolio test: per-sleeve stats -> correlation (diversification) map -> ensemble (static ERC + breadth-tilt).

Heavy ML sleeves (MOM/DM) save weights/returns to /tmp when their scripts run; ALLTEST consumes those. Light
sleeves (MR, VALUE, BETA) are computed inline from the hub. Monthly basis (daily MR aggregated to monthly) so all
sleeves are comparable and ensemble-able. NO external loaders — `from DATAHUB import DataHub` only.

Run the sleeves first (python MOM.py; python DM.py) to refresh /tmp weights, then: python ALLTEST.py
"""
import warnings; warnings.filterwarnings("ignore")
import os, pickle, numpy as np, pandas as pd
from DATAHUB import DataHub
import BACKTEST, BETANEUT

hub = DataHub(start="2000-01-01", min_days=0)
me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
def M(x):                                                                      # any daily/dated series -> monthly period-end
    s = pd.Series(x).dropna(); s.index = pd.DatetimeIndex(s.index); return s

# ---- pull sleeves ----
streams = {}
# MOM / DM : consume saved net-return streams (from running MOM.py / DM.py)
if os.path.exists("/tmp/mom_champ.pkl"):
    s = M(pickle.load(open("/tmp/mom_champ.pkl","rb"))["n1"]); streams["MOM"] = s
if os.path.exists("/tmp/dm_returns.pkl"):
    streams["DM"] = M(pickle.load(open("/tmp/dm_returns.pkl","rb")))

# BETA : vol-managed SPY (rule, monthly) — computed from hub-adjacent spy cache
if os.path.exists("/tmp/spy.parquet"):
    spm = pd.read_parquet("/tmp/spy.parquet")["SPY"].resample("ME").last().pct_change()
    fvol = spm.rolling(6, min_periods=3).std().shift(1)*np.sqrt(12)
    streams["BETA"] = ((0.15/fvol).clip(0.25,3.0).fillna(1.0)*spm).dropna()

# VALUE : rule z(B/M)+z(E/P), decile L/S, beta-neut (TENTATIVE until clean mcap lands)
bm, ep = hub.bm("monthly"), hub.ep("monthly")
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
val = (zc(bm).fillna(0)*bm.notna()+zc(ep).fillna(0)*ep.notna())/(bm.notna().astype(float)+ep.notna().astype(float)).replace(0,np.nan)
Wv = pd.DataFrame(0.0, index=me, columns=m_px.columns)
for d in me:
    s = val.where(elig).loc[d].dropna()
    if len(s) < 80: continue
    n=max(1,int(len(s)*0.10)); Wv.loc[d, s.nlargest(n).index]=1./n; Wv.loc[d, s.nsmallest(n).index]=-1./n
Wv = BETANEUT.betaneut(Wv, BETA)
rv = BACKTEST.backtest(Wv.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in me if Wv.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)
streams["VALUE*"] = M(rv["returns"])

# MR : QP reversion (rarity + 200d trend gate + event-hold) on daily hub, aggregated to monthly
px = hub.px_d; z = ((px/px.shift(3)-1) - (px/px.shift(3)-1).rolling(63,min_periods=30).mean())/((px/px.shift(3)-1).rolling(63,min_periods=30).std()+1e-9)
up = px > hub.sma_d(200); ed = hub.elig("relaxed","daily")
entry = ((z<-2)&up&ed).astype(float) - ((z>2)&(~up)&ed).astype(float)
held = entry.rolling(10, min_periods=1).sum(); L=held.clip(lower=0); S=held.clip(upper=0)
book = L.div(L.sum(1).replace(0,np.nan),axis=0).fillna(0.0) + S.div((-S.sum(1)).replace(0,np.nan),axis=0).fillna(0.0)
_,mr_net,_ = hub.backtest_daily(book, H=10, cost_bps=5.0)
streams["MR"] = M((1+mr_net).resample("ME").prod()-1)

# ---- align monthly, report ----
for k in streams: streams[k].index = pd.DatetimeIndex([pd.Timestamp(x).to_period("M").to_timestamp("M") for x in streams[k].index])
R = pd.DataFrame(streams)
def st(x, w0="2011-06-01", w1="2023-01-01"):
    x=x[(x.index>=w0)&(x.index<w1)].dropna(); e=(1+x).cumprod()
    return x.mean()/x.std()*np.sqrt(12), (1+x).prod()**(12/len(x))-1, (e/e.cummax()-1).min()
if hub.spy_m is not None: R["SPY(B&H)"] = hub.spy_m                              # market benchmark (we want up when SPY up)
print("="*80); print("ALLTEST — every sleeve + SPY benchmark, one DataHub frame (net, 2011-2022 honest window)")
print(f"  {'sleeve':10}{'SR':>6}{'ann':>8}{'maxDD':>8}{'corr-SPY':>9}   (* tentative; SPY = buy&hold benchmark)")
spy = R.get("SPY(B&H)")
for k in R.columns:
    sr,ann,dd = st(R[k]); cs = R[[k,"SPY(B&H)"]].dropna().corr().iloc[0,1] if spy is not None and k!="SPY(B&H)" else 1.0
    print(f"  {k:10}{sr:>6.2f}{ann:>8.1%}{dd:>8.1%}{cs:>+9.2f}")
print("\n  correlation map:"); print("   " + R.dropna().corr().round(2).to_string().replace("\n","\n   "))
# per-year returns vs SPY — do we capture the up years?
print("\n  per-year returns (%) vs SPY buy-and-hold:")
yrs = sorted(set(R.index.year)); yrs = [y for y in yrs if 2011<=y<=2026]
hdr = "    year " + " ".join(f"{k[:6]:>7}" for k in R.columns); print(hdr)
for y in yrs:
    row = R[[d.year==y for d in R.index]]
    cells = " ".join(f"{((1+row[k].dropna()).prod()-1)*100:>+7.0f}" if row[k].notna().any() else f"{'—':>7}" for k in R.columns)
    print(f"    {y}  {cells}")

# ---- ensemble: static ERC vs breadth-tilt ----
core = R[[c for c in R.columns if not c.endswith("*")]].dropna()
def erc_w(C):
    w=np.ones(C.shape[1])/C.shape[1]
    for _ in range(400): m=C@w; wn=1/np.maximum(m,1e-12); wn/=wn.sum(); w=0.5*w+0.5*wn
    return w
W=pd.DataFrame(np.nan,index=core.index,columns=core.columns)
for i in range(24,len(core)): W.iloc[i]=erc_w(np.cov(core.iloc[:i].T.values)+1e-6*np.eye(core.shape[1]))
erc=(core*W).sum(1)[W.notna().all(1)]
sr,ann,dd=st(erc); print(f"\n  ENSEMBLE static-ERC (core sleeves): SR {sr:.2f}  ann {ann:.1%}  maxDD {dd:.1%}")
