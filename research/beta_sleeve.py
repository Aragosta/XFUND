#!/usr/bin/env python3
"""beta_sleeve.py — add a VOL-MANAGED INDEX (beta) sleeve, allocate MOM/DM/beta by DISPERSION. Beat the blend?

From beta_harvest.py: (a) beta Sharpe is concentrated in LOW-vol regimes -> vol-manage it (scale by 1/vol);
(b) direction is NOT timeable (autocorr ~0); (c) our CS-alpha is weak in LOW-dispersion regimes (SR 0.46) but
spectacular in HIGH-dispersion (2.58) — so lean BETA when dispersion is low, ALPHA when high. All leak-free.
  beta sleeve  = SPY * exposure, exposure = median(vol)/trailing_vol (capped)          [vol-managed static long]
  allocation   = ERC(MOM,DM,beta), then tilt beta weight up in low-dispersion regimes  [dispersion-conditional]
Compare: MOM+DM (2-way) vs +beta static ERC vs +beta dispersion-tilted. Net, 2011+, yearly for the flat years."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from scipy.stats import skew
import BACKTEST, sys; sys.path.insert(0,"research"); import allocation as A

BW=60
px=pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t=px.columns.str.match(r"^Z[A-Z]ZZT$")|px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px=px.loc[:,~t]
vb=pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me=pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me=me[me>=pd.Timestamp("2010-12-01")]
m_px=px.reindex(me); mret=m_px.pct_change(fill_method=None).where(lambda z:z<1.0); synth=(1+mret.fillna(0.0)).cumprod()
mdv=(px*vb).resample("ME").sum().reindex(me,method="ffill"); cov=px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig=(m_px>5)&(cov>0.9)&(mdv>5e6); tc=BACKTEST.tiered_transaction_costs(mdv); bf=BACKTEST.tiered_borrow_fees(mdv); meP=pd.PeriodIndex(me,freq="M")
mkt=mret.where(elig).mean(axis=1)
mr_=mret.rolling(BW,min_periods=36).mean(); mm=mkt.rolling(BW,min_periods=36).mean()
mrm=mret.mul(mkt,axis=0).rolling(BW,min_periods=36).mean(); vm=(mkt**2).rolling(BW,min_periods=36).mean()-mm**2
BETA=mrm.sub(mr_.mul(mm,axis=0)).div(vm,axis=0).shift(1)

def to_grid(W):
    W=W.copy(); W.index=pd.PeriodIndex(pd.DatetimeIndex(W.index),freq="M"); W=W[~W.index.duplicated()].reindex(meP); W.index=me
    return W.reindex(columns=px.columns).astype(float)
def neutralize(w,dt):
    b=BETA.loc[dt].reindex(w.index).fillna(1.0).values; wv=w.values-b*((w.values@b)/(b@b+1e-9)); g=np.abs(wv).sum()
    return pd.Series(wv*(2.0/g) if g>0 else wv,index=w.index)
def bn(W):
    W=W.reindex(columns=synth.columns); rows={dt:neutralize(W.loc[dt].dropna(),dt) for dt in W.index if dt in me}
    return pd.DataFrame(rows).T.reindex(columns=synth.columns).fillna(0.0)
def stream(W):
    r=BACKTEST.backtest(W.fillna(0.0),synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)["returns"]
    s=pd.Series(r); s.index=pd.PeriodIndex(s.index,freq="M"); return s
mom=stream(bn(to_grid(pickle.load(open("/tmp/mom_weights.pkl","rb")))))
dm =stream(bn(to_grid(pickle.load(open("/tmp/mhdm_weights_s5_base.pkl","rb")))))

# ---- vol-managed SPY beta sleeve (leak-free) ----
spy=pd.read_parquet("/tmp/spy_long.parquet")["SPY"].dropna()
sret=spy.reindex(me,method="ffill").pct_change()
rv=(spy.pct_change().pow(2).groupby(spy.index.to_period("M")).sum()**0.5)*np.sqrt(21); rv.index=[p.to_timestamp("M") for p in rv.index]
rv=rv.reindex(me,method="ffill"); tgt=rv.expanding(min_periods=24).median()             # leak-free target vol
expo=(tgt/rv).clip(upper=2.5)                                                            # vol-managed exposure
beta_ret=(expo.shift(1)*sret) - 0.0001*expo.diff().abs().fillna(0)                       # earns t->t+1, token cost
beta=pd.Series(beta_ret.values,index=meP)

# ---- dispersion (leak-free trailing) ----
disp=mret.where(elig).std(axis=1); disp_pct=disp.shift(1).expanding(min_periods=24).apply(lambda a: (a.iloc[-1]>a).mean(),raw=False)
disp_pct=pd.Series(disp_pct.values,index=meP)

def rep(name, r):
    r=r.dropna(); r=r[r.index>=pd.Period("2011-01")]
    sr=r.mean()/r.std()*np.sqrt(12); ann=(1+r).prod()**(12/len(r))-1; e=(1+r).cumprod(); dd=(e/e.cummax()-1).min()
    y={yr:(1+r[[p.year==yr for p in r.index]]).prod()-1 for yr in (2013,2016,2017,2020,2022)}
    print(f"  {name:26}{sr:>6.2f}{ann:>8.1%}{dd:>8.1%}{skew(r):>+7.2f}   "+" ".join(f"{yr}:{v:>+6.1%}" for yr,v in y.items()),flush=True)
    return r

streams=pd.DataFrame({"MOM":mom,"DM":dm,"BETA":beta}).dropna()
def erc_combine(cols, beta_tilt=0.0):
    s=streams[cols]; Aw=A.expanding_alloc(s,method="erc",win=36)
    if beta_tilt>0 and "BETA" in cols:                                                   # tilt beta up in LOW-dispersion regimes
        mult=np.exp(beta_tilt*(0.5-disp_pct.reindex(Aw.index).fillna(0.5))); Aw=Aw.copy(); Aw["BETA"]=Aw["BETA"]*mult
        Aw=Aw.div(Aw.sum(axis=1),axis=0)
    return (s*Aw.reindex(s.index)).sum(axis=1), Aw.mean()

print("="*118); print("BETA SLEEVE + DISPERSION-CONDITIONAL ALLOCATION (net, 2011+, return-level ERC)")
print(f"  {'strategy':26}{'SR':>6}{'ann':>8}{'maxDD':>8}{'skew':>7}   flat-year participation (2013/16/17/20/22)")
rep("MOM (beta-neut)", mom); rep("DM (beta-neut)", dm); rep("BETA (vol-managed SPY)", beta)
print("  "+"-"*112)
c2,w2=erc_combine(["MOM","DM"]);          rep("ERC MOM+DM (2-way)", c2)
c3,w3=erc_combine(["MOM","DM","BETA"]);   rep("ERC +BETA (static tilt=0)", c3)
for g in (1.0,2.0,3.0,4.0):
    cg,wg=erc_combine(["MOM","DM","BETA"],beta_tilt=g); rep(f"ERC +BETA (disp-tilt={g:.0f})", cg)
c4,w4=erc_combine(["MOM","DM","BETA"],beta_tilt=2.0)
pickle.dump({"MOM":mom,"DM":dm,"BETA":beta,"disp_pct":disp_pct,"c3":c3}, open("/tmp/sleeve_streams.pkl","wb"))
print(f"\n  avg weights 3-way static : {dict(w3.round(2))}")
print(f"  corr(BETA, MOM+DM 2-way) : {pd.DataFrame({'b':beta,'c':c2}).dropna().corr().iloc[0,1]:+.2f}")

# ---- YEAR-BY-YEAR: 3-way champion vs 2-way vs SPY (why is total ann return ~equal?) ----
spym=pd.Series(sret.values,index=meP)
def ddpath(r): e=(1+r.fillna(0)).cumprod(); return e/e.cummax()-1
def yblk(r):
    r=r.dropna();
    if len(r)<2: return (np.nan,np.nan,np.nan,np.nan)
    d=ddpath(r); return ((1+r).prod()-1, r.mean()/r.std()*np.sqrt(12) if r.std()>0 else np.nan, d.mean(), d.min())
C3=c3.dropna(); C3=C3[C3.index>=pd.Period("2011-01")]; C2=c2.dropna(); C2=C2[C2.index>=pd.Period("2011-01")]
SP=spym.reindex(C3.index)
print("\n"+"="*94); print("YEAR-BY-YEAR — 3-way champion (MOM+DM+BETA) vs 2-way vs SPY"); print("="*94)
print(f"  {'year':6}| {'3-way ret':>10}{'Sharpe':>8}{'maxDD':>8} | {'2-way ret':>10} | {'SPY ret':>9}")
for y in sorted({p.year for p in C3.index}):
    m=[p.year==y for p in C3.index]; r3,s3,_,dd3=yblk(C3[m]); r2,_,_,_=yblk(C2[m]); rs,_,_,_=yblk(SP.reindex(C3.index)[m])
    print(f"  {y:6}| {r3:>10.1%}{s3:>8.2f}{dd3:>8.1%} | {r2:>10.1%} | {rs:>9.1%}")
a3=yblk(C3); a2=yblk(C2); asp=yblk(SP)
def annz(r): r=r.dropna(); return (1+r).prod()**(12/len(r))-1
print("  "+"-"*84)
print(f"  {'ALL':6}| {annz(C3):>10.1%}{a3[1]:>8.2f}{a3[3]:>8.1%} | {annz(C2):>10.1%} | {annz(SP):>9.1%}   (ann return / SR / maxDD)")
print(f"\n  => 3-way and 2-way earn ~equal ANN RETURN ({annz(C3):.1%} vs {annz(C2):.1%}) but 3-way SR {a3[1]:.2f} vs {a2[1]:.2f}:")
print(f"     the beta sleeve REDISTRIBUTES return from big-alpha years into the flat years -> smoother path, higher Sharpe.")
