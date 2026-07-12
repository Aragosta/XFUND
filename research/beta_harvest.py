#!/usr/bin/env python3
"""beta_harvest.py — how to harvest MARKET BETA effectively. Analyze the return distribution, dispersion, and
regime structure of the equity premium to find the efficient way to capture it.

Sections:
  1. DISTRIBUTION of monthly market returns (SPY cap-wt + eligible-universe equal-wt): moments, tails, asymmetry.
  2. VOL-MANAGED harvest (Moreira-Muir): does scaling exposure by 1/vol beat buy-and-hold? (the key test)
  3. VOL-QUINTILE: forward market return & Sharpe by trailing-vol bucket -> where is the beta Sharpe concentrated?
  4. DISPERSION regimes: cross-sectional return dispersion vs forward market return & vs our CS-alpha blend.
  5. TIMEABILITY: autocorrelation of monthly market returns + breadth (fraction above trend) as a predictor."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from scipy.stats import skew, kurtosis

spy=pd.read_parquet("/tmp/spy_long.parquet")["SPY"].dropna()
last=spy.groupby(spy.index.to_period("M")).apply(lambda s:s.index[-1]); me=pd.DatetimeIndex(last.values)
mp=spy.loc[me]; sret=mp.pct_change()
rv=(spy.pct_change().pow(2).groupby(spy.index.to_period("M")).sum()**0.5); rv.index=me; rv=rv*np.sqrt(21)
def ann_sr(r): r=r.dropna(); return (r.mean()*12)/(r.std()*np.sqrt(12)) if r.std()>0 else np.nan
def mdd(r): e=(1+r.dropna()).cumprod(); return (e/e.cummax()-1).min()

# ---- 1. distribution ----
r11=sret[sret.index>="2011-01-01"].dropna(); rall=sret[sret.index>="1994-01-01"].dropna()
print("="*80); print("1. MARKET RETURN DISTRIBUTION (SPY monthly)")
for tag,r in [("1994+",rall),("2011+",r11)]:
    print(f"  {tag}: mean {r.mean()*12:+.1%}/yr  vol {r.std()*np.sqrt(12):.1%}  SR {ann_sr(r):.2f}  skew {skew(r):+.2f}"
          f"  kurt {kurtosis(r):+.1f}  %neg {(r<0).mean():.0%}  worst {r.min():.1%}  best {r.max():.1%}")

# ---- 2. vol-managed (Moreira-Muir): exposure = target/trailing_vol ----
print("\n2. VOL-MANAGED HARVEST — scale exposure by 1/vol (Moreira-Muir). Does it beat buy&hold?")
volsig=rv.shift(1)                                                           # trailing vol known at t
for tag,start in [("1994+","1994-01-01"),("2011+","2011-01-01")]:
    bh=sret[sret.index>=start].dropna()
    tgt=volsig[volsig.index>=start].median()
    exp=(tgt/volsig).clip(upper=2.5)                                        # vol-target exposure, capped
    vm=(exp*sret).dropna(); vm=vm[vm.index>=start]
    print(f"  {tag}:  buy&hold SR {ann_sr(bh):.2f} (maxDD {mdd(bh):.0%})   vol-managed SR {ann_sr(vm):.2f} (maxDD {mdd(vm):.0%})"
          f"   delta {ann_sr(vm)-ann_sr(bh):+.2f}")

# ---- 3. vol quintiles: where is the beta Sharpe? ----
print("\n3. VOL-QUINTILE — trailing vol bucket -> NEXT-month market return & Sharpe (where beta pays)")
df=pd.DataFrame({"vol":volsig,"fwd":sret}).dropna(); df=df[df.index>="1994-01-01"]
df["q"]=pd.qcut(df["vol"],5,labels=["Q1 low","Q2","Q3","Q4","Q5 high"])
for q,g in df.groupby("q",observed=True):
    print(f"  {q:8}: avg trail-vol {g['vol'].mean():.0%}  ->  fwd ret {g['fwd'].mean()*12:+.1%}/yr  SR {ann_sr(g['fwd']):.2f}  %neg {(g['fwd']<0).mean():.0%}")

# ---- 4. dispersion regimes (needs stock panel) ----
print("\n4. DISPERSION REGIMES — cross-sectional return dispersion vs forward market & vs our CS-alpha")
px=pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t=px.columns.str.match(r"^Z[A-Z]ZZT$")|px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px=px.loc[:,~t]
vb=pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
mep=pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); mep=mep[mep>=pd.Timestamp("2010-12-01")]
mpx=px.reindex(mep); mr=mpx.pct_change(fill_method=None).where(lambda z:z<1.0)
mdv=(px*vb).resample("ME").sum().reindex(mep,method="ffill"); cov=px.notna().rolling(252,min_periods=200).mean().reindex(mep,method="ffill")
elig=(mpx>5)&(cov>0.9)&(mdv>5e6)
disp=mr.where(elig).std(axis=1); mkt_ew=mr.where(elig).mean(axis=1)
blend=pd.Series(pickle.load(open("/tmp/exec_full_streams.pkl","rb"))["ERC"]); blend.index=pd.DatetimeIndex(blend.index)
D=pd.DataFrame({"disp":disp.shift(1),"mkt_fwd":mkt_ew,"blend":blend.reindex(mep)}).dropna(); D=D[D.index>="2011-01-01"]
D["q"]=pd.qcut(D["disp"],3,labels=["low disp","mid","high disp"])
for q,g in D.groupby("q",observed=True):
    print(f"  {q:10}: mkt fwd SR {ann_sr(g['mkt_fwd']):.2f} ({g['mkt_fwd'].mean()*12:+.1%})   CS-alpha blend SR {ann_sr(g['blend']):.2f} ({g['blend'].mean()*12:+.1%})")
print(f"  corr(dispersion, fwd market ret): {D['disp'].corr(D['mkt_fwd']):+.2f}   corr(dispersion, blend ret): {D['disp'].corr(D['blend']):+.2f}")

# ---- 5. timeability ----
print("\n5. TIMEABILITY — market-return autocorrelation + breadth predictor")
for lag in (1,3,6,12):
    ac=pd.concat([sret, sret.shift(lag)],axis=1).dropna(); ac=ac[ac.index>="1994-01-01"]
    print(f"  autocorr(market, lag {lag:2}m): {ac.corr().iloc[0,1]:+.2f}", end="   ")
print()
breadth=(mpx>mpx.rolling(10).mean()).where(elig).mean(axis=1)               # % above 10mo MA
B=pd.DataFrame({"breadth":breadth.shift(1),"fwd":mkt_ew}).dropna(); B=B[B.index>="2011-01-01"]
print(f"  corr(breadth, fwd market ret): {B['breadth'].corr(B['fwd']):+.2f}   (breadth>median fwd SR {ann_sr(B[B['breadth']>B['breadth'].median()]['fwd']):.2f} vs <median {ann_sr(B[B['breadth']<=B['breadth'].median()]['fwd']):.2f})")
