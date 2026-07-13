#!/usr/bin/env python3
"""model_zoo.py — 'what models do we need?' Build a DIVERSE factor zoo (momentum, short-reversal, long-term
reversal/value-proxy, low-vol, size, illiquidity) as monthly gross streams, then measure the MARGINAL value of
each family to the TILT-to-winner ensemble. Hypothesis: diverse models spanning different regimes make tilt an
implicit regime-rotation (always some winner to lean toward) -> more diversity = better tilt. Test incrementally."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
dret = px.pct_change(fill_method=None).where(lambda z: z.abs()<0.5)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2004-01-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z.abs()<0.8)
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig = (m_px>5)&(cov>0.9)&(mdv>5e6)
vol12 = mret.rolling(12, min_periods=6).std()
amihud = (dret.abs()/(px*vb).replace(0,np.nan)).resample("ME").mean().reindex(me)

def stream(sig, long_high, q=0.1):
    S = sig.where(elig); R = S.rank(axis=1, pct=True)
    top=(R>=1-q).astype(float); bot=(R<=q).astype(float)
    long_,short = (top,bot) if long_high else (bot,top)
    L=long_.div(long_.sum(1).replace(0,np.nan),axis=0); Sh=short.div(short.sum(1).replace(0,np.nan),axis=0)
    return ((L-Sh).fillna(0.0).shift(1)*mret).sum(axis=1)

fam = {}                                                                        # family -> {name: stream}
fam["MOM"]   = {f"mom{lb}_{s}_{int(q*100)}": stream(m_px.shift(s)/m_px.shift(lb)-1, True, q)
                for lb in (3,6,9,12) for s in (0,1) for q in (0.1,0.2)}
fam["REV"]   = {f"rev{lb}": stream(m_px.shift(0)/m_px.shift(lb)-1, False) for lb in (1,2,3)}
fam["LTREV"] = {f"ltrev{lb}": stream(m_px.shift(12)/m_px.shift(lb)-1, False) for lb in (36,48,60)}  # value proxy
fam["LOWVOL"]= {f"lowvol{q}": stream(vol12, False, q) for q in (0.1,0.2)}
fam["SIZE"]  = {"size_small": stream(np.log(mdv), False)}                       # long small
fam["ILLIQ"] = {f"illiq{q}": stream(amihud, True, q) for q in (0.1,0.2)}        # long illiquid

def perf(c,w0="2005-06-01"):
    x=c[(c.index>=w0)].dropna(); e=(1+x).cumprod()
    return x.mean()/x.std()*np.sqrt(12), (e/e.cummax()-1).min()
def tilt(STR, win=6, lam=10, warm=24):
    STR=STR.loc["2005-01-01":].dropna(how="all"); n=STR.shape[1]; W=pd.DataFrame(np.nan,index=STR.index,columns=STR.columns)
    for i in range(warm,len(STR)):
        H=STR.iloc[:i]; cr=(1+H.iloc[-win:]).prod().values-1; e=np.exp(lam*(cr-np.nanmean(cr))); W.iloc[i]=e/np.nansum(e)
    c=(STR*W).sum(axis=1); return c[W.notna().all(axis=1)]
def eq(STR):
    STR=STR.loc["2005-01-01":].dropna(how="all"); return STR.mean(axis=1)

# per-family standalone SR (tilted within family)
print("PER-FAMILY (standalone, tilted within family):")
for k,d in fam.items():
    STR=pd.DataFrame(d); s,dd=perf(tilt(STR)); print(f"  {k:8} n={len(d):>2}  tilt SR {s:>5.2f}  maxDD {dd:>6.1%}  avg-corr {STR.corr().values[np.triu_indices(len(d),1)].mean():+.2f}")

# incremental: add families one at a time, static vs tilt
order=["MOM","REV","LTREV","LOWVOL","SIZE","ILLIQ"]; acc={}
print("\nINCREMENTAL ENSEMBLE (add family -> static equal vs 6mo-tilt):")
print(f"  {'+family':10}{'#mods':>6}{'static SR':>11}{'tilt SR':>9}{'tilt maxDD':>12}")
for k in order:
    acc.update(fam[k]); STR=pd.DataFrame(acc)
    es,_=perf(eq(STR)); ts,td=perf(tilt(STR))
    print(f"  {k:10}{STR.shape[1]:>6}{es:>11.2f}{ts:>9.2f}{td:>12.1%}")
