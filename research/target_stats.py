"""target_stats.py — statistical analysis of candidate VALUE targets: are they DENOISED (learnable), GAUSSIAN,
and PERSISTENT? A good regression target has (1) high learnable signal (cross-sectional R^2 of target on current
fundamentals — high = denoised, low = noise), (2) a near-Gaussian distribution (low |skew|, kurtosis~3 — else
rank-transform), (3) persistence (autocorr — fundamentals persist, returns don't). Compare price-return targets
vs fundamental targets (profitability level & improvement)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.stats import skew, kurtosis
from DATAHUB import DataHub
hub = DataHub(); me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid"); f = hub.fund
mcap = hub.mcap()
gpa = ((f("rev")-f("cogs"))/f("assets")).where(f("assets")>0); roe = (f("ni")/f("book")).where(f("book")>0)
# candidate targets (all FORWARD / at prediction horizon)
H=6; logp=np.log(m_px); xs=np.arange(H); xm=xs.mean(); sxx=((xs-xm)**2).sum(); Tr=pd.DataFrame(np.nan,index=me,columns=m_px.columns)
for i in range(len(me)-H):
    y=logp.iloc[i+1:i+1+H].values; yb=np.nanmean(y,0); sl=np.nansum((xs[:,None]-xm)*(y-yb),0)/sxx
    r=y-(yb+np.outer(xs-xm,sl)); se=np.sqrt(np.nansum(r**2,0)/(H-2)/sxx); Tr.iloc[i]=sl/(se+1e-9)
targets = {
    "fwd 6m return":      m_px.shift(-6)/m_px - 1,
    "trend-scan (ret)":   Tr,
    "future GP/A":        gpa.shift(-12),
    "future ROE":         roe.shift(-12),
    "improve GP/A (dGPA)":gpa.shift(-12)-gpa,
    "improve ROE (dROE)": roe.shift(-12)-roe,
}
# learnability: cross-sectional R^2 of target on current fundamental features (in-sample proxy)
FEAT = {"bm":f("book")/mcap,"ep":f("ni")/mcap,"gpa":gpa,"roe":roe,"sp":f("rev")/mcap,
        "ag":f("assets")/f("assets").shift(12)-1,"lev":f("debt_lt")/f("assets"),"size":np.log(mcap)}
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-4,4)
FZ={k:zc(v.replace([np.inf,-np.inf],np.nan)) for k,v in FEAT.items()}
def stats(tg):
    sk,ku,r2,per=[],[],[],[]
    for d in me:
        y=tg.loc[d].where(elig.loc[d]).replace([np.inf,-np.inf],np.nan).dropna()
        if len(y)<100: continue
        sk.append(skew(y)); ku.append(kurtosis(y, fisher=False))               # fisher=False -> normal kurt=3
        X=pd.DataFrame({k:FZ[k].loc[d] for k in FZ}).reindex(y.index).fillna(0.0).values
        A=np.column_stack([np.ones(len(y)),X]); b,*_=np.linalg.lstsq(A,y.values,rcond=None)
        pred=A@b; ss=1-((y.values-pred)**2).sum()/(((y.values-y.mean())**2).sum()+1e-12); r2.append(ss)
    # persistence: cross-sectional corr(target_t, target_{t-12})
    for d in me[12:]:
        a=tg.loc[d].where(elig.loc[d]); bb=tg.shift(12).loc[d]; df=pd.DataFrame({"a":a,"b":bb}).dropna()
        if len(df)>100: per.append(df["a"].corr(df["b"]))
    return np.nanmean(sk),np.nanmean(ku),np.nanmean(r2),np.nanmean(per)
print("="*84); print("VALUE TARGET STATISTICS — denoised (R2) / Gaussian (skew,kurt) / persistent (autocorr)")
print(f"  {'target':22}{'skew':>7}{'kurt':>7}{'learn R2':>10}{'persist':>9}   verdict")
for nm,tg in targets.items():
    sk,ku,r2,pe=stats(tg)
    gauss = "gaussian" if abs(sk)<0.7 and abs(ku-3)<3 else "fat/skew"
    dn = "DENOISED" if r2>0.15 else ("mid" if r2>0.05 else "noisy")
    print(f"  {nm:22}{sk:>+7.2f}{ku:>7.1f}{r2:>10.3f}{pe:>+9.2f}   {dn}, {gauss}")
