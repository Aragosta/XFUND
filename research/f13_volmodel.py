#!/usr/bin/env python3
"""f13_volmodel.py — is the 13F->vol signal INCREMENTAL, or just a size proxy? Decides if 13F has real
construction value. Target = fwd realized vol. Compare cross-sectional OOS R2 of vol forecasts:
  (A) trailing vol only   (B) trailing vol + size   (C) trailing vol + size + 13F breadth/value.
If C >> B, 13F carries idiosyncratic-vol info beyond price+size -> a genuine risk-model input (Door 2 done right)."""
import warnings; warnings.filterwarnings("ignore")
import glob, json, numpy as np, pandas as pd

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2012-01-01")]
m_px = px.reindex(me); mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6)
dvol = px.pct_change().rolling(21).std().resample("ME").last().reindex(me)*np.sqrt(21)
fvol = dvol.shift(-1)                                                          # target: next-month vol
size = np.log(mdv)

c2t = json.load(open("data/13f/cusip_ticker.json")); up = {c.upper(): c for c in px.columns}
rows = []
for f in sorted(glob.glob("data/13f/parsed/*.parquet")):
    d = pd.read_parquet(f, columns=["cik","period","cusip","value"]); d["tic"] = d["cusip"].str.slice(0,9).map(c2t)
    d = d.dropna(subset=["tic"]); d["tic"] = d["tic"].str.upper().map(up); d = d.dropna(subset=["tic"])
    rows.append(d.groupby(["period","tic"]).agg(hold=("cik","nunique"), val=("value","sum")).reset_index())
P = pd.concat(rows, ignore_index=True); P["pend"] = pd.to_datetime(P["period"], format="%d-%b-%Y", errors="coerce"); P = P.dropna(subset=["pend"])
HOLD = P.pivot_table(index="pend", columns="tic", values="hold").reindex(columns=px.columns)
VAL  = P.pivot_table(index="pend", columns="tic", values="val").reindex(columns=px.columns)
def mpit(Q): Qs = Q.copy(); Qs.index = [pd.Timestamp(p)+pd.Timedelta(days=50) for p in Q.index]; return Qs.reindex(me, method="ffill")
breadth = mpit(np.log1p(HOLD)); value = mpit(np.log1p(VAL))

def zc(df): return df.sub(df.mean(axis=1),axis=0).div(df.std(axis=1)+1e-9,axis=0)                        # cross-sectional z
Zv, Zs, Zb, Zval, Zy = zc(dvol), zc(size), zc(breadth), zc(value), zc(fvol)
def oos_r2(feats):
    pred = pd.Series(dtype=float); act = pd.Series(dtype=float)
    for d in me[me>=pd.Timestamp("2014-06-01")]:
        # pooled cross-sectional OLS trained on all months strictly before d
        tr = []
        for e in me[me<d]:
            m = elig.loc[e]; X = pd.DataFrame({k:f.loc[e] for k,f in feats.items()}); X["y"]=Zy.loc[e]; X=X[m].dropna(); tr.append(X)
        tr = pd.concat(tr);
        if len(tr)<500: continue
        A = np.column_stack([np.ones(len(tr))]+[tr[k].values for k in feats]); b,*_=np.linalg.lstsq(A,tr["y"].values,rcond=None)
        m = elig.loc[d]; Xc = pd.DataFrame({k:f.loc[d] for k,f in feats.items()}); Xc["y"]=Zy.loc[d]; Xc=Xc[m].dropna()
        if len(Xc)<40: continue
        p = np.column_stack([np.ones(len(Xc))]+[Xc[k].values for k in feats])@b
        pred=pd.concat([pred,pd.Series(p,index=Xc.index)]); act=pd.concat([act,Xc["y"]])
    return 1 - ((act-pred)**2).sum()/((act-act.mean())**2).sum()
print("VOL FORECAST — cross-sectional OOS R2 (target = next-month realized vol, z-scored):")
print(f"  (A) trailing vol            R2 = {oos_r2({'v':Zv}):+.4f}")
print(f"  (B) trailing vol + size     R2 = {oos_r2({'v':Zv,'s':Zs}):+.4f}")
print(f"  (C) + 13F breadth + value   R2 = {oos_r2({'v':Zv,'s':Zs,'b':Zb,'val':Zval}):+.4f}")
# incremental: IC of breadth on the residual vol after removing trailing vol + size
resid = {}
for d in me:
    m=elig.loc[d]; X=pd.DataFrame({'v':Zv.loc[d],'s':Zs.loc[d],'y':Zy.loc[d]})[m].dropna()
    if len(X)<40: continue
    A=np.column_stack([np.ones(len(X)),X['v'],X['s']]); b,*_=np.linalg.lstsq(A,X['y'],rcond=None)
    resid[d]=X['y']-A@b
ics=[pd.Series(resid[d]).corr(Zb.loc[d].reindex(resid[d].index),method='spearman') for d in resid if Zb.loc[d].reindex(resid[d].index).notna().sum()>40]
ics=np.array(ics); print(f"\n  incremental IC(13F breadth, vol residual | trailing-vol+size) = {np.nanmean(ics):+.3f}  t {np.nanmean(ics)/(np.nanstd(ics)/np.sqrt(len(ics))):+.1f}")
