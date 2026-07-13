#!/usr/bin/env python3
"""multitarget_switch.py — JOINT vol+dispersion 'turbulence' regime -> switch alpha(neutral)/beta(net-long).
Predict BOTH next-month market vol AND cross-sectional dispersion (the two forecastable 2nd moments); combine into
a turbulence percentile. LOW turbulence (calm) -> lean BETA (net-long, capture the melt-up); HIGH turbulence ->
lean ALPHA (neutral, harvest dispersion). vs static neutral / static 60/40 / dispersion-only. 2013-2020."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
disp = mret.where(elig).std(axis=1)
spy = pd.read_parquet("/tmp/spy.parquet")["SPY"].reindex(me, method="ffill"); sret = spy.pct_change()
rvol = sret.pow(2).groupby([me.year, me.month]).transform("sum")  # placeholder
mvol = (sret.rolling(21).std()*np.sqrt(21)).reindex(me, method="ffill")
g = pd.read_parquet("data/fnspid/sent_monthly.parquet"); g["d"] = g["per"].map({d.to_period("M").__str__(): d for d in me}); g = g.dropna(subset=["d"])
sent = g.pivot_table(index="d", columns="sym", values="mean").reindex(index=me); cnt = g.pivot_table(index="d", columns="sym", values="count").reindex(index=me)
newsvol = np.log1p(cnt.sum(1)); sentneg = (sent < -0.1).sum(1)/cnt.notna().sum(1)

def predict(y, feats, warm=24):
    D = pd.concat([feats, y.rename("y")], axis=1).dropna(); COLS = list(feats.columns); out = pd.Series(np.nan, index=me)
    for i in range(warm, len(D)):
        tr = D.iloc[:i]; A = np.column_stack([np.ones(len(tr))]+[tr[c].values for c in COLS]); b,*_ = np.linalg.lstsq(A, tr["y"].values, rcond=None)
        out.loc[D.index[i]] = np.concatenate([[1.0],[D[c].iloc[i] for c in COLS]]) @ b
    return out
Fvol  = pd.DataFrame({"v1":mvol,"v3":mvol.rolling(3).mean(),"v6":mvol.rolling(6).mean(),"nv":newsvol}, index=me)
Fdisp = pd.DataFrame({"d1":disp,"d3":disp.rolling(3).mean(),"d6":disp.rolling(6).mean(),"mv":mvol,"nv":newsvol,"sn":sentneg}, index=me)
pv = predict(mvol.shift(-1), Fvol); pd_ = predict(disp.shift(-1), Fdisp)
def zc(s): return (s - s.expanding(12).mean())/(s.expanding(12).std()+1e-9)
turb = (zc(pv) + zc(pd_)) / 2                                                # joint turbulence
tpct = pd.Series(turb.expanding(12).apply(lambda a:(a.iloc[-1]>a).mean()).values, index=me)
dpct = pd.Series(pd_.expanding(12).apply(lambda a:(a.iloc[-1]>a).mean()).values, index=me)

al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
def tilt(W, netser):
    out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    for d in W.index:
        w=W.loc[d]; L=w.clip(lower=0); S=w.clip(upper=0); gl,gs=L.sum(),-S.sum(); n=float(netser.get(d,0.0))
        out.loc[d]=(L*((1+n)/gl)+S*((1-n)/gs)) if (gl>1e-9 and gs>1e-9) else w
    return out
def bt(W):
    r=BACKTEST.backtest(W,synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)["returns"]
    s=pd.Series(r); s.index=pd.DatetimeIndex(s.index); return s[(s.index>="2013-01-01")&(s.index<"2020-07-01")].dropna()
def show(name,x):
    x=x.dropna(); e=(1+x).cumprod()
    print(f"  {name:26} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  maxDD {(e/e.cummax()-1).min():>6.1%}  cSPY {pd.DataFrame({'a':x,'s':sret.reindex(x.index)}).dropna().corr().iloc[0,1]:>+5.2f}")
    print("    "+" ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2013,2021)))
nturb = pd.Series(np.where(tpct.notna(),(0.35*(1-tpct)).clip(0,0.35),0.20), index=me).fillna(0.20)   # low turbulence -> beta
ndisp = pd.Series(np.where(dpct.notna(),(0.35*(1-dpct)).clip(0,0.35),0.20), index=me).fillna(0.20)
print("MULTI-TARGET (vol+dispersion) TURBULENCE SWITCH — 2013-2020")
show("static neutral", bt(al)); show("static 60/40", bt(tilt(al, pd.Series(0.20,index=me))))
show("dispersion-only switch", bt(tilt(al, ndisp))); show("JOINT turbulence switch", bt(tilt(al, nturb)))
print(f"\n  joint net-long: calm yrs (2013,16,17) {nturb[[d.year in (2013,2016,2017) for d in me]].mean():.2f}  vs turbulent (2015,18,20) {nturb[[d.year in (2015,2018,2020) for d in me]].mean():.2f}")
