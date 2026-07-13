#!/usr/bin/env python3
"""two_state.py — 2-STATE system from PREDICTED dispersion (+vol): lean ALPHA (neutral) when high-dispersion is
coming, lean BETA (net-long) in low-dispersion calm regimes. Vol prediction sizes the gross. Test vs static
neutral and static 60/40. 2011-2020 (news window), year-by-year. Let the data decide vs the static prior."""
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
spy = pd.read_parquet("/tmp/spy.parquet")["SPY"].reindex(me, method="ffill"); sret = spy.pct_change(); mvol = sret.rolling(21).std().reindex(me, method="ffill")*np.sqrt(21)
g = pd.read_parquet("data/fnspid/sent_monthly.parquet"); g["d"] = g["per"].map({d.to_period("M").__str__(): d for d in me}); g = g.dropna(subset=["d"])
sent = g.pivot_table(index="d", columns="sym", values="mean").reindex(index=me); cnt = g.pivot_table(index="d", columns="sym", values="count").reindex(index=me)
X = pd.DataFrame({"disp1":disp,"disp3":disp.rolling(3).mean(),"disp6":disp.rolling(6).mean(),"disp12":disp.rolling(12).mean(),
                 "mvol":mvol,"news_vol":np.log1p(cnt.sum(1)),"sent_neg":(sent<-0.1).sum(1)/cnt.notna().sum(1)}, index=me)
D = X.assign(y=disp.shift(-1)).dropna(); COLS=list(X.columns); pred=pd.Series(np.nan,index=me)
for i in range(36,len(D)):
    tr=D.iloc[:i]; A=np.column_stack([np.ones(len(tr))]+[tr[c].values for c in COLS]); b,*_=np.linalg.lstsq(A,tr["y"].values,rcond=None)
    pred.loc[D.index[i]]=np.concatenate([[1.0],[D[c].iloc[i] for c in COLS]])@b
dpct = pd.Series(pred.expanding(24).apply(lambda a:(a.iloc[-1]>a).mean()).values, index=me)   # predicted-disp percentile

al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
def tilt(W, netser):                                                        # per-month net-long fraction
    out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    for d in W.index:
        w=W.loc[d]; L=w.clip(lower=0); S=w.clip(upper=0); gl,gs=L.sum(),-S.sum(); n=float(netser.get(d,0.0))
        out.loc[d]=(L*((1+n)/gl)+S*((1-n)/gs)) if (gl>1e-9 and gs>1e-9) else w
    return out
def bt(W):
    r=BACKTEST.backtest(W,synth,freq=12,lag=0,signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9],transaction_cost=tc,borrow_fee=bf)["returns"]
    s=pd.Series(r); s.index=pd.DatetimeIndex(s.index); return s[(s.index>="2011-06-01")&(s.index<"2020-07-01")].dropna()
def show(name,x):
    x=x.dropna(); e=(1+x).cumprod()
    print(f"  {name:24} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  maxDD {(e/e.cummax()-1).min():>6.1%}  cSPY {pd.DataFrame({'a':x,'s':sret.reindex(x.index)}).dropna().corr().iloc[0,1]:>+5.2f}")
    print("    "+" ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2011,2021)))
# net-long series: 2-STATE = low predicted disp -> beta (0.35), high -> neutral (0.0); warmup -> static 0.2
net_dyn = pd.Series(np.where(dpct.notna(), (0.35*(1-dpct)).clip(0,0.35), 0.20), index=me).fillna(0.20)
print("2-STATE (predicted-dispersion -> lean alpha/beta) vs STATIC — 2011-2020")
show("static neutral (50/50)", bt(al))
show("static 60/40 net-long", bt(tilt(al, pd.Series(0.20,index=me))))
show("2-STATE dyn (disp->beta)", bt(tilt(al, net_dyn)))
print(f"\n  dyn net-long: flat yrs (2013,16) {net_dyn[[d.year in (2013,2016) for d in me]].mean():.2f}  vs high-disp (2011,18) {net_dyn[[d.year in (2011,2018) for d in me]].mean():.2f}")
