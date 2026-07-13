#!/usr/bin/env python3
"""dispersion_size.py — size the ALPHA by PREDICTED dispersion (news-enhanced). Up when high-disp coming (harvest),
down in low-disp junk-rally regimes (dodge). Uses the cached FNSPID sentiment. Backtest 2011-2020 (news window),
year-by-year vs constant-gross alpha."""
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
spy = pd.read_parquet("/tmp/spy.parquet")["SPY"].reindex(me, method="ffill"); mvol = spy.pct_change().rolling(21).std().reindex(me, method="ffill")*np.sqrt(21)
g = pd.read_parquet("data/fnspid/sent_monthly.parquet")
g["d"] = g["per"].map({d.to_period("M").__str__(): d for d in me}); g = g.dropna(subset=["d"])
sent = g.pivot_table(index="d", columns="sym", values="mean").reindex(index=me); cnt = g.pivot_table(index="d", columns="sym", values="count").reindex(index=me)
X = pd.DataFrame({"disp1":disp,"disp3":disp.rolling(3).mean(),"disp6":disp.rolling(6).mean(),"disp12":disp.rolling(12).mean(),
                 "mvol":mvol,"news_vol":np.log1p(cnt.sum(1)),"sent_neg":(sent<-0.1).sum(1)/cnt.notna().sum(1)}, index=me)
y = disp.shift(-1); COLS = list(X.columns)
D = X.assign(y=y).dropna()
pred = pd.Series(np.nan, index=me)
for i in range(36, len(D)):
    tr = D.iloc[:i]; A = np.column_stack([np.ones(len(tr))]+[tr[c].values for c in COLS]); b,*_ = np.linalg.lstsq(A, tr["y"].values, rcond=None)
    pred.loc[D.index[i]] = np.concatenate([[1.0],[D[c].iloc[i] for c in COLS]]) @ b
# gross multiplier from predicted dispersion (up in high-disp, down in low); known at t -> sizes t->t+1
mult = (pred / pred.expanding(min_periods=12).median()).clip(0.3, 1.8).fillna(1.0)  # warmup = constant gross (no artifact)

al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
r = BACKTEST.backtest(al, synth, freq=12, lag=0, signal_dates=[d for d in al.index if al.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)["returns"]
alpha = pd.Series(r); alpha.index = pd.DatetimeIndex(alpha.index)
sized = (mult.shift(1) * alpha).dropna()                                    # dispersion-timed sizing
def win(x): x=x.dropna(); return x[(x.index>="2011-06-01")&(x.index<"2020-07-01")]
def show(name, x):
    x=win(x); e=(1+x).cumprod()
    print(f"  {name:26} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}")
    print("    " + " ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2011,2021)))
print("DISPERSION-SIZED ALPHA (news-enhanced predicted dispersion) — 2011-2020 (news window)")
show("constant-gross alpha", alpha); show("dispersion-sized (x1)", sized)
mult2=(1.0+3.0*(mult-1.0)).clip(0.2,2.5); show("dispersion-sized (x3 aggressive)", (mult2.shift(1)*alpha).dropna())
print(f"  mult range: {mult.min():.2f}..{mult.max():.2f} (std {mult.std():.3f})")
print(f"\n  avg gross mult in flat years (2013,2016): {mult[[d.year in (2013,2016) for d in mult.index]].mean():.2f}  vs high-disp years: {mult[[d.year in (2011,2018) for d in mult.index]].mean():.2f}")
