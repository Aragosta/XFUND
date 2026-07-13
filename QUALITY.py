#!/usr/bin/env python3
"""QUALITY.py — the QUALITY/PROFITABILITY sleeve (Novy-Marx), from SEC EDGAR point-in-time fundamentals.

The factor that WORKED where value didn't, and is DEFENSIVE (hedges the momentum crash / flights to quality).
SIGNALS: Gross Profitability GP/A = (Revenue - COGS)/Assets (Novy-Marx); ROE = NetIncome/BookEquity;
ROA = NetIncome/Assets. Composite = cross-sectional z-avg (winsorized). Long high-quality / short junk ->
BETANEUT -> monthly decile L/S. Point-in-time via reporting lag. Net via BACKTEST.py; year-by-year + corr to
the MOM/DM alpha + 3-way ERC."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST, BETANEUT

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)

f = pd.read_parquet("data/edgar/facts.parquet").dropna(subset=["val"]).copy(); f["end"] = pd.to_datetime(f["end"])
LAG = {"book":90, "shares":90, "assets":90, "ni":120, "rev":120, "cogs":120}
f["avail"] = f["end"] + pd.to_timedelta(f["concept"].map(LAG).fillna(90), unit="D")
def pit(concept):
    out = pd.DataFrame(np.nan, index=me, columns=px.columns)
    for tk, g in f[f.concept == concept].groupby("ticker"):
        if tk not in px.columns: continue
        g = g.sort_values(["avail","end"]).drop_duplicates("avail", keep="last")
        idx = np.searchsorted(g["avail"].values, me.values, side="right") - 1
        out[tk] = np.where(idx >= 0, g["val"].values[idx.clip(min=0)], np.nan)
    return out
book, assets, ni, rev, cogs = pit("book"), pit("assets"), pit("ni"), pit("rev"), pit("cogs")

gpa = ((rev - cogs) / assets).where((assets > 0) & rev.notna() & cogs.notna())   # Novy-Marx gross profitability
roe = (ni / book).where(book > 0)
roa = (ni / assets).where(assets > 0)
def zwin(df): z = df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
parts = [gpa, roe, roa]; num = sum(zwin(p).fillna(0)*p.notna() for p in parts); den = sum(p.notna().astype(float) for p in parts).replace(0, np.nan)
qual = (num/den).where(elig)
cover = (gpa.notna() | roe.notna())
print(f"[quality] coverage {int((cover & elig).sum(axis=1).mean())} names/mo (GP/A {int((gpa.notna()&elig).sum(1).mean())}, ROE {int((roe.notna()&elig).sum(1).mean())})", flush=True)

Wq = pd.DataFrame(0.0, index=me, columns=px.columns)
for d in me:
    s = qual.loc[d].dropna()
    if len(s) < 50: continue
    n = max(1, int(len(s)*0.10)); Wq.loc[d, s.nlargest(n).index] = 1.0/n; Wq.loc[d, s.nsmallest(n).index] = -1.0/n
Wq = BETANEUT.betaneut(Wq, BETA)
def stream(W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)
    s = pd.Series(r["returns"]); s.index = pd.DatetimeIndex(s.index); return s[s.index >= "2011-01-01"].dropna()
def show(name, x):
    x = x.dropna(); e = (1+x).cumprod()
    print(f"  {name:22} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}  skew {skew(x):>+5.2f}")
    print("    " + " ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2011, 2027)))
qr = stream(Wq)
al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
ar = stream(al)
print("="*90); print("QUALITY SLEEVE (Novy-Marx GP/A + ROE + ROA, beta-neut) — net, 2011+")
show("QUALITY", qr); show("ALPHA (MOM+DM meta)", ar)
print(f"\n  corr(QUALITY, ALPHA) = {pd.DataFrame({'q':qr,'a':ar}).dropna().corr().iloc[0,1]:+.2f}")
both = pd.DataFrame({"a":ar,"q":qr}).dropna(); wq = (1/both['q'].expanding(24).std())/((1/both['a'].expanding(24).std())+(1/both['q'].expanding(24).std()))
show("ALPHA + QUALITY (ERC)", ((1-wq)*both['a']+wq*both['q']).dropna())
