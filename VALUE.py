#!/usr/bin/env python3
"""VALUE.py — the VALUE sleeve, built from SEC EDGAR point-in-time fundamentals (data/edgar/facts.parquet).

POINT-IN-TIME: at each month-end use only facts with FILED <= month-end (as-first-reported; latest balance / latest
annual earnings known at t). SIGNALS: Book-to-Market = book_equity / (shares*price); Earnings yield = annual
NetIncomeLoss / (shares*price). Composite = cross-sectional z(B/M)+z(E/P) (winsorized; require book>0). Long cheap
/ short expensive -> BETANEUT -> monthly decile L/S. Value is the classic momentum diversifier (-corr). Net via
BACKTEST.py. Reports standalone + year-by-year + correlation to the MOM/DM alpha, and the 3-way ERC."""
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

# ---- POINT-IN-TIME fundamental panels (frames data: end + val; PIT via conservative reporting LAG) ----
f = pd.read_parquet("data/edgar/facts.parquet").dropna(subset=["val"]).copy()
f["end"] = pd.to_datetime(f["end"])
LAG = {"book": 90, "shares": 90, "ni": 120}                                 # days after period-end when data is assumed available
f["avail"] = f["end"] + pd.to_timedelta(f["concept"].map(LAG).fillna(90), unit="D")
def pit(sub):
    """latest value AVAILABLE at each month-end (period-end + reporting lag <= t)."""
    out = pd.DataFrame(np.nan, index=me, columns=px.columns)
    for tk, g in sub.groupby("ticker"):
        if tk not in px.columns: continue
        g = g.sort_values(["avail","end"]).drop_duplicates("avail", keep="last")
        idx = np.searchsorted(g["avail"].values, me.values, side="right") - 1
        v = np.where(idx >= 0, g["val"].values[idx.clip(min=0)], np.nan)
        out[tk] = np.where(idx >= 0, v, np.nan)
    return out
book   = pit(f[f.concept == "book"])
shares = pit(f[f.concept == "shares"])
earn   = pit(f[f.concept == "ni"])                                          # annual net income (10-K FY)
cover  = book.notna() & shares.notna()
print(f"[value] fundamentals coverage: {int((cover & elig).sum(axis=1).mean())} names/mo (of ~{int(elig.sum(axis=1).mean())} eligible)", flush=True)

# ---- value signals ----
mcap = shares * m_px
bm = (book / mcap).where(book > 0)                                          # book-to-market (cheap = high)
ep = (earn / mcap)                                                          # earnings yield (cheap = high)
def zwin(df):
    z = df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0); return z.clip(-3, 3)
val = (zwin(bm).fillna(0) * (bm.notna()) + zwin(ep).fillna(0) * (ep.notna())) / (bm.notna().astype(float) + ep.notna().astype(float)).replace(0, np.nan)
val = val.where(elig & cover)

# ---- decile L/S, beta-neut ----
Wv = pd.DataFrame(0.0, index=me, columns=px.columns)
for d in me:
    s = val.loc[d].dropna()
    if len(s) < 50: continue
    n = max(1, int(len(s) * 0.10)); Wv.loc[d, s.nlargest(n).index] = 1.0/n; Wv.loc[d, s.nsmallest(n).index] = -1.0/n
Wv = BETANEUT.betaneut(Wv, BETA)

def stream(W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)
    s = pd.Series(r["returns"]); s.index = pd.DatetimeIndex(s.index); return s[s.index >= "2011-01-01"].dropna()
spy = pd.read_parquet("/tmp/spy.parquet")["SPY"].reindex(me, method="ffill").pct_change()
def show(name, x):
    x = x.dropna(); e = (1+x).cumprod(); dd = (e/e.cummax()-1).min()
    print(f"  {name:22} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {dd:>6.1%}  skew {skew(x):>+5.2f}")
    print("    " + " ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2011, 2027)))
vr = stream(Wv)
al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
ar = stream(al)
print("="*90); print("VALUE SLEEVE (SEC EDGAR fundamentals, B/M + E/P, beta-neut) — net, 2011+")
show("VALUE (B/M + E/P)", vr); show("ALPHA (MOM+DM meta)", ar)
print(f"\n  corr(VALUE, ALPHA) = {pd.DataFrame({'v':vr,'a':ar}).dropna().corr().iloc[0,1]:+.2f}")
both = pd.DataFrame({"a":ar,"v":vr}).dropna(); wv = (1/both['v'].expanding(24).std())/((1/both['a'].expanding(24).std())+(1/both['v'].expanding(24).std()))
show("ALPHA + VALUE (ERC)", ((1-wv)*both['a']+wv*both['v']).dropna())
