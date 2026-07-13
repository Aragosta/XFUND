#!/usr/bin/env python3
"""NEWS.py — news-sentiment sleeve from FNSPID (free financial-news dataset) + VADER sentiment.

The first NON-PRICE, orthogonal signal in the stack. Score each article title with VADER (fast lexicon; FNSPID's
own paper uses VADER/LM), aggregate to (ticker, month) mean sentiment. SIGNAL = cross-sectional sentiment level
(long positive / short negative), BETANEUT, monthly decile L/S. Point-in-time: month-t news -> signal at end of t
-> earns t->t+1. Net via BACKTEST.py; year-by-year + correlation to the MOM/DM alpha (is it the orthogonal
diversifier value/quality weren't?)."""
import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np, pandas as pd, pickle
from scipy.stats import skew
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import BACKTEST, BETANEUT

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
up = {c.upper(): c for c in px.columns}

# ---- load FNSPID news, VADER-score titles ----
files = sorted(glob.glob("data/fnspid/*.parquet"))
df = pd.concat([pd.read_parquet(f, columns=["Date","Stock_symbol","Article_title"]) for f in files], ignore_index=True)
df = df.dropna(subset=["Date","Stock_symbol","Article_title"])
df["sym"] = df["Stock_symbol"].astype(str).str.upper().map(up)                # map to our ticker casing
df = df.dropna(subset=["sym"])
df["dt"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None); df = df.dropna(subset=["dt"])
print(f"[news] {len(df):,} articles, {df['sym'].nunique()} tickers in our universe, {df['dt'].min().date()}..{df['dt'].max().date()}", flush=True)
sia = SentimentIntensityAnalyzer()
uniq = df["Article_title"].astype(str).unique(); sc = {t: sia.polarity_scores(t)["compound"] for t in uniq}   # score unique titles
print(f"[news] scored {len(uniq):,} unique titles", flush=True)
df["s"] = df["Article_title"].astype(str).map(sc)

# ---- aggregate to (ticker, month-end) mean sentiment ----
df["per"] = df["dt"].dt.to_period("M")
g = df.groupby(["per","sym"])["s"].agg(["mean","count"]).reset_index()
mmap = {d.to_period("M"): d for d in me}
g["d"] = g["per"].map(mmap); g = g.dropna(subset=["d"])
sent = g.pivot_table(index="d", columns="sym", values="mean").reindex(index=me, columns=px.columns)
cnt  = g.pivot_table(index="d", columns="sym", values="count").reindex(index=me, columns=px.columns).fillna(0.0)
sig = sent.where((cnt >= 2) & elig)                                            # need >=2 articles
print(f"[news] coverage {int((sig.notna()).sum(axis=1)[me>='2011-01-01'].mean())} names/mo", flush=True)

Wn = pd.DataFrame(0.0, index=me, columns=px.columns)
for d in me:
    s = sig.loc[d].dropna()
    if len(s) < 30: continue
    n = max(1, int(len(s)*0.20)); Wn.loc[d, s.nlargest(n).index] = 1.0/n; Wn.loc[d, s.nsmallest(n).index] = -1.0/n
Wn = BETANEUT.betaneut(Wn, BETA)
def stream(W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)
    s = pd.Series(r["returns"]); s.index = pd.DatetimeIndex(s.index); return s[s.index >= "2011-01-01"].dropna()
def show(name, x):
    x = x.dropna(); e = (1+x).cumprod()
    print(f"  {name:22} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}  skew {skew(x):>+5.2f}")
    print("    " + " ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2011, 2024)))
nr = stream(Wn)
al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
ar = stream(al)
print("="*90); print("NEWS-SENTIMENT SLEEVE (FNSPID + VADER, beta-neut) — net, 2011+ (FNSPID ends 2023)")
show("NEWS sentiment", nr); show("ALPHA (MOM+DM meta)", ar)
c = pd.DataFrame({"n":nr,"a":ar}).dropna(); c = c[c.index < "2024-01-01"]
print(f"\n  corr(NEWS, ALPHA) = {c.corr().iloc[0,1]:+.2f}  ({len(c)} months, 2011-2023)")
