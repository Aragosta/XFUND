#!/usr/bin/env python3
"""dispersion_predict.py — is cross-sectional DISPERSION predictable, and does NEWS improve the forecast?

Dispersion (a 2nd moment, like vol) should be forecastable; our alpha's return is driven by it. Target =
next-month cross-sectional return dispersion. Baseline predictors = HAR dispersion (trailing 1/3/6/12m) + market
vol. Test whether adding NEWS features (news volume = mean log article count; sentiment DISAGREEMENT = cross-
sectional std of monthly sentiment) improves OOS R². If yes -> predicted-dispersion can size the alpha (up when
high-disp coming, down in low-disp junk-rally regimes). Caches VADER sentiment aggregate to avoid re-scoring."""
import warnings; warnings.filterwarnings("ignore")
import os, glob, numpy as np, pandas as pd

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6)
disp = mret.where(elig).std(axis=1)                                          # cross-sectional return dispersion (monthly)
spy = pd.read_parquet("/tmp/spy.parquet")["SPY"].reindex(me, method="ffill")
mvol = spy.pct_change().rolling(21).std().reindex(me, method="ffill")*np.sqrt(21)  # ~market vol proxy

# ---- news monthly aggregate (cached) ----
CACHE = "data/fnspid/sent_monthly.parquet"
if os.path.exists(CACHE):
    g = pd.read_parquet(CACHE)
else:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    up = {c.upper(): c for c in px.columns}
    df = pd.concat([pd.read_parquet(f, columns=["Date","Stock_symbol","Article_title"]) for f in sorted(glob.glob("data/fnspid/*.parquet"))], ignore_index=True).dropna()
    df["sym"] = df["Stock_symbol"].astype(str).str.upper().map(up); df = df.dropna(subset=["sym"])
    df["dt"] = pd.to_datetime(df["Date"], utc=True, errors="coerce").dt.tz_localize(None); df = df.dropna(subset=["dt"])
    sia = SentimentIntensityAnalyzer(); uniq = df["Article_title"].astype(str).unique(); sc = {x: sia.polarity_scores(x)["compound"] for x in uniq}
    df["s"] = df["Article_title"].astype(str).map(sc); df["per"] = df["dt"].dt.to_period("M").astype(str)
    g = df.groupby(["per","sym"])["s"].agg(["mean","count"]).reset_index(); g.to_parquet(CACHE); print("[cache] saved", CACHE, flush=True)
g["d"] = g["per"].map({d.to_period("M").__str__(): d for d in me}); g = g.dropna(subset=["d"])
sent = g.pivot_table(index="d", columns="sym", values="mean").reindex(index=me); cnt = g.pivot_table(index="d", columns="sym", values="count").reindex(index=me)
news_vol = np.log1p(cnt.sum(axis=1))                                         # total news volume (attention)
sent_disagree = sent.std(axis=1)                                            # cross-sectional sentiment DISAGREEMENT
sent_neg = (sent < -0.1).sum(axis=1) / cnt.notna().sum(axis=1)              # fraction negative-sentiment

# ---- feature matrix ----
X = pd.DataFrame({
    "disp1": disp, "disp3": disp.rolling(3).mean(), "disp6": disp.rolling(6).mean(), "disp12": disp.rolling(12).mean(),  # HAR
    "mvol": mvol, "dvol": disp/disp.rolling(6).mean(),
    "news_vol": news_vol, "sent_disagree": sent_disagree, "sent_neg": sent_neg,
}, index=me)
y = disp.shift(-1)                                                          # next-month dispersion
BASE = ["disp1","disp3","disp6","disp12","mvol","dvol"]; NEWSF = ["news_vol","sent_disagree","sent_neg"]
D = X.assign(y=y).dropna(); D = D[(D.index >= "2011-06-01") & (D.index < "2020-07-01")]   # news coverage window
print(f"[disp] {len(D)} months (2011-2020, news window)", flush=True)

def oos_r2(cols):                                                          # expanding-window OLS OOS R^2
    idx = D.index; pred = pd.Series(np.nan, index=idx)
    for i in range(36, len(idx)):
        tr = D.iloc[:i]; A = np.column_stack([np.ones(len(tr))] + [tr[c].values for c in cols])
        b, *_ = np.linalg.lstsq(A, tr["y"].values, rcond=None)
        xr = np.concatenate([[1.0], [D[c].iloc[i] for c in cols]]); pred.iloc[i] = xr @ b
    r = D["y"].reindex(pred.dropna().index); p = pred.dropna()
    return 1 - ((r-p)**2).sum()/((r-r.mean())**2).sum(), r.corr(p)
print("DISPERSION PREDICTABILITY (next-month cross-sectional dispersion, expanding OOS):")
for name, cols in [("persistence only (disp1)", ["disp1"]), ("HAR+vol (baseline)", BASE), ("HAR+vol + NEWS", BASE+NEWSF)]:
    r2, cc = oos_r2(cols); print(f"  {name:26} OOS R2 {r2:>+6.2f}   corr {cc:>+5.2f}", flush=True)
# does news alone correlate with next dispersion?
for nf in NEWSF:
    print(f"  corr({nf:14}, next-disp) = {D[nf].corr(D['y']):+.2f}")
