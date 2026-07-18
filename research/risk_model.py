#!/usr/bin/env python3
"""risk_model.py — MULTI-FACTOR second-moment risk model (the RISK layer). The 2nd moment is forecastable for
EVERY factor (Moreira-Muir), not just momentum. Two independent, complementary pieces:
  (A) MARGINAL vols — each factor's own trailing vol forecasts its risk -> vol-manage EACH factor.
  (B) COUPLING regime — the STATE (absorption/turbulence) forecasts WHEN correlations spike and diversification
      fails (Kritzman). Scale total gross down in high-coupling regimes. This is what per-factor vol CANNOT see.
Test: per-factor vol-forecastability, then static vs vol-managed-each vs +coupling-overlay (SR/DD/per-era).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub

hub = DataHub(start="2000-01-01", min_days=0)
me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid"); T = len(me)
mdv = hub.mdv
mkt = mret.where(elig).mean(axis=1)
mom12 = m_px.shift(1) / m_px.shift(12) - 1; vol6 = mret.rolling(6, min_periods=4).std()
def ls(rankdf, hi_lo=True):
    o = pd.Series(np.nan, index=me)
    for t in range(1, T):
        s = rankdf.loc[me[t-1]].where(elig.loc[me[t-1]]).dropna()
        if len(s) < 100: continue
        q = pd.qcut(s.rank(method="first"), 10, labels=False); r = mret.loc[me[t]]
        hi = r.reindex(s.index[q == 9]).mean(); lo = r.reindex(s.index[q == 0]).mean()
        o.iloc[t] = (hi - lo) if hi_lo else (lo - hi)
    return o
FAC = pd.DataFrame({"mkt": mkt, "mom": ls(mom12, True),
                    "size": ls(np.log(mdv.replace(0, np.nan)), False), "lowvol": ls(vol6, False)}).dropna()

# ── (A) per-factor 2nd-moment forecastability + vol-managed returns ───────────
print("=" * 84)
print("(A) MARGINAL 2nd moment is forecastable for EVERY factor (trailing-vol -> next-mo |return|):")
volm = {}
for c in FAC.columns:
    r = FAC[c]; tv = r.abs().rolling(6, min_periods=4).mean()                    # trailing vol forecast
    ic = pd.DataFrame({"f": tv.shift(1), "y": r.abs()}).dropna()
    ic = ic[ic.index >= "2011-01-01"]; icv = ic["f"].corr(ic["y"], method="spearman")
    tgt = tv.median(); w = (tgt / tv).clip(0.2, 3.0)
    volm[c] = (w.shift(1) * r)
    print(f"   {c:8} vol-forecast IC {icv:>+.3f}   raw SR {r[r.index>='2011'].mean()/r[r.index>='2011'].std()*np.sqrt(12):>5.2f}"
          f"   vol-managed SR {volm[c][volm[c].index>='2011'].mean()/volm[c][volm[c].index>='2011'].std()*np.sqrt(12):>5.2f}")
VM = pd.DataFrame(volm)

# ── (B) coupling regime from STATE (absorption/turbulence) ────────────────────
ST = pd.read_parquet("/tmp/state.parquet"); ST.index = pd.DatetimeIndex(ST.index)
coupling = ST[["absorption", "turbulence"]].reindex(me).apply(lambda s: (s - s.expanding(24).mean()) / (s.expanding(24).std() + 1e-9)).mean(axis=1)
cpct = coupling.expanding(24).apply(lambda a: (a.iloc[-1] > a).mean())          # coupling percentile (leak-free)

def combo(df, overlay=None):
    """equal-vol combine the factors; optional gross overlay (scale down in high-coupling regimes)."""
    iv = 1.0 / df.rolling(12, min_periods=6).std().shift(1)
    w = iv.div(iv.sum(axis=1), axis=0)
    x = (w * df).sum(axis=1)
    if overlay is not None: x = overlay.shift(1).reindex(x.index).fillna(1.0) * x
    return x
def stat(x, lbl):
    x = x[(x.index >= "2011-01-01") & (x.index < "2027-01-01")].dropna(); e = (1 + x).cumprod()
    sr = x.mean() / x.std() * np.sqrt(12); dd = (e / e.cummax() - 1).min()
    def sub(a, b):
        xx = x[(x.index >= a) & (x.index < b)]; return xx.mean()/xx.std()*np.sqrt(12) if xx.std() > 0 else np.nan
    print(f"   {lbl:26} SR {sr:>5.2f}  maxDD {dd:>7.1%}   11-15 {sub('2011','2016'):>5.2f}  16-18 {sub('2016','2019'):>5.2f}"
          f"  19-21 {sub('2019','2022'):>5.2f}  22-26 {sub('2022','2027'):>5.2f}")

ov = (1.0 - 0.6 * cpct).clip(0.4, 1.0)                                          # high coupling -> cut gross to 0.4
print("\n(B) MULTI-FACTOR construction (equal-vol combine), 2011-26:")
stat(combo(FAC), "static factors")
stat(combo(VM), "vol-managed each")
stat(combo(VM, ov), "vol-managed + STATE coupling")
print("\nRead: (A) if every factor's vol-forecast IC>0 & vol-managed SR>raw, the 2nd moment is a general lever.")
print("(B) if the STATE coupling overlay cuts DD / lifts stress-era SR beyond per-factor vol, the state adds the")
print("COVARIANCE information (correlation spikes) that marginal vols miss — its correct job in the risk layer.")
