#!/usr/bin/env python3
"""vol_forecast.py — is the 2nd moment REALLY unforecastable for market/size, or was the test broken? Redo it
properly: realized variance from DAILY returns (low-noise target) + HAR features (realized vol at 1/3/12-month
horizons, Corsi). Compare the NAIVE monthly-|return| setup (what risk_model.py used) vs the HAR realized-variance
setup, per factor. Expect: HAR forecasts realized variance with high IC/R^2 for EVERY factor incl. market."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub

hub = DataHub(start="2000-01-01", min_days=0)
me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid")
ret_d, days = hub.ret_d, hub.days; T = len(me)
mom12 = m_px.shift(1) / m_px.shift(12) - 1; vol6 = mret.rolling(6, min_periods=4).std()
elig_d = elig.reindex(days, method="ffill")

# ── daily factor return series ────────────────────────────────────────────────
mktd = ret_d.where(elig_d).mean(axis=1)                                         # daily market (eq-weight eligible)
def ls_daily(rankdf, hi_lo=True):
    o = pd.Series(np.nan, index=days)
    for t in range(1, T):
        s = rankdf.loc[me[t-1]].where(elig.loc[me[t-1]]).dropna()
        if len(s) < 100: continue
        q = pd.qcut(s.rank(method="first"), 10, labels=False)
        lo_i, hi_i = s.index[q == 0], s.index[q == 9]
        d0, d1 = me[t-1], me[t]; dd = days[(days > d0) & (days <= d1)]
        r = ret_d.loc[dd, hi_i].mean(axis=1) - ret_d.loc[dd, lo_i].mean(axis=1)
        o.loc[dd] = r if hi_lo else -r
    return o
series = {"mkt": mktd, "mom": ls_daily(mom12, True),
          "size": ls_daily(np.log(hub.mdv.replace(0, np.nan)), False), "lowvol": ls_daily(vol6, False)}

def monthly_rv(sd):                                                            # realized variance per month from daily
    rv = sd.pow(2).groupby(sd.index.to_period("M")).sum()
    rv.index = rv.index.to_timestamp("M"); return rv.reindex(pd.PeriodIndex(me, freq="M").to_timestamp("M")).set_axis(me)

print("=" * 88)
print("2nd-moment forecastability: NAIVE monthly-|ret| vs HAR realized-variance (IC to next-month, 2011-26)")
print(f"  {'factor':8}{'NAIVE IC':>10}{'HAR IC':>9}{'HAR R2':>9}")
for c, sd in series.items():
    rv = monthly_rv(sd).replace(0, np.nan)
    lrv = np.log(rv)
    # HAR features (all known at t): log-RV 1m, 3m avg, 12m avg
    F = pd.DataFrame({"h1": lrv, "h3": lrv.rolling(3, min_periods=2).mean(), "h12": lrv.rolling(12, min_periods=6).mean()})
    y = lrv.shift(-1)
    D = pd.concat([F, y.rename("y")], axis=1).dropna(); pred = pd.Series(np.nan, index=me)
    for i in range(36, len(D)):
        tr = D.iloc[:i]; A = np.column_stack([np.ones(len(tr))] + [tr[k].values for k in ["h1","h3","h12"]])
        b = np.linalg.lstsq(A, tr["y"].values, rcond=None)[0]
        pred.loc[D.index[i]] = np.concatenate([[1.0], D[["h1","h3","h12"]].iloc[i].values]) @ b
    # HAR IC/R2 vs realized next-month RV
    ev = pd.DataFrame({"p": pred, "y": rv.shift(-1)}).dropna(); ev = ev[ev.index >= "2011-01-01"]
    har_ic = ev["p"].corr(np.log(ev["y"]), method="spearman")
    r2 = 1 - ((np.log(ev["y"]) - ev["p"])**2).sum() / ((np.log(ev["y"]) - np.log(ev["y"]).mean())**2).sum()
    # naive: trailing monthly |ret| -> next-month |ret|
    fmret = mret_f = pd.Series({d: sd[sd.index.to_period("M") == d.to_period("M")].sum() for d in me})  # monthly factor ret
    tv = fmret.abs().rolling(6, min_periods=4).mean().shift(1); nv = pd.DataFrame({"f": tv, "y": fmret.abs()}).dropna()
    nv = nv[nv.index >= "2011-01-01"]; naive_ic = nv["f"].corr(nv["y"], method="spearman")
    print(f"  {c:8}{naive_ic:>+10.3f}{har_ic:>+9.3f}{r2:>9.2f}")
print("\nRead: if HAR IC/R2 >> NAIVE for market/size too, the 2nd moment IS forecastable everywhere — the earlier")
print("~0 was a broken target (single monthly |ret|) + no daily data, not genuine unpredictability.")
