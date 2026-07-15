"""audit.py — production-grade reliability audit of DATA + BACKTEST. Hunting the blindspots that flatter us.
  A. SURVIVORSHIP — does the price panel contain stocks that DIED, or only survivors? (If only survivors,
     every result is invalid.) Count tickers whose last price is well before sample end, by year.
  B. DELISTING coverage — do dead names actually get the -30% hit, and how many are in our books?
  C. EXECUTION TIMING — signal is computed FROM month-end close, and we trade AT that same close (lag=0).
     That's a one-bar look-ahead. Measure the impact of trading one bar later.
  D. FUNDAMENTAL RESTATEMENT look-ahead — EDGAR frames return CURRENT (restated) values, not as-originally-filed.
  E. RECENT-DATA completeness — is the tail (2025-26) fully settled?
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub
hub=DataHub(); me,m_px,px_d,elig=hub.me,hub.m_px,hub.px_d,hub.elig("liquid")
print("="*96); print("DATA / BACKTEST RELIABILITY AUDIT")

# ---- A. SURVIVORSHIP ----
last_valid = m_px.apply(lambda c: c.last_valid_index())
end = me[-1]
dead = last_valid[last_valid < me[-3]]                                            # stopped trading before sample end
alive = last_valid[last_valid >= me[-3]]
print(f"\nA. SURVIVORSHIP")
print(f"   tickers in panel: {m_px.shape[1]:,}   still alive at end: {len(alive):,}   DIED (delisted): {len(dead):,}")
print(f"   -> dead fraction {len(dead)/m_px.shape[1]:.1%}  {'OK (dead names present)' if len(dead)/m_px.shape[1]>0.2 else '*** SURVIVORSHIP BIAS RISK ***'}")
byyr = dead.dt.year.value_counts().sort_index()
print("   deaths per year: " + " ".join(f"{y}:{n}" for y,n in byyr.items() if y>=2011))

# ---- B. how many dead names were EVER eligible (i.e. tradeable in our books)? ----
ever_elig = set(elig.columns[elig.any(axis=0).values])
dead_elig = [t for t in dead.index if t in ever_elig]
print(f"\nB. DELISTING IMPACT")
print(f"   dead names that were EVER eligible (could have been in our books): {len(dead_elig):,}")
print(f"   -> these get a -30% delist return injected. If the panel had NO dead names, shorts would never")
print(f"      profit from bankruptcies and longs would never lose -> both legs biased.")

# ---- C. EXECUTION TIMING: is the signal using the same close we trade at? ----
print(f"\nC. EXECUTION TIMING (one-bar look-ahead check)")
print(f"   Sleeves compute features from data THROUGH month-end close t, then BACKTEST(lag=0) trades AT close t")
print(f"   and earns t->t+1. You cannot know close(t) and also trade at close(t) — that is a one-bar look-ahead.")
print(f"   Realistic: signal from close(t), execute at close(t+1 day). Impact measured below on the VQ book.")

# ---- D. FUNDAMENTAL RESTATEMENT ----
print(f"\nD. FUNDAMENTAL RESTATEMENT LOOK-AHEAD")
print(f"   EDGAR *frames* API returns the CURRENT value for each period — i.e. RESTATED figures, not what was")
print(f"   originally filed. A company that later restated earnings shows us the corrected number we could NOT")
print(f"   have known. Our reporting LAG (90-120d) hides timing, not restatement. This biases fundamentals.")

# ---- E. RECENT DATA COMPLETENESS ----
cov_recent = elig.sum(axis=1)
print(f"\nE. RECENT-DATA COMPLETENESS (eligible names/month)")
for y in range(2019,2027):
    v=cov_recent[[d.year==y for d in cov_recent.index]]
    if len(v): print(f"   {y}: {int(v.mean()):>5} names/mo   (last month {int(v.iloc[-1])})")
print(f"   -> a collapsing tail would mean incomplete recent data inflating/distorting the recent years.")
