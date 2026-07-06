#!/usr/bin/env python3
"""Statistical anatomy of momentum in OUR data — the analog of Han's bimodal analysis.
  (A) Cross-sectional relative-return distribution conditional on momentum (Han Fig 3): sort into
      momentum deciles, show the PMF over next-month return deciles. Is it bimodal / U-shaped?
  (B) Cross-sectional dispersion by momentum decile (Han: H & L are more dispersed).
  (C) Time-series distribution of the momentum L/S return: skewness, kurtosis, worst months (crashes).
  (D) Conditional crash structure: momentum return by market state (Daniel-Moskowitz).
Clean liquid universe ($vol>5M, px>$5), monthly, delisting-imputed."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
V = m_px.notna().values; last = len(me) - 1
for j in range(V.shape[1]):
    w = np.where(V[:, j])[0]
    if len(w) and w[-1] < last - 1 and w[-1] + 1 < len(me): mret.iat[w[-1] + 1, j] = -0.30
dvm = (px * vb).resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mdv = dvm.reindex(pd.PeriodIndex(me, freq="M")); mdv.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6)
mom = m_px.shift(1) / m_px.shift(12) - 1                      # ~12-1 momentum (skip most recent month)
fwd = mret.shift(-1)                                          # next-month return
oos = me >= "2011-01-01"

def dec(s):                                                   # 0 = highest, 9 = lowest
    r = s.rank(method="first"); return (9 - ((r - 1) * 10 // len(r)).clip(upper=9)).astype(int)

# ── (A) conditional relative-return distribution + (B) dispersion ──
cnt = np.zeros((10, 10)); disp = {c: [] for c in range(10)}; nmo = 0
for k in np.where(oos)[0]:
    e = elig.iloc[k].fillna(False)
    idx = mom.iloc[k].index[mom.iloc[k].notna() & fwd.iloc[k].notna() & e.values]
    if len(idx) < 100: continue
    md = 9 - dec(mom.iloc[k].reindex(idx)); fdi = 9 - dec(fwd.iloc[k].reindex(idx))   # 0=high,9=low
    f = fwd.iloc[k].reindex(idx)
    for c in range(10):
        m = (md.values == c)
        if m.any():
            for r in fdi.values[m]: cnt[c, r] += 1
            disp[c].append(f.values[m].std())
    nmo += 1
PMF = cnt / cnt.sum(1, keepdims=True)

print("=" * 74)
print(f"(A) CROSS-SECTIONAL RELATIVE-RETURN DISTRIBUTION conditional on momentum ({nmo} months)")
print("    rows = momentum decile (H=high mom .. L=low mom); cols = next-month return decile (H..L)")
print("    (Han's bimodality = U-shape: mass at BOTH ends within a momentum decile)\n")
print("mom\\ret   " + "".join(f"{c:>6}" for c in ["H", "2", "3", "4", "5", "6", "7", "8", "9", "L"]))
labs = ["H", "2", "3", "4", "5", "6", "7", "8", "9", "L"]
for c in range(10):
    print(f"  {labs[c]:<7}" + "".join(f"{PMF[c, r]*100:>6.1f}" for r in range(10)))
print("\n(B) cross-sectional dispersion (std of next-month return) by momentum decile:")
print("    " + "  ".join(f"{labs[c]}:{np.mean(disp[c]):.3f}" for c in range(10)))
# bimodality index: (P(H)+P(L)) / P(middle) within extreme momentum deciles
def bim(row): return (row[0] + row[9]) / (row[4] + row[5])
print(f"\n    bimodality index (P_extreme/P_middle):  H-momentum={bim(PMF[0]):.2f}   L-momentum={bim(PMF[9]):.2f}   mid={bim(PMF[4]):.2f}")
print(f"    (>1 => U-shaped/bimodal; Han finds L-momentum most bimodal)")

# ── (C) time-series distribution of the momentum L/S return ──
ls = {}
for k in np.where(oos)[0]:
    e = elig.iloc[k].fillna(False)
    idx = mom.iloc[k].index[mom.iloc[k].notna() & fwd.iloc[k].notna() & e.values]
    if len(idx) < 100: continue
    md = 9 - dec(mom.iloc[k].reindex(idx)); f = fwd.iloc[k].reindex(idx)
    ls[me[k]] = f[md.values == 0].mean() - f[md.values == 9].mean()      # long high-mom, short low-mom
ls = pd.Series(ls).dropna()
uni = pd.Series({me[k]: fwd.iloc[k][elig.iloc[k].fillna(False).values].mean() for k in np.where(oos)[0]}).dropna()
def st(r):
    return (f"mean {r.mean()*12:>6.1%}  vol {r.std()*np.sqrt(12):>5.1%}  Sharpe {(r.mean()*12)/(r.std()*np.sqrt(12)):>5.2f}  "
            f"SKEW {stats.skew(r):>6.2f}  kurt {stats.kurtosis(r):>5.1f}  min {r.min():>6.1%}  %neg {(r<0).mean():>4.0%}")
print("\n" + "=" * 74)
print("(C) TIME-SERIES DISTRIBUTION of the raw 12-1 momentum L/S return (monthly):")
print(f"    momentum L/S : {st(ls)}")
print(f"    universe(EW) : {st(uni)}")
print(f"    worst 5 momentum months: " + ", ".join(f"{d.strftime('%Y-%m')}:{v:.0%}" for d, v in ls.nsmallest(5).items()))

# ── (D) crash structure: momentum return by prior market state ──
mkt12 = uni.rolling(12).sum()                                # trailing 12m market (universe) return
bear = mkt12.reindex(ls.index) < 0
print("\n(D) CRASH STRUCTURE — momentum L/S conditional on market state (Daniel-Moskowitz):")
print(f"    after BEAR (trailing-12m mkt<0): mean {ls[bear.values].mean()*12:>6.1%}  SKEW {stats.skew(ls[bear.values]) if bear.sum()>3 else float('nan'):>5.2f}  min {ls[bear.values].min():>6.1%}  (n={int(bear.sum())})")
print(f"    after BULL (trailing-12m mkt>=0):mean {ls[~bear.values].mean()*12:>6.1%}  SKEW {stats.skew(ls[~bear.values]):>5.2f}  min {ls[~bear.values].min():>6.1%}  (n={int((~bear).sum())})")
print("[done]", flush=True)
