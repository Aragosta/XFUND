#!/usr/bin/env python3
"""combine_erc.py — ERC (risk-parity) combination of the 3 CONFIRMED-real sleeves on the NEW honest data:
MOM (/tmp/mom_new.pkl) + DM (/tmp/dm_returns.pkl) + MR (/tmp/mr_new.pkl). Leak-free expanding-window ERC.
Reports combined net SR/Sortino/maxDD, per-sleeve, pairwise corr, and the diversification lift over the best single."""
import warnings, os, sys, pickle
warnings.filterwarnings("ignore"); sys.path.insert(0, "/Users/enzokreeft/XFUND"); os.chdir("/Users/enzokreeft/XFUND")
import numpy as np, pandas as pd
import ERC
from DATAHUB import DataHub

def S(p, k=None):
    o = pickle.load(open(p, "rb")); s = pd.Series(dict(o) if isinstance(o, list) else (o[k] if k else o)).dropna()
    s.index = pd.DatetimeIndex(s.index); return s[~s.index.duplicated()].sort_index()

MOM = S("/tmp/mom_new.pkl"); DM = S("/tmp/dm_returns.pkl"); MR = S("/tmp/mr_new.pkl")
R = pd.DataFrame({"MOM": MOM, "DM": DM, "MR": MR}).dropna()
R = R[(R.index >= "2011-01-01")]
print(f"aligned sleeve returns: {len(R)} months  {R.index.min():%Y-%m}..{R.index.max():%Y-%m}\n", flush=True)

def stats(x, lab):
    x = x.dropna(); e = (1 + x).cumprod(); dd = (e / e.cummax() - 1).min()
    sr = x.mean() / x.std() * np.sqrt(12); dn = x[x < 0].std() * np.sqrt(12)
    sor = x.mean() * 12 / dn if dn > 0 else np.nan
    print(f"  {lab:20} SR {sr:>5.2f}  Sortino {sor:>4.1f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  "
          f"vol {x.std()*np.sqrt(12):>5.1%}  maxDD {dd:>6.1%}", flush=True)
    return sr

print("PAIRWISE CORRELATIONS (the diversification that ERC monetizes):")
print(R.corr().round(2).to_string(), flush=True)

print("\nPER-SLEEVE (net, 2011+):")
srs = {c: stats(R[c], c) for c in R.columns}

# leak-free expanding ERC (risk-parity), min 24m history
W = ERC.expanding_alloc(R, method="erc", win=24, shrink=0.1)
erc_ret = (R * W).sum(1)[W.iloc[:, 0].notna()].loc["2013-01-01":]     # start after warmup
iv = ERC.expanding_alloc(R, method="invvol", win=24)
iv_ret = (R * iv).sum(1).loc["2013-01-01":]
ew_ret = R.mean(1).loc["2013-01-01":]

print("\nCOMBINED BOOKS (2013+, leak-free weights):")
best_single = max(srs, key=srs.get)
stats(R[best_single].loc["2013-01-01":], f"best single ({best_single})")
stats(ew_ret, "equal-weight")
stats(iv_ret, "inverse-vol")
erc_sr = stats(erc_ret, "ERC (risk-parity)")

print(f"\n  avg ERC weights: {dict(zip(R.columns, W.dropna().mean().round(2)))}", flush=True)
print(f"  DIVERSIFICATION LIFT: ERC SR {erc_sr:.2f} vs best single {srs[best_single]:.2f} "
      f"(+{erc_sr - srs[best_single]:.2f})", flush=True)
pickle.dump(erc_ret.to_dict(), open("/tmp/erc_combined.pkl", "wb"))
