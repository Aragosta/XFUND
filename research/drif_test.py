#!/usr/bin/env python3
"""DRIF standalone validation (Cakici et al. 2026) on our liquid daily universe.

Maps the trailing 21 DAILY returns -> next-month cross-sectional return via elastic-net,
re-fit annually inception-to-date. Gate: is the DAILY signal a real alpha, or — like our
crude MONTHLY short-term reversal in the PCA test — just a carry-bleeding hedge?
Benchmarks: monthly STR (reverse last month's total return) and 12-1 momentum, same universe."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler

DAILY = "/Users/enzokreeft/XFUND/tiingo_daily_checkpoint.parquet"
NDAY, Q, TC = 21, 0.10, 0.0010   # 21 trading days, top/bottom decile, 10bp one-way cost

daily = pd.read_parquet(DAILY); daily.index = pd.to_datetime(daily.index)
daily = daily.sort_index()
dret = daily.pct_change().clip(-0.5, 0.5)
mclose = daily.resample("ME").last()
mret = mclose.pct_change()
print(f"[drif] daily {daily.shape}  monthly {mret.shape}  {mret.index[0].date()}–{mret.index[-1].date()}", flush=True)

# ── assemble (formation_date, ticker) samples: 21 daily returns -> next-month return ──
dates = list(mret.index)
feats, meta = [], []
for i in range(12, len(dates) - 1):              # need 12m history for the momentum benchmark
    d = dates[i]
    win = dret.loc[:d].iloc[-NDAY:]
    if len(win) < NDAY:
        continue
    fwd = mret.iloc[i + 1]                        # next-month return (held month i+1)
    mom = (1 + mret.iloc[i - 11:i]).prod() - 1    # 12-1 momentum (months i-11..i-1, skip i)
    rev = -mret.iloc[i]                           # monthly short-term reversal score
    for tk in daily.columns:
        col = win[tk].values
        if np.isnan(col).any() or pd.isna(fwd.get(tk, np.nan)):
            continue
        feats.append(col)
        meta.append((d, tk, fwd[tk], mom.get(tk, np.nan), rev[tk]))

X = np.asarray(feats)
S = pd.DataFrame(meta, columns=["date", "tk", "fwd", "mom", "rev"])
S["yr"] = S["date"].dt.year
S["fwd_dm"] = S.groupby("date")["fwd"].transform(lambda x: x - x.mean())   # cross-sectional demean
print(f"[drif] {len(S):,} stock-month samples, {X.shape[1]} daily features", flush=True)

# ── walk-forward annual elastic-net (inception-to-date training) ──
preds = np.full(len(S), np.nan)
for Y in sorted(S["yr"].unique()):
    tr = (S["yr"] < Y).values
    te = (S["yr"] == Y).values
    if tr.sum() < 3000:
        continue
    sc = StandardScaler().fit(X[tr])
    m = ElasticNetCV(l1_ratio=0.5, cv=3, n_jobs=-1, max_iter=4000).fit(sc.transform(X[tr]), S["fwd_dm"].values[tr])
    preds[te] = m.predict(sc.transform(X[te]))
    print(f"  [{Y}] train={tr.sum():,}  alpha={m.alpha_:.2e}  nz={np.sum(m.coef_!=0)}/{X.shape[1]}", flush=True)
S["drif"] = preds

# ── form L/S portfolios per formation date for each signal ──
def ls_series(score_col):
    rows = {}
    prev_l, prev_s = set(), set()
    turn = {}
    for d, g in S.dropna(subset=[score_col]).groupby("date"):
        if len(g) < 30:
            continue
        n = max(1, int(len(g) * Q))
        s = g.set_index("tk")[score_col]
        L = set(s.nlargest(n).index); Sh = set(s.nsmallest(n).index)
        fwd = g.set_index("tk")["fwd"]
        rows[d] = fwd[list(L)].mean() - fwd[list(Sh)].mean()
        turn[d] = (len(L ^ prev_l) + len(Sh ^ prev_s)) / (2 * n)
        prev_l, prev_s = L, Sh
    r = pd.Series(rows).sort_index()
    t = pd.Series(turn).reindex(r.index)
    return r, t

def perf(r, t=None, cost=0.0):
    r = r.dropna()
    if t is not None:
        r = r - cost * t.reindex(r.index).fillna(0) * 2     # round-trip cost
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12); dn = r[r < 0].std() * np.sqrt(12)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol, (r.mean() * 12) / dn, dd

drif_r, drif_t = ls_series("drif")
str_r,  str_t  = ls_series("rev")
mom_r,  mom_t  = ls_series("mom")

print("\n" + "=" * 88 + f"\n=== DRIF STANDALONE (top liquid {daily.shape[1]} names, q={Q}) ===")
print(f"{'signal':24}{'ann':>8}{'sharpe':>8}{'sortino':>8}{'maxDD':>9}{'turn/mo':>9}")
for nm, (r, t) in [("DRIF (daily, elastic-net)", (drif_r, drif_t)),
                   ("Monthly STR (reversal)", (str_r, str_t)),
                   ("12-1 Momentum", (mom_r, mom_t))]:
    a, sh, so, dd = perf(r)
    print(f"{nm:24}{a:>8.1%}{sh:>8.2f}{so:>8.2f}{dd:>9.1%}{t.mean():>9.1%}  [GROSS]")
    a, sh, so, dd = perf(r, t, TC)
    print(f"{'  └ net '+str(int(TC*1e4))+'bp':24}{a:>8.1%}{sh:>8.2f}{so:>8.2f}{dd:>9.1%}", flush=True)

# correlation of DRIF to the monthly reversal & momentum
allr = pd.DataFrame({"DRIF": drif_r, "STR": str_r, "MOM": mom_r}).dropna()
print("\n=== DRIF return correlation ===")
print(allr.corr()["DRIF"].round(3).to_string())
print("\n[done]", flush=True)
