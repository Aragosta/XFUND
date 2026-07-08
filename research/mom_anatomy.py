#!/usr/bin/env python3
"""Anatomy of the MOM sleeve: WHY does the multi-output regressor underperform / drag DM?
Rebuilds the validated base sleeve (Poh+FFD+vol regressor, top-1000, the 0.98 config), and for
EVERY selected name each month stores the predicted z per horizon AND the realized t+1/t+2/t+3.
Then dissects:
  A. Distribution shape of realized fwd returns per horizon, long-leg vs short-leg (Han-style:
     mean/std/skew/kurt/frac<0/bimodality-coeff). Which leg and which horizon is the drag?
  B. Horizon coherence: do the picks reshuffle across horizons? rank-corr of PREDICTED across
     horizons; overlap (Jaccard) of top-decile-by-pred-t1 vs by-pred-t3; same for REALIZED.
  C. Per-horizon IC: corr(pred_h, realized_h) and rank-IC. Is t+1 the only predictable horizon?
  D. Which horizon carries the tradeable edge: LS spread of the MEAN-scored book measured at each
     horizon (edge decay/growth), and per-horizon-SCORED books earning the actual t+1 hold.
  E. Long/short leg attribution: cumulative + Sharpe of each leg separately (momentum shorts?).
  F. Time: rolling 12m Sharpe; corr-to-market; worst months.
top-1000, expanding+quarterly, N_SEEDS=1, gross."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from xgboost import XGBRegressor
import deep_momentum_xgb as d

N_SEEDS, TOPN, MINPX, MINDV, DELIST = 1, 1000, 5.0, 5e6, -0.30
REG = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
V = m_px.notna().values; last = len(me) - 1
for j in range(V.shape[1]):
    w = np.where(V[:, j])[0]
    if len(w) and w[-1] < last - 1 and w[-1] + 1 < len(me): mret.iat[w[-1] + 1, j] = DELIST
dret = px.pct_change(fill_method=None); vol_d = dret.ewm(span=63, min_periods=20).std(); T = len(me)
dvd = px * vb; dvm = dvd.resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mdv = dvm.reindex(pd.PeriodIndex(me, freq="M")); mdv.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > MINPX) & (cov > 0.9) & (mdv > MINDV) & (mdv.rank(axis=1, ascending=False) <= TOPN)
mkt = mret.mean(axis=1)                                                    # EW market
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)
def zrow(df): return df.sub(df.mean(1), 0).div(df.std(1) + 1e-9, 0)

print("[feat] Poh+FFD+volume ...", flush=True)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1).where(lambda z: z.abs() < 5); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, Lg) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(Lg)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp)
ffd = d._ffd_from_training_window(m_px, int(np.argmax(me.year >= 2011)))
for m, v in ffd.items():
    z_ = v.reindex(me); pf[f"ffd{m}"] = z_.sub(z_.mean(1), 0).div(z_.std(1) + 1e-9, 0)
COREREQ = list(pf.keys())
dv63 = dvd.rolling(63, min_periods=40).mean(); dv252 = dvd.rolling(252, min_periods=150).mean()
pf["amihud"]  = zrow(at_me((dret.abs() / dvd.replace(0, np.nan)).rolling(252, min_periods=120).mean()))
pf["dvtrend"] = zrow(at_me(np.log(dv63 / dv252)))
pf["abnvol"]  = zrow(at_me(dvd.rolling(21, min_periods=15).mean() / dv252))
PF = list(pf.keys())

print("[pool] ...", flush=True)
pool = {}
for k in range(13, T - 3):
    dt = me[k]; el = elig.loc[dt].fillna(False); idx0 = el.index[el.values]
    P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in PF})
    ok = P[COREREQ].notna().all(axis=1) & mret.iloc[k+1].reindex(idx0).notna() & mret.iloc[k+2].reindex(idx0).notna() & mret.iloc[k+3].reindex(idx0).notna()
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    R = np.column_stack([mret.iloc[k+h].reindex(idx).values for h in (1, 2, 3)])   # REALIZED fwd
    pool[k] = dict(P=P.loc[idx].fillna(0.0), Y=Y, R=R, dt=dt, vol=vol_d.loc[dt].reindex(idx))
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)

print("[walk] storing per-month preds + realized ...", flush=True)
S = []                                                                     # per test-month records
ms = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month in (1, 4, 7, 10) or ms is None:
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt]
        if len(tr) >= 36:
            X = pd.concat([pool[t]["P"] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
            ms = [XGBRegressor(**REG, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
    if ms is None: continue
    Pmat = np.mean([m.predict(pool[k]["P"].values) for m in ms], axis=0)   # (n,3) predicted z per horizon
    S.append(dict(dt=dt, pred=Pmat, R=pool[k]["R"], vol=pool[k]["vol"].values))
print(f"[walk] {len(S)} test months\n", flush=True)

# ---------- helpers ----------
def bimod(x):                                                              # Sarle's bimodality coeff (>0.555 => bimodal)
    x = x[np.isfinite(x)]
    if len(x) < 8: return np.nan
    g = stats.skew(x); k = stats.kurtosis(x, fisher=True); return (g*g + 1) / (k + 3.0)
def desc(x):
    x = x[np.isfinite(x)]
    return dict(mean=x.mean(), std=x.std(), skew=stats.skew(x), kurt=stats.kurtosis(x),
                fneg=(x < 0).mean(), bc=bimod(x), n=len(x))

# select top/bottom decile by MEAN pred each month; gather realized per horizon
long_R = [[], [], []]; short_R = [[], [], []]
for s in S:
    sc = s["pred"].mean(1); n = max(1, int(len(sc) * 0.10))
    lo = np.argsort(sc)[-n:]; sh = np.argsort(sc)[:n]
    for h in range(3):
        long_R[h].append(s["R"][lo, h]); short_R[h].append(s["R"][sh, h])
long_R = [np.concatenate(v) for v in long_R]; short_R = [np.concatenate(v) for v in short_R]

print("=" * 78)
print("A. REALIZED forward-return distribution of SELECTED names (pooled, per horizon)")
print(f"{'leg / horizon':16}{'mean':>8}{'std':>8}{'skew':>7}{'kurt':>7}{'frac<0':>8}{'bimod':>7}")
for h in range(3):
    a = desc(long_R[h]);  print(f"{'LONG  t+'+str(h+1):16}{a['mean']:>8.2%}{a['std']:>8.1%}{a['skew']:>7.2f}{a['kurt']:>7.1f}{a['fneg']:>8.1%}{a['bc']:>7.2f}")
for h in range(3):
    a = desc(short_R[h]); print(f"{'SHORT t+'+str(h+1):16}{a['mean']:>8.2%}{a['std']:>8.1%}{a['skew']:>7.2f}{a['kurt']:>7.1f}{a['fneg']:>8.1%}{a['bc']:>7.2f}")
print("  (LS spread per horizon = mean LONG - mean SHORT)")
for h in range(3):
    print(f"    t+{h+1}: {np.nanmean(long_R[h]) - np.nanmean(short_R[h]):+.2%}")

print("\n" + "=" * 78)
print("B. HORIZON COHERENCE — do the picks reshuffle across horizons?")
rc_pred = np.zeros((3, 3)); jac = np.zeros((3, 3)); rc_real = np.zeros((3, 3)); nvalid = 0
for s in S:
    P = s["pred"]; R = s["R"]; n = max(1, int(len(P) * 0.10))
    if not np.all(np.isfinite(R)):
        Rz = np.where(np.isfinite(R), R, np.nan)
    else: Rz = R
    for a in range(3):
        for b in range(3):
            rc_pred[a, b] += stats.spearmanr(P[:, a], P[:, b]).correlation
            ta = set(np.argsort(P[:, a])[-n:]); tb = set(np.argsort(P[:, b])[-n:])
            jac[a, b] += len(ta & tb) / len(ta | tb)
            m = np.isfinite(R[:, a]) & np.isfinite(R[:, b])
            rc_real[a, b] += stats.spearmanr(R[m, a], R[m, b]).correlation
    nvalid += 1
rc_pred /= nvalid; jac /= nvalid; rc_real /= nvalid
print("  rank-corr of PREDICTED z across horizons (how similar are the 3 targets the model learns):")
print("        t+1   t+2   t+3");  [print(f"   t+{a+1} " + " ".join(f"{rc_pred[a,b]:5.2f}" for b in range(3))) for a in range(3)]
print("  Jaccard overlap of top-decile picks selected by pred_a vs pred_b:")
print("        t+1   t+2   t+3");  [print(f"   t+{a+1} " + " ".join(f"{jac[a,b]:5.2f}" for b in range(3))) for a in range(3)]
print("  rank-corr of REALIZED returns across horizons (do the same names keep winning):")
print("        t+1   t+2   t+3");  [print(f"   t+{a+1} " + " ".join(f"{rc_real[a,b]:5.2f}" for b in range(3))) for a in range(3)]

print("\n" + "=" * 78)
print("C. PER-HORIZON IC — corr(pred_h, realized_h), cross-sectional monthly avg")
print(f"{'':10}{'pearson-IC':>12}{'rank-IC':>10}{'hitrate':>9}")
for h in range(3):
    ics, ric, hit = [], [], []
    for s in S:
        p = s["pred"][:, h]; r = s["R"][:, h]; m = np.isfinite(r)
        if m.sum() < 20: continue
        ics.append(np.corrcoef(p[m], r[m])[0, 1]); ric.append(stats.spearmanr(p[m], r[m]).correlation)
        hit.append(np.mean(np.sign(p[m] - np.median(p[m])) == np.sign(r[m] - np.median(r[m]))))
    print(f"  pred_t+{h+1} {np.mean(ics):>10.3f}{np.mean(ric):>10.3f}{np.mean(hit):>9.1%}")
# cross-IC: does the MEAN score predict each horizon
print("  cross-IC of MEAN score vs each realized horizon:")
for h in range(3):
    ics = []
    for s in S:
        p = s["pred"].mean(1); r = s["R"][:, h]; m = np.isfinite(r)
        if m.sum() < 20: continue
        ics.append(np.corrcoef(p[m], r[m])[0, 1])
    print(f"    mean -> t+{h+1}: {np.mean(ics):>7.3f}")

print("\n" + "=" * 78)
print("D. WHICH SCORE trades best — LS book (earns t+1 hold), by scoring rule")
def book(scorer):
    rows = []
    for s in S:
        sc = scorer(s); n = max(1, int(len(sc) * 0.10))
        iv = np.clip(1 / (s["vol"] * np.sqrt(21)), None, 50)
        lo = np.argsort(sc)[-n:]; sh = np.argsort(sc)[:n]
        wl = iv[lo] / iv[lo].sum(); ws = iv[sh] / iv[sh].sum()
        r1 = s["R"][:, 0]                                                  # actual 1-month hold
        rl = np.nansum(wl * r1[lo]); rs = np.nansum(ws * r1[sh])
        rows.append((s["dt"], rl - rs, rl, rs))
    df = pd.DataFrame(rows, columns=["dt", "ls", "lg", "sh"]).set_index("dt")
    return df
def sr(r): return (r.mean() * 12) / (r.std() * np.sqrt(12) + 1e-12)
def mdd(r): eq = (1 + r).cumprod(); return (eq / eq.cummax() - 1).min()
for nm, fn in [("mean (current)", lambda s: s["pred"].mean(1)),
               ("t+1 only", lambda s: s["pred"][:, 0]),
               ("t+2 only", lambda s: s["pred"][:, 1]),
               ("t+3 only", lambda s: s["pred"][:, 2])]:
    df = book(fn); r = df["ls"]
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    print(f"  {nm:16} ann {ann:>6.1%}  SR {sr(r):>5.2f}  MDD {mdd(r):>6.1%}  | longSR {sr(df['lg']):>5.2f}  shortSR {sr(-df['sh']):>5.2f}")

print("\n" + "=" * 78)
print("E. LONG vs SHORT LEG attribution (mean-scored book)")
df = book(lambda s: s["pred"].mean(1))
lg, sh = df["lg"], df["sh"]
print(f"  LONG  leg : ann {(1+lg).prod()**(12/len(lg))-1:>6.1%}  SR {sr(lg):>5.2f}  MDD {mdd(lg):>6.1%}")
print(f"  SHORT leg : ann {(1+(-sh)).prod()**(12/len(sh))-1:>6.1%}  SR {sr(-sh):>5.2f}  MDD {mdd(-sh):>6.1%}   (return of the short book)")
print(f"  corr(long, short-book): {lg.corr(-sh):>5.2f}   corr(LS, market): {df['ls'].corr(mkt.reindex(df.index)):>5.2f}")
print(f"  LONG-ONLY (long leg minus EW market): SR {sr(lg - mkt.reindex(lg.index)):>5.2f}")

print("\n" + "=" * 78)
print("F. TIME — rolling 12m Sharpe of mean-scored LS book, worst stretch")
r = df["ls"]; roll = r.rolling(12).apply(lambda x: sr(x))
by_yr = r.groupby(r.index.year).apply(lambda x: (1 + x).prod() - 1)
print("  annual LS return:")
print("   " + "  ".join(f"{y}:{v:+.0%}" for y, v in by_yr.items()))
print(f"  rolling-12m SR: min {roll.min():.2f}  median {roll.median():.2f}  max {roll.max():.2f}  frac<0 {np.mean(roll<0):.0%}")
worst = r.nsmallest(5)
print("  worst 5 months: " + ", ".join(f"{d.strftime('%Y-%m')} {v:.1%}" for d, v in worst.items()))
print("[done]", flush=True)
