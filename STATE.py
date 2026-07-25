#!/usr/bin/env python3
"""STATE.py — the shared systematic-state layer (Data -> STATE -> sleeves/ERC/risk). ONE object: a covariance
forecast Sigma_t. Everything the state layer outputs is a read-out of it. No regime classifier, no return-timing
targets, no multi-output vol model — all three were tested and did NOT add end-to-end value (see below).

WHAT SURVIVED THE TESTS (research/state_value.py, state_cov_native.py):
  * THE OBJECT   = EWMA covariance of daily factor+sector returns (hl=63), snapshotted at each month-end.
  * THE DIAL     = vol-target gross overlay  gross = clip(median(sigma_port)/sigma_port, 0.4, 1.0),
                   sigma_port = sqrt(wᵀ Sigma w).  NO hand-picked slope/percentile (median is estimated, not fit).
                   Matches/beats the old hand-tuned Kritzman 0.6/clip overlay at half the turnover.
  * READ-OUTS    = ENB (Meucci effective-#-of-bets = modern absorption) and Mahalanobis surprise (= modern
                   turbulence), both parameter-free functionals of Sigma. DIAGNOSTIC ONLY — the ablation showed
                   they add NO predictive signal over portfolio vol / VIX (all are the same 'overall risk' factor).

WHAT WAS REMOVED (proven dead, do not resurrect):
  - multi-output XGBoost forecasting factor vols  (IC 0.5 but LOST to trailing vol end-to-end; state_size.py)
  - mode-return targets squeeze/reversal/value/mom_crash  (return-timing is the wall; AQR 'factor timing is hard')
  - HMM/GMM and the statistical jump-model regime  (jump model LOST to the one-line vol-target dial)

OUTPUTS: /tmp/state_cov.parquet (flattened Sigma_t snapshots -> ERC/construction),
         /tmp/state_dial.parquet (gross dial + read-outs -> risk layer),
         /tmp/state_feat.parquet (lean state features -> sleeve regime conditioning).
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from DATAHUB import DataHub

hub = DataHub(start="2000-01-01", min_days=0)
me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid"); T = len(me)
ret_d, days, mdv, sec = hub.ret_d, hub.days, hub.mdv, hub.sector
elig_d = elig.reindex(days, method="ffill")
mom12 = m_px.shift(1) / m_px.shift(12) - 1; vol6 = mret.rolling(6, min_periods=4).std()

# ── daily factor returns (the assets the book holds) + sector returns (market coupling) ──
def ls_daily(rankdf, hi_lo=True):
    o = pd.Series(np.nan, index=days)
    for t in range(1, T):
        s = rankdf.loc[me[t-1]].where(elig.loc[me[t-1]]).dropna()
        if len(s) < 100: continue
        q = pd.qcut(s.rank(method="first"), 10, labels=False); dd = days[(days > me[t-1]) & (days <= me[t])]
        r = ret_d.loc[dd, s.index[q == 9]].mean(axis=1) - ret_d.loc[dd, s.index[q == 0]].mean(axis=1)
        o.loc[dd] = r if hi_lo else -r
    return o
print("[state] daily factor + sector returns ...", flush=True)
FD = pd.DataFrame({"mkt": ret_d.where(elig_d).mean(axis=1), "mom": ls_daily(mom12, True),
                   "size": ls_daily(np.log(mdv.replace(0, np.nan)), False), "lowvol": ls_daily(vol6, False)})
SD = {}
if sec is not None:
    for sn, g in pd.Series(sec).dropna().groupby(sec.dropna()):
        cols = [c for c in g.index if c in ret_d.columns]
        if len(cols) >= 8: SD[sn] = ret_d[cols].where(elig_d[cols]).mean(axis=1)
SD = pd.DataFrame(SD)
FACS, SECS = list(FD.columns), list(SD.columns)
R = pd.concat([FD, SD], axis=1).dropna(how="all")

# ── THE ONE OBJECT: EWMA covariance of daily returns, snapshotted at month-ends ──
def ewma_cov_snapshots(Rdf, hl=63):
    lam = 0.5 ** (1.0 / hl); X = Rdf.values; idx = Rdf.index; S = None; snaps = {}
    day_pos = {}
    for mm in me:
        pos = np.searchsorted(idx.values, np.datetime64(mm), side="right") - 1
        if pos >= 0: day_pos[mm] = pos
    want = {p: mm for mm, p in day_pos.items()}
    for j in range(len(idx)):
        x = np.where(np.isfinite(X[j]), X[j], 0.0)
        S = np.outer(x, x) if S is None else lam * S + (1 - lam) * np.outer(x, x)
        if j in want: snaps[want[j]] = S.copy()
    return list(Rdf.columns), snaps
print("[state] EWMA covariance snapshots (the one object) ...", flush=True)
cols, SNAP = ewma_cov_snapshots(R, hl=63)
ci = {c: k for k, c in enumerate(cols)}
fac_ix = [ci[c] for c in FACS]; sec_ix = [ci[c] for c in SECS]

# reference book weights = equal-vol combine of the factors (the risk layer substitutes actual ERC weights)
MFAC = pd.DataFrame({c: (1 + FD[c].fillna(0)).groupby(FD.index.to_period("M")).prod() - 1 for c in FACS})
MFAC.index = MFAC.index.to_timestamp("M"); MFAC = MFAC.reindex(me)
iv = 1.0 / MFAC.rolling(12, min_periods=6).std().shift(1); W = iv.div(iv.sum(axis=1), axis=0)
sec_rm = pd.DataFrame({c: (1 + SD[c].fillna(0)).groupby(SD.index.to_period("M")).prod() - 1 for c in SECS})
sec_rm.index = sec_rm.index.to_timestamp("M"); sec_rm = sec_rm.reindex(me)

# ── read-outs off Sigma_t ──────────────────────────────────────────────────────
sig_port = pd.Series(np.nan, index=me); enb = pd.Series(np.nan, index=me); surp = pd.Series(np.nan, index=me)
facvol = pd.DataFrame(np.nan, index=me, columns=FACS); covflat = {}
for mm in me:
    if mm not in SNAP: continue
    S = SNAP[mm]; covflat[mm] = S[np.ix_(fac_ix, fac_ix)].flatten()
    facvol.loc[mm] = np.sqrt(np.clip(np.diag(S[np.ix_(fac_ix, fac_ix)]) * 21, 0, None))
    w = W.loc[mm, FACS].values
    if np.isfinite(w).all():
        v = float(w @ S[np.ix_(fac_ix, fac_ix)] @ w)
        if v > 0: sig_port.loc[mm] = np.sqrt(v * 21)                    # ~monthly portfolio vol forecast
    Ss = S[np.ix_(sec_ix, sec_ix)]; d = np.sqrt(np.clip(np.diag(Ss), 1e-16, None)); Cor = Ss / np.outer(d, d)
    ev = np.clip(np.linalg.eigvalsh(Cor), 1e-10, None); p = ev / ev.sum()
    enb.loc[mm] = float(np.exp(-(p * np.log(p)).sum()))                 # Meucci ENB (modern absorption)
    rm = sec_rm.loc[mm].values
    if np.isfinite(rm).all():
        try: surp.loc[mm] = float(np.sqrt(rm @ np.linalg.solve(Ss * 21 + 1e-12 * np.eye(len(rm)), rm)))
        except Exception: pass

# ── THE DIAL: vol-target gross overlay (no hand-picked constants) ──────────────
gross = (sig_port.expanding(24).median() / sig_port).clip(0.4, 1.0)     # median estimated monthly, 0.4 = guardrail

# ── lean state feature store for sleeves (regime conditioning; INPUTS not targets) ──
F = pd.DataFrame(index=me)
F["enb"] = enb; F["surprise"] = surp; F["sig_port"] = sig_port; F["gross"] = gross
for f in FACS: F[f"vol_{f}"] = facvol[f]
F["disp"] = mret.where(elig).std(axis=1)
mktm = mret.where(elig).mean(axis=1); mkt_cum = (1 + mktm.fillna(0)).cumprod()
F["mkt_2yr"] = mkt_cum / mkt_cum.shift(24) - 1; F["mkt_1m"] = mktm; F["bear"] = (F["mkt_2yr"] < 0).astype(float)
# ── MOMENTUM-FACTOR AUTOCORRELATION STATE (Ehsani-Linnainmaa crash timer) ──────────
# Mechanism: cross-sectional momentum = timing factor autocorrelation; it pays while the momentum
# factor stays positively autocorrelated and CRASHES when that autocorrelation breaks/flips. So the
# trailing AC(1) of the momentum factor return is a continuous, model-free crash-state INPUT (STATE style).
mf = MFAC["mom"]                                                       # monthly momentum-factor L/S return
def _rollac(s, w): return s.rolling(w, min_periods=max(4, w // 2)).apply(lambda x: pd.Series(x).autocorr(lag=1), raw=False)
F["mom_ac12"] = _rollac(mf, 12); F["mom_ac6"] = _rollac(mf, 6)         # trailing factor-return autocorrelation
if hub.macro_m is not None:                                            # raw macro regime cols (sleeves split on these)
    for c in ["credit", "vix", "slope_2s10s", "breakeven", "y10", "funding", "vix_ts"]:
        if c in hub.macro_m: F[c] = hub.macro_m[c]
try:
    lbm = np.log(((hub.fund("book") / hub.mcap()).replace([np.inf, -np.inf], np.nan)).where(lambda z: z > 0))
    vs = lbm.where(elig); F["valspread"] = (vs.quantile(0.8, axis=1) - vs.quantile(0.2, axis=1)).reindex(me)
except Exception: pass
try:
    g = pd.read_parquet("data/fnspid/sent_monthly.parquet"); g["d"] = g["per"].map({d.to_period("M").__str__(): d for d in me}); g = g.dropna(subset=["d"])
    F["sentiment"] = g.groupby("d").apply(lambda x: np.average(x["mean"], weights=x["count"])).reindex(me)
except Exception: pass

# ── save + report ──────────────────────────────────────────────────────────────
pd.DataFrame(covflat, index=[f"{a}_{b}" for a in FACS for b in FACS]).T.to_parquet("/tmp/state_cov.parquet")
pd.DataFrame({"gross": gross, "sig_port": sig_port, "enb": enb, "surprise": surp}).to_parquet("/tmp/state_dial.parquet")
F.to_parquet("/tmp/state_feat.parquet")

nxt = MFAC.mul(W).sum(axis=1).shift(-1); Dw = F[F.index >= "2011-01-01"]
def dcorr(c):
    e = pd.DataFrame({"x": Dw[c], "y": nxt.reindex(Dw.index)}).dropna(); return e["x"].corr(e["y"]) if len(e) > 24 else np.nan
print("\n[state] ONE object (Sigma_t) -> read-outs. Diagnostics (corr with next-mo ref-book return, 2011-26):")
print(f"   ENB {dcorr('enb'):+.2f} (low=coupled, +exp)   surprise {dcorr('surprise'):+.2f}   sig_port {dcorr('sig_port'):+.2f}   vix {dcorr('vix'):+.2f}")
# MOM-AUTOCORR crash-timer diagnostic: does trailing factor AC lead momentum's own forward return / crash?
mnx1 = mf.shift(-1); mf6 = mf.rolling(6).sum().shift(-6)               # next-1mo and next-6mo momentum-factor return (fwd)
def _c(a, b):
    e = pd.DataFrame({"x": Dw[a] if a in Dw else F[a].reindex(Dw.index), "y": b.reindex(Dw.index)}).dropna(); return e["x"].corr(e["y"]) if len(e) > 24 else np.nan
lo = Dw["mom_ac12"] < Dw["mom_ac12"].median()
print(f"[state] MOM-AC crash timer: corr(ac12, mom_next1) {_c('mom_ac12', mnx1):+.2f}  corr(ac12, mom_fwd6) {_c('mom_ac12', mf6):+.2f}"
      f"   mom_fwd6 mean  loAC {mf6.reindex(Dw.index)[lo].mean():+.3f}  hiAC {mf6.reindex(Dw.index)[~lo].mean():+.3f}"
      f"   ac12 range [{Dw['mom_ac12'].min():+.2f},{Dw['mom_ac12'].max():+.2f}]")
print(f"   ENB range [{enb.dropna().min():.1f},{enb.dropna().max():.1f}] / {len(SECS)} sectors   gross range [{gross.dropna().min():.2f},{gross.dropna().max():.2f}]  mean {gross.mean():.2f}")
print("\nSaved: /tmp/state_cov.parquet (Sigma_t -> ERC/construction), /tmp/state_dial.parquet (gross dial -> risk layer),")
print("       /tmp/state_feat.parquet (lean regime features -> sleeves). ONE object, read-outs only. No regime model.")
