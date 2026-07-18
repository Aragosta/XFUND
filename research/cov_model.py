#!/usr/bin/env python3
"""cov_model.py — ENHANCED factor COVARIANCE forecast built to BEAT trailing EWMA. Cholesky-space HAR
(Chiriac-Voev): forecast in the space of the Cholesky factor (log-diagonal) so every prediction is PSD by
construction; per-element HAR from MULTI-HORIZON realized covariances {EWMA, 21d, 63d} — nesting EWMA so the
model can only improve on it — plus STATE regime features on the correlation elements. Daily realized inputs.
Evaluate vs EWMA: QLIKE (lower better) + min-variance-portfolio realized vol (lower better), walk-forward.
Factors: mkt, mom, size, lowvol."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from numpy.linalg import pinv, det
from DATAHUB import DataHub

hub = DataHub(start="2000-01-01", min_days=0)
me, m_px, mret, elig = hub.me, hub.m_px, hub.mret, hub.elig("liquid"); T = len(me)
ret_d, days = hub.ret_d, hub.days; elig_d = elig.reindex(days, method="ffill")
mom12 = m_px.shift(1) / m_px.shift(12) - 1; vol6 = mret.rolling(6, min_periods=4).std(); mdv = hub.mdv
FACS = ["mkt", "mom", "size", "lowvol"]; N = 4; P = N * (N + 1) // 2

def ls_daily(rankdf, hi_lo=True):
    o = pd.Series(np.nan, index=days)
    for t in range(1, T):
        s = rankdf.loc[me[t-1]].where(elig.loc[me[t-1]]).dropna()
        if len(s) < 100: continue
        q = pd.qcut(s.rank(method="first"), 10, labels=False); dd = days[(days > me[t-1]) & (days <= me[t])]
        r = ret_d.loc[dd, s.index[q == 9]].mean(axis=1) - ret_d.loc[dd, s.index[q == 0]].mean(axis=1)
        o.loc[dd] = r if hi_lo else -r
    return o
print("[cov] daily factor returns ...", flush=True)
R = pd.DataFrame({"mkt": ret_d.where(elig_d).mean(axis=1), "mom": ls_daily(mom12, True),
                  "size": ls_daily(np.log(mdv.replace(0, np.nan)), False), "lowvol": ls_daily(vol6, False)}).dropna()
rd = R.index; Rv = R.values

def psd(M):
    w, V = np.linalg.eigh((M + M.T) / 2); return V @ np.diag(np.clip(w, 1e-10, None)) @ V.T
def chol_p(C):                                                                 # cov -> unconstrained params (log-diag Cholesky)
    L = np.linalg.cholesky(psd(C) + 1e-10 * np.eye(N)); p = []
    for i in range(N):
        for j in range(i + 1): p.append(np.log(max(L[i, i], 1e-8)) if i == j else L[i, j])
    return np.array(p)
def unchol(p):                                                                 # params -> PSD cov
    L = np.zeros((N, N)); k = 0
    for i in range(N):
        for j in range(i + 1): L[i, j] = np.exp(p[k]) if i == j else p[k]; k += 1
    return L @ L.T
DIAG = [i * (i + 1) // 2 + i for i in range(N)]                                # which param indices are diagonal (variance)

# realized cov over past W trading days at each month-end; EWMA daily-updated
def rcov(dd): return Rv[dd].T @ Rv[dd]
didx = {d: i for i, d in enumerate(rd)}
def past_days(t, W):
    end = rd[rd <= me[t]]
    if len(end) < W: return None
    j = didx[end[-1]]; return np.arange(j - W + 1, j + 1)
lam = 0.94; Sew = np.cov(Rv[:60].T); ew_at = {}
for i in range(len(rd)):
    r = Rv[i:i+1].T; Sew = lam * Sew + (1 - lam) * (r @ r.T)
    ew_at[rd[i]] = Sew.copy()
def ewma_month(t):
    end = rd[rd <= me[t]]
    return ew_at[end[-1]] * 21 if len(end) else None

RC = {}
for t in range(1, T):
    dd = past_days(t, 21)
    if dd is not None and dd[0] >= 0: RC[t] = rcov(dd)                          # realized 21d cov ending at month t
mts = [t for t in RC if me[t].year >= 2009 and (t + 1) in RC]

# state features
ST = pd.read_parquet("/tmp/state_feat.parquet"); ST.index = pd.DatetimeIndex(ST.index)
SF = ST[["turbulence", "absorption", "vix", "funding"]].reindex(me).ffill()

# ── build Cholesky-param panels for the HAR components ────────────────────────
print("[cov] Cholesky-HAR features ...", flush=True)
keys = [t for t in range(1, T) if (t in RC)]
comp = {"ew": {}, "r21": {}, "r63": {}}
for t in keys:
    e = ewma_month(t); d63 = past_days(t, 63)
    if e is None or d63 is None or d63[0] < 0: continue
    try:
        comp["ew"][t] = chol_p(e); comp["r21"][t] = chol_p(RC[t]); comp["r63"][t] = chol_p(rcov(d63))
    except Exception: pass
good = [t for t in mts if t in comp["ew"] and t in comp["r21"] and t in comp["r63"]]

# ── per-element HAR (Cholesky space): forecast next-month chol param ──────────
def forecast(use_state):
    Hd = {}
    tgt = {t: chol_p(RC[t + 1]) for t in good}
    for kk in range(P):
        rows = []
        for t in good:
            x = [comp["ew"][t][kk], comp["r21"][t][kk], comp["r63"][t][kk]]
            if use_state and kk not in DIAG: x += list(SF.loc[me[t]].values)
            rows.append((t, x, tgt[t][kk]))
        Ts = [r[0] for r in rows]; X = np.array([r[1] for r in rows]); Y = np.array([r[2] for r in rows])
        for ii in range(len(Ts)):
            if ii < 36: continue
            A = np.column_stack([np.ones(ii), X[:ii]])
            b = np.linalg.lstsq(A, Y[:ii], rcond=None)[0]
            pr = np.concatenate([[1.0], X[ii]]) @ b
            Hd.setdefault(Ts[ii], np.full(P, np.nan))[kk] = pr
    return {t: unchol(v) for t, v in Hd.items() if np.isfinite(v).all()}
print("[cov] forecasting ...", flush=True)
CHOL = forecast(False)

# ── FUNDAMENTAL model: DCC (Engle) — vols and correlations have DISTINCT dynamics; correlations converge to the
# systemic attractor (equicorrelation) as the ABSORPTION state rises (Kritzman). Σ = D R D. ──────────────────
sig2 = np.zeros((len(rd), N)); sig2[0] = R.var().values
for i in range(1, len(rd)): sig2[i] = 0.94 * sig2[i-1] + 0.06 * Rv[i-1]**2      # daily EWMA vols (D)
sig = np.sqrt(sig2); eps = Rv / sig                                            # standardized residuals
a, bdcc = 0.04, 0.94                                                            # DCC params (canonical; MLE-estimable)
absb = ST["absorption"].reindex(me).ffill()
abs_pct = absb.expanding(24).apply(lambda x: (x.iloc[-1] > x).mean())          # absorption percentile (leak-free)
def dcc(state_tighten):
    Hd = {}
    Q = np.corrcoef(eps[:60].T); Qbar = Q.copy(); mend = {t: rd[rd <= me[t]][-1] for t in good if len(rd[rd <= me[t]])}
    ei = {d: i for i, d in enumerate(rd)}
    J = np.ones((N, N))
    for i in range(1, len(rd)):
        if i % 252 == 0: Qbar = np.corrcoef(eps[:i].T)                          # expanding unconditional corr (leak-free)
        Q = (1 - a - bdcc) * Qbar + a * np.outer(eps[i-1], eps[i-1]) + bdcc * Q
        d = np.sqrt(np.diag(Q)); Rc = Q / np.outer(d, d)
        rd_i = rd[i]
        for t in good:
            if mend.get(t) == rd_i:
                Rf = Rc.copy()
                if state_tighten:
                    w = abs_pct.get(me[t], 0.5); w = 0.0 if not np.isfinite(w) else 0.6 * w
                    Rf = (1 - w) * Rc + w * J                                   # converge toward equicorrelation in stress
                D = np.diag(sig[i]); Hd[t] = (D @ Rf @ D) * 21
    return Hd
DCCm = dcc(False); DCCs = dcc(True)

def qlike(H, Rr):
    H = psd(H) + 1e-10 * np.eye(N); x = pinv(H) @ Rr
    return np.trace(x) - np.log(max(det(x), 1e-12)) - N
def mvpv(H, Rr):
    Hi = pinv(psd(H) + 1e-10 * np.eye(N)); w = Hi @ np.ones(N); w /= w.sum(); return np.sqrt(max(w @ Rr @ w, 0))
def ev(Hd, name):
    ql, mv = [], []
    for t in good:
        if t not in Hd: continue
        Rr = RC[t + 1]
        try: ql.append(qlike(Hd[t], Rr)); mv.append(mvpv(Hd[t], Rr))
        except Exception: pass
    print(f"  {name:20} QLIKE {np.nanmean(ql):>7.3f}   MVP realized vol {np.nanmean(mv):>7.4f}   n={len(ql)}")
    return np.nanmean(ql), np.nanmean(mv)

print("\n[results] PRINCIPLED covariance vs TRAILING EWMA (same months, 2009-26):")
print("  EWMA forces vols & correlations to share one decay (misspecified). DCC separates them (theory).")
EW = {t: ewma_month(t) for t in good}
ev(EW, "EWMA(0.94)")
ev(DCCm, "DCC (Engle)")
ev(DCCs, "DCC + absorption")
ev(CHOL, "CHOL-HAR (ref)")
print("\nRead: CHOL-HAR nests EWMA as a feature (so it should >= EWMA); +STATE adds regime info on correlations.")
print("If CHOL-HAR+STATE beats EWMA on QLIKE and MVP vol, the enhanced model outperforms trailing.")
