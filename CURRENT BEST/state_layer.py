#!/usr/bin/env python3
"""
STATE LAYER — CURRENT BEST
==========================
The shared systematic-state layer:  DATA -> [STATE] -> sleeves / ERC / risk.

ONE object: a covariance forecast Sigma_t of daily factor + sector returns (EWMA, hl=63 = the RiskMetrics
standard). Everything the state layer outputs is a read-out of Sigma_t. There is NO regime classifier, NO
return-timing target, NO learned vol model — all were tested and none added end-to-end value.

OUTPUTS (all leak-free; each month-end value uses only data up to that month-end):
  gross     : the vol-target gross dial in [0.4, 1.0]  — the risk layer's exposure multiplier.
              gross = clip( expanding_median(sigma_port) / sigma_port , 0.4, 1.0 )
              No hand-picked slope/percentile: the median is ESTIMATED monthly (not fit to P&L); 0.4 is a
              leverage guardrail. Matches/beats the old hand-tuned Kritzman overlay at half the turnover.
  sigma_port: forecast monthly vol of the reference book (sqrt(w' Sigma w)); the risk layer substitutes its
              own post-ERC weights w to get the book-specific dial.
  Sigma     : the factor covariance snapshots (4x4 per month) -> ERC / construction.
  facvol    : per-factor forecast vols (diag of Sigma).
  enb       : Meucci effective-number-of-bets = exp(-sum p_i ln p_i) over correlation eigenvalues.
              The parameter-free generalization of Kritzman's absorption ratio (no arbitrary top-k). DIAGNOSTIC.
  surprise  : Mahalanobis distance of the month's sector returns under Sigma (generalized turbulence). DIAGNOSTIC.
  feat      : a lean feature store for sleeves (raw macro regime cols + dispersion + valuespread + sentiment).

WHY THIS AND NOT MORE (each proven on our data, see research/ + memory 'state-layer'):
  * regime models (HMM/GMM/statistical jump model) LOSE to this one-line dial.
  * a multi-output XGBoost vol forecaster has IC ~0.5 but LOSES to trailing vol end-to-end.
  * mode-return targets (squeeze/reversal/value/mom-crash) are return-timing = the wall (AQR 'factor timing is hard').
  * dynamic / multi-horizon (HAR, ensemble, +VIX) forecasts are MORE accurate but do NOT beat EWMA-63 into this
    dial — the coarse [0.4,1.0] clip can't spend the extra accuracy. (Revisit VIX-blend only if/when the dial
    becomes a fine daily no-trade-band, where reaction speed pays.)
  * ENB / coupling read-outs add NO predictive signal over portfolio vol — they are diagnostics, not sizing inputs.

The honest scope: STATE = Sigma_t -> portfolio vol -> vol-target gross dial, with ENB/surprise as free read-outs.
"""
from __future__ import annotations
import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)                       # import DATAHUB
os.chdir(_REPO)                                 # DATAHUB reads cwd-relative data/ paths -> run as if from repo root
from DATAHUB import DataHub

# ── settled constants (see module docstring for why these and not learned) ─────
HL          = 63      # EWMA half-life in trading days = RiskMetrics 60-day span standard
GROSS_FLOOR = 0.40    # leverage guardrail (never de-risk below 40% gross)
MED_WIN     = 24      # expanding-median warmup for the dial normalizer (months)
MIN_SEC     = 8       # min names per sector to include it in the coupling cross-section
FACS        = ["mkt", "mom", "size", "lowvol"]


class StateLayer:
    """Build the covariance forecast and its read-outs. Use `.build()` then read the attributes."""

    def __init__(self, hub: DataHub | None = None, start: str = "2000-01-01"):
        self.hub = hub if hub is not None else DataHub(start=start, min_days=0)
        self.built = False

    # -- factor & sector daily returns -------------------------------------------------
    def _daily_returns(self):
        h = self.hub
        me, elig = h.me, h.elig("liquid"); T = len(me)
        ret_d, days, mdv = h.ret_d, h.days, h.mdv
        elig_d = elig.reindex(days, method="ffill")
        mom12 = h.m_px.shift(1) / h.m_px.shift(12) - 1
        vol6 = h.mret.rolling(6, min_periods=4).std()

        def ls_daily(rankdf, hi_lo=True):
            o = pd.Series(np.nan, index=days)
            for t in range(1, T):
                s = rankdf.loc[me[t-1]].where(elig.loc[me[t-1]]).dropna()
                if len(s) < 100:
                    continue
                q = pd.qcut(s.rank(method="first"), 10, labels=False)
                dd = days[(days > me[t-1]) & (days <= me[t])]
                r = ret_d.loc[dd, s.index[q == 9]].mean(axis=1) - ret_d.loc[dd, s.index[q == 0]].mean(axis=1)
                o.loc[dd] = r if hi_lo else -r
            return o

        FD = pd.DataFrame({
            "mkt":    ret_d.where(elig_d).mean(axis=1),
            "mom":    ls_daily(mom12, True),
            "size":   ls_daily(np.log(mdv.replace(0, np.nan)), False),
            "lowvol": ls_daily(vol6, False),
        })
        SD = {}
        if h.sector is not None:
            for sn, g in pd.Series(h.sector).dropna().groupby(h.sector.dropna()):
                cols = [c for c in g.index if c in ret_d.columns]
                if len(cols) >= MIN_SEC:
                    SD[sn] = ret_d[cols].where(elig_d[cols]).mean(axis=1)
        return FD, pd.DataFrame(SD)

    # -- EWMA covariance snapshots at month-ends (the one object) -----------------------
    def _ewma_cov_snapshots(self, Rdf, me):
        lam = 0.5 ** (1.0 / HL); X = Rdf.values; idx = Rdf.index; S = None; snaps = {}
        want = {}
        for mm in me:
            pos = np.searchsorted(idx.values, np.datetime64(mm), side="right") - 1
            if pos >= 0:
                want[pos] = mm
        for j in range(len(idx)):
            x = np.where(np.isfinite(X[j]), X[j], 0.0)
            S = np.outer(x, x) if S is None else lam * S + (1 - lam) * np.outer(x, x)
            if j in want:
                snaps[want[j]] = S.copy()
        return snaps

    # -- build ------------------------------------------------------------------------
    def build(self):
        h = self.hub; me = h.me
        FD, SD = self._daily_returns()
        SECS = list(SD.columns)
        R = pd.concat([FD, SD], axis=1).dropna(how="all")
        cols = list(R.columns); ci = {c: k for k, c in enumerate(cols)}
        fac_ix = [ci[c] for c in FACS]; sec_ix = [ci[c] for c in SECS]
        SNAP = self._ewma_cov_snapshots(R, me)

        # reference book = equal-vol combine of the factors (risk layer substitutes actual ERC weights)
        MFAC = pd.DataFrame({c: (1 + FD[c].fillna(0)).groupby(FD.index.to_period("M")).prod() - 1 for c in FACS})
        MFAC.index = MFAC.index.to_timestamp("M"); MFAC = MFAC.reindex(me)
        ivv = 1.0 / MFAC.rolling(12, min_periods=6).std().shift(1)
        W = ivv.div(ivv.sum(axis=1), axis=0)
        sec_rm = pd.DataFrame({c: (1 + SD[c].fillna(0)).groupby(SD.index.to_period("M")).prod() - 1 for c in SECS})
        sec_rm.index = sec_rm.index.to_timestamp("M"); sec_rm = sec_rm.reindex(me)

        sig_port = pd.Series(np.nan, index=me); enb = pd.Series(np.nan, index=me); surp = pd.Series(np.nan, index=me)
        facvol = pd.DataFrame(np.nan, index=me, columns=FACS); Sigma = {}
        for mm in me:
            if mm not in SNAP:
                continue
            S = SNAP[mm]; Sf = S[np.ix_(fac_ix, fac_ix)]
            Sigma[mm] = Sf
            facvol.loc[mm] = np.sqrt(np.clip(np.diag(Sf) * 21, 0, None))
            w = W.loc[mm, FACS].values
            if np.isfinite(w).all():
                v = float(w @ Sf @ w)
                if v > 0:
                    sig_port.loc[mm] = np.sqrt(v * 21)                        # ~monthly book vol forecast
            Ss = S[np.ix_(sec_ix, sec_ix)]
            d = np.sqrt(np.clip(np.diag(Ss), 1e-16, None)); Cor = Ss / np.outer(d, d)
            ev = np.clip(np.linalg.eigvalsh(Cor), 1e-10, None); p = ev / ev.sum()
            enb.loc[mm] = float(np.exp(-(p * np.log(p)).sum()))               # Meucci ENB
            rm = sec_rm.loc[mm].values
            if np.isfinite(rm).all():
                try:
                    surp.loc[mm] = float(np.sqrt(rm @ np.linalg.solve(Ss * 21 + 1e-12 * np.eye(len(rm)), rm)))
                except Exception:
                    pass

        # THE DIAL — vol-target, no hand-picked constants
        gross = (sig_port.expanding(MED_WIN).median() / sig_port).clip(GROSS_FLOOR, 1.0)

        # lean sleeve feature store (regime conditioning INPUTS, not targets)
        F = pd.DataFrame(index=me)
        F["enb"] = enb; F["surprise"] = surp; F["sig_port"] = sig_port; F["gross"] = gross
        for f in FACS:
            F[f"vol_{f}"] = facvol[f]
        F["disp"] = h.mret.where(h.elig("liquid")).std(axis=1)
        mktm = h.mret.where(h.elig("liquid")).mean(axis=1); mkt_cum = (1 + mktm.fillna(0)).cumprod()
        F["mkt_2yr"] = mkt_cum / mkt_cum.shift(24) - 1; F["mkt_1m"] = mktm; F["bear"] = (F["mkt_2yr"] < 0).astype(float)
        if h.macro_m is not None:
            for c in ["credit", "vix", "slope_2s10s", "breakeven", "y10", "funding", "vix_ts"]:
                if c in h.macro_m:
                    F[c] = h.macro_m[c]
        try:
            lbm = np.log(((h.fund("book") / h.mcap()).replace([np.inf, -np.inf], np.nan)).where(lambda z: z > 0))
            vs = lbm.where(h.elig("liquid"))
            F["valspread"] = (vs.quantile(0.8, axis=1) - vs.quantile(0.2, axis=1)).reindex(me)
        except Exception:
            pass
        try:
            g = pd.read_parquet("data/fnspid/sent_monthly.parquet")
            g["d"] = g["per"].map({d.to_period("M").__str__(): d for d in me}); g = g.dropna(subset=["d"])
            F["sentiment"] = g.groupby("d").apply(lambda x: np.average(x["mean"], weights=x["count"])).reindex(me)
        except Exception:
            pass

        # expose
        self.me = me; self.Sigma = Sigma; self.facvol = facvol
        self.sig_port = sig_port; self.gross = gross; self.enb = enb; self.surprise = surp
        self.ref_weights = W; self.ref_book = (W * MFAC).sum(axis=1); self.feat = F; self.n_sectors = len(SECS)
        self.built = True
        return self

    def book_gross(self, weights: pd.DataFrame) -> pd.Series:
        """Book-specific dial: pass your post-ERC monthly factor weights (columns = FACS) to get the vol-target
        gross for YOUR book (uses the same Sigma_t and the same parameter-free dial rule)."""
        assert self.built, "call .build() first"
        sig = pd.Series(np.nan, index=self.me)
        for mm in self.me:
            if mm not in self.Sigma:
                continue
            w = weights.reindex(columns=FACS).loc[mm].values if mm in weights.index else None
            if w is not None and np.isfinite(w).all():
                v = float(w @ self.Sigma[mm] @ w)
                if v > 0:
                    sig.loc[mm] = np.sqrt(v * 21)
        return (sig.expanding(MED_WIN).median() / sig).clip(GROSS_FLOOR, 1.0)

    def save(self, outdir: str | None = None):
        """Persist the outputs. Defaults to this folder's ./out/ ."""
        assert self.built, "call .build() first"
        outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
        os.makedirs(outdir, exist_ok=True)
        pd.DataFrame({m: s.flatten() for m, s in self.Sigma.items()},
                     index=[f"{a}_{b}" for a in FACS for b in FACS]).T.to_parquet(os.path.join(outdir, "state_cov.parquet"))
        pd.DataFrame({"gross": self.gross, "sig_port": self.sig_port,
                      "enb": self.enb, "surprise": self.surprise}).to_parquet(os.path.join(outdir, "state_dial.parquet"))
        self.feat.to_parquet(os.path.join(outdir, "state_feat.parquet"))
        return outdir


if __name__ == "__main__":
    st = StateLayer().build()
    outdir = st.save()
    g = st.gross.dropna()
    nxt = st.ref_book.shift(-1); Dw = st.feat[st.feat.index >= "2011-01-01"]
    def dcorr(c):
        e = pd.DataFrame({"x": Dw[c], "y": nxt.reindex(Dw.index)}).dropna()
        return e["x"].corr(e["y"]) if len(e) > 24 else np.nan
    print("\nSTATE LAYER (CURRENT BEST) — one object (Sigma_t), read-outs only.")
    print(f"  gross dial  range [{g.min():.2f}, {g.max():.2f}]  mean {g.mean():.2f}   (floor {GROSS_FLOOR}, EWMA hl={HL})")
    print(f"  ENB         range [{st.enb.dropna().min():.1f}, {st.enb.dropna().max():.1f}]  over {st.n_sectors} sectors")
    print("  diagnostics (corr with next-mo ref-book return, 2011-26; DIAGNOSTIC ONLY, not sizing inputs):")
    print(f"    ENB {dcorr('enb'):+.2f} (low=coupled)   surprise {dcorr('surprise'):+.2f}   "
          f"sig_port {dcorr('sig_port'):+.2f}   vix {dcorr('vix'):+.2f}")
    print(f"  saved -> {outdir}/  (state_cov.parquet, state_dial.parquet, state_feat.parquet)")
