#!/usr/bin/env python3
"""
mom_layer.py — the MOM sleeve  ✅ settled

ONE object: a cross-sectional forecast of each name's forward TREND SIGNIFICANCE, read out as a
monthly per-name score the ERC/META layer ranks on.

The settled recipe (decided by research T7–T21, see ../research/MOM_research.md):

  signal / TARGET   tval = slope ÷ SE of the forward 6-month log-price path (significance-weighted
                    trend; the "honest trend-scan" with a FIXED horizon, no window selection).
                    Beats the plain-return target (T19: IC .091 vs .079). Kept as-is.
  HORIZON           single 6 months. Multi-horizon blending does NOT raise SR — it only ensembles
                    the same signal (variance-reduction DD tilt), and is redundant with a rich
                    feature set, so it is NOT included here (T20/T21).
  REPRESENTATION    predict the histogram of the (rank-bucketed) target with a softprob classifier,
                    then RECLASSIFY: RET = Σ pₖ·centerₖ (law of total expectation) and rank on RET.
                    This is DM's estimator applied to MOM's target.
  CENTERS           Gaussian-quantile bin centers (not integer) — a free, deterministic +0.03 SR
                    over uniform centers at identical IC/turnover (T21 hist_gauss).
  FEATURES          the DM feature set (features.make_features): Han cumulative-momentum
                    (windows 1,3,6,9,12,18) + dynamics (accel / vol / %-up) + size-decile dummies,
                    all cross-sectionally z-scored. (Gaussian-ranking the features barely moved it,
                    so we keep DM's z-scored set unchanged.)
  MODEL             XGBoost multi:softprob, 10 classes, seed-ensembled.
  TRAINING          rolling window (default 72m), refit semiannually, embargo k−6 so every training
                    label is fully realized before the prediction month (leak-free, audited T19).
  DEPLOYMENT        cross-sectional (like DM): rank names each month, long top decile + discard
                    bottom. The layer emits the raw score; construction/sizing is the risk layer's job.

REJECTED and therefore absent (../research/): magnitude composites (tval×return), residual-tval target,
nested / disjoint-return multi-horizon, cubic+ path terms, deflation beyond α≈1, point-regression readout.
"""
import warnings, os, sys
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from xgboost import XGBClassifier
import BACKTEST
from DATAHUB import DataHub
from UNIVERSE import ffd_scores                                   # eligibility now comes from DATAHUB (hub.elig) — global filter
from features import make_features, MOM_WINDOWS

KB, HZ = 10, 6                                                    # 10 histogram bins, 6-month target horizon
CEN = norm.ppf((np.arange(KB) + 0.5) / KB)                       # Gaussian-quantile bin centers (the free win)
REGC = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
def _bkt(y, K):                                                  # cross-sectional rank -> integer bin 0..K-1
    r = pd.Series(y).rank(method="first").values; return np.clip((r - 0.5) / len(r) * K, 0, K - 1).astype(int)


class MomLayer:
    def __init__(self, hub: DataHub | None = None, start: str = "2000-01-01", seeds: int = 5,
                 maxtrain: int = 72, topn: int | None = None, oos_year: int = 2011,
                 addts: bool = True, addffd: bool = False, pool: bool = False, tier: str = "liquid"):
        # addts (DM default ON): 52wk-high / trend-R² / tsmom. addffd, pool: available, DM-default OFF.
        # tier: DATAHUB GLOBAL long universe — 'liquid'|'relaxed'|'base'. Shorts ALWAYS clear hub.elig('shortable')
        #       (borrowable, mdv > $25M) — the global borrow filter, shared by all sleeves.
        self.hub = hub or DataHub(start=start, min_days=0)
        self.seeds, self.maxtrain, self.topn, self.oos_year = seeds, maxtrain, topn, oos_year
        self.addts, self.addffd, self.pool_f = addts, addffd, pool
        self.tier = tier

    # ── data + the tval target ────────────────────────────────────────────────
    def _prep(self):
        h = self.hub
        self.pm = h.delisted_prices("monthly")                   # delisting-injected monthly prices
        self.rm = h.clean_returns("monthly")                     # winsorized monthly returns (feature input)
        self.sm = h.dollar_size("monthly")                       # close × month-end volume (size input)
        self.elig = self.hub.elig(self.tier, "monthly")             # DATAHUB global long universe
        self.short_elig = self.hub.elig("shortable", "monthly")     # DATAHUB global BORROW filter (mdv > $25M)
        self.pnl = h.pnl("monthly")
        self.me = self.rm.index; self.T = len(self.me); self.cols = self.rm.columns
        self._build_target(); self._build_aux()

    def _build_aux(self):
        """DM's extra feature blocks appended alongside make_features(): TS trend (default), FFD, MOM-pool."""
        pm, rm = self.pm, self.rm
        self.FFD = None
        if self.addffd:
            F = ffd_scores(pm, 5 * 12 + max(MOM_WINDOWS) + 1)                 # frac-diff log-price scores
            self.FFD = {m: F[m].reindex(rm.index) for m in F}
        self.TSF = None
        if self.addts:                                                       # 52wk-high, trend-R², tsmom (DM ADDTS)
            logpm = np.log(pm); mi = pd.Series(np.arange(len(pm)), index=pm.index); w = 6
            xd = mi - mi.rolling(w, min_periods=4).mean()
            yd = logpm.sub(logpm.rolling(w, min_periods=4).mean(), axis=0)
            cxy = (yd.mul(xd, axis=0)).rolling(w, min_periods=4).mean()
            r2 = (cxy ** 2).div((xd ** 2).rolling(w, min_periods=4).mean(), axis=0).div(yd.pow(2).rolling(w, min_periods=4).mean() + 1e-12)
            self.TSF = {"hi52": pm / pm.rolling(12, min_periods=8).max() - 1, "trendR2": r2,
                        "tsmom": (pm / pm.shift(6) - 1).where(lambda z: z.abs() < 5) * r2}
        self.POOLF = None
        if self.pool_f:                                                      # MOM resmom + Baz MACD composite
            mktm = rm.mean(axis=1)
            bcov = rm.mul(mktm, axis=0).rolling(36, min_periods=24).mean() - rm.rolling(36, min_periods=24).mean().mul(mktm.rolling(36, min_periods=24).mean(), axis=0)
            bvar = (mktm ** 2).rolling(36, min_periods=24).mean() - mktm.rolling(36, min_periods=24).mean() ** 2
            RES = rm.sub((bcov.div(bvar, axis=0).shift(1)).mul(mktm, axis=0))
            resmom = RES.shift(1).rolling(11, min_periods=8).sum() / (RES.rolling(11, min_periods=8).std().shift(1) + 1e-9)
            hl = lambda s: np.log(0.5) / np.log(1 - 1 / s); comp = 0.0
            for Sh, Lg in [(2, 6), (4, 12), (8, 24)]:
                q = (pm.ewm(halflife=hl(Sh)).mean() - pm.ewm(halflife=hl(Lg)).mean()) / pm.rolling(12, min_periods=6).std()
                y = q / q.rolling(24, min_periods=12).std(); comp = comp + y * np.exp(-y ** 2 / 4) / 0.89
            self.POOLF = {"resmom": resmom, "macd": comp}

    def _build_target(self):
        """TVAL[t, i] = slope/SE of stock i's forward log-price over months [t, t+HZ] (a t-statistic)."""
        lp = np.log(self.pm.values); T = self.T; n = HZ + 1
        x = np.arange(n).astype(float); xc = x - x.mean(); Sxx = (xc ** 2).sum()
        TV = pd.DataFrame(np.nan, index=self.me, columns=self.cols)
        for t in range(T - HZ - 1):
            Y = lp[t:t + n]; Ym = np.nanmean(Y, 0); sl = np.nansum(xc[:, None] * (Y - Ym), 0) / Sxx
            resd = Y - (Ym + xc[:, None] * sl); se = np.sqrt(np.nansum(resd ** 2, 0) / max(HZ - 1, 1) / Sxx)
            TV.loc[self.me[t]] = sl / (se + 1e-9)
        self.TVAL = TV

    # ── walk-forward cross-sectional model → monthly per-name RET score ─────────
    def _features_at(self, t):
        """DM feature matrix at t: make_features (Han+dynamics+size [+FFD]) + TS trend [+ MOM-pool]."""
        F = make_features(self.rm, self.sm, t, ffd_scores=self.FFD).dropna()
        if self.TSF is not None:
            for nm, df in self.TSF.items(): F[nm] = df.iloc[t - 1].reindex(F.index)
        if self.POOLF is not None:
            for nm, df in self.POOLF.items(): F[nm] = df.iloc[t - 1].reindex(F.index)
        return F

    def build_pool(self):
        self._prep()
        first = max(MOM_WINDOWS) + 1
        pool = {}
        for t in range(first, self.T - HZ - 1):                  # need features (t-1) AND realized target (t+HZ)
            F = self._features_at(t)
            e = self.elig.iloc[t - 1]; idx = F.index.intersection(e.index[e.values])
            if self.topn is not None:                            # optional liquidity cap (research/backtest speed)
                idx = self.sm.iloc[t - 1].reindex(idx).dropna().nlargest(self.topn).index
            tv = self.TVAL.iloc[t].reindex(idx); ok = tv.notna().values
            idx = idx[ok]
            if len(idx) < 200: continue
            fwd6 = (np.exp(np.log(self.pm.iloc[t + HZ]) - np.log(self.pm.iloc[t])) - 1).reindex(idx).values  # 6m fwd ret (DM target)
            short_ok = self.short_elig.iloc[t - 1].reindex(idx).fillna(False).values.astype(bool)  # borrowable subset (PIT, lag 1)
            pool[t] = dict(dt=self.me[t], idx=idx, X=F.loc[idx].values.astype(np.float32),
                           tval=tv.loc[idx].values, fwd=self.rm.iloc[t].reindex(idx).values, fwd6=fwd6, short_ok=short_ok)
        self.pool = pool
        keys = sorted(pool); self.keys = keys
        self.feat_names = self._features_at(keys[0]).columns.tolist()
        return pool

    def build(self):
        if not hasattr(self, "pool"): self.build_pool()
        pool, keys = self.pool, self.keys
        score = {}; mdl = None
        for k in keys:
            dt = pool[k]["dt"]
            if dt.year < self.oos_year:                          # warm-up: no OOS score before oos_year
                continue
            if dt.month in (1, 7) or mdl is None:                # semiannual refit
                tr = [j for j in keys if j <= k - HZ][-self.maxtrain:]   # embargo: label realized by k
                if len(tr) >= 24:
                    Xt = np.vstack([pool[j]["X"] for j in tr])
                    yl = np.concatenate([_bkt(pool[j]["tval"], KB) for j in tr])
                    mdl = [XGBClassifier(**REGC, objective="multi:softprob", num_class=KB, random_state=s).fit(Xt, yl)
                           for s in range(self.seeds)]
            if mdl is None: continue
            P = np.mean([m.predict_proba(pool[k]["X"]) for m in mdl], axis=0)
            score[k] = pd.Series(P @ CEN, index=pool[k]["idx"])   # RET = Σ pₖ·centerₖ
        self.score = pd.DataFrame({pool[k]["dt"]: score[k] for k in score}).T.reindex(columns=self.cols)
        return self.score

    # ── validation book: constructed via the shared CONSTRUCT layer (liquidity sizing + banding), net of costs ──
    def backtest(self, weighting: str = "mdv", band=(0.10, 0.20), decile: float = 0.10, ls: bool = False):
        # Construction is the SHARED CONSTRUCT layer (scores→weights). Default = mdv-weighted + banded = the +0.47-SR
        # win over the old equal-weight decile (research/MOM_research.md T29). Knobs: weighting/band (capacity setting
        # = sqrt_mdv, band=(.10,.30)). ls=False long-only (discard the rest); ls=True dollar-neutral, borrowable shorts.
        import RISK
        tc = BACKTEST.tiered_transaction_costs(self.sm); bf = BACKTEST.tiered_borrow_fees(self.sm)
        W = RISK.risk_book(self.score, self.hub, tier=self.tier, weighting=weighting, band=band, decile=decile, ls=ls)
        ics = []                                               # rank-IC of the score vs next-month realized (diagnostic)
        for k in self.keys:
            dt = self.pool[k]["dt"]
            if dt not in self.score.index: continue
            s = self.score.loc[dt].reindex(self.pool[k]["idx"]).dropna()
            if len(s) < 20: continue
            fwd = pd.Series(self.pool[k]["fwd"], index=self.pool[k]["idx"]).reindex(s.index)
            ics.append(spearmanr(s.values, fwd.values).correlation)
        r = BACKTEST.backtest(W, self.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
        r["rankIC"] = float(np.nanmean(ics)); return r

    def save(self, outdir: str | None = None):
        outdir = outdir or os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
        os.makedirs(outdir, exist_ok=True)
        self.score.to_parquet(os.path.join(outdir, "mom_score.parquet"))
        return outdir


if __name__ == "__main__":
    seeds = int(os.environ.get("SEEDS", 5)); topn = os.environ.get("TOPN"); topn = int(topn) if topn else None
    addffd = bool(int(os.environ.get("ADDFFD", 0))); poolf = bool(int(os.environ.get("POOL", 0)))
    tier = os.environ.get("TIER", "liquid")                          # DATAHUB global universe tier
    print(f"[MOM] build  seeds={seeds}  maxtrain=72  tier={tier}  topn={topn or 'all'}  target=tval@6  centers=gaussian"
          f"  feats=DM(make_features+TS{'+FFD' if addffd else ''}{'+pool' if poolf else ''})  shorts=borrowable", flush=True)
    ml = MomLayer(seeds=seeds, topn=topn, addffd=addffd, pool=poolf, tier=tier)
    ml.build()
    rL = ml.backtest(ls=False); rLS = ml.backtest(ls=True)
    print(f"[MOM] features ({len(ml.feat_names)}): {', '.join(ml.feat_names)}", flush=True)
    print(f"[MOM] OOS long-decile net: rankIC {rL['rankIC']:.4f}  SR {rL['sharpe']:.2f}  ann {rL['ann_return']:.1%}"
          f"  maxDD {rL['max_drawdown']:.1%}  turn {rL['ann_turnover']:.1f}", flush=True)
    print(f"[MOM] OOS L/S dollar-neutral net: rankIC {rLS['rankIC']:.4f}  SR {rLS['sharpe']:.2f}  ann {rLS['ann_return']:.1%}"
          f"  maxDD {rLS['max_drawdown']:.1%}  turn {rLS['ann_turnover']:.1f}", flush=True)
    out = ml.save(); print(f"[MOM] saved score -> {out}/mom_score.parquet  ({ml.score.shape[0]} months × {ml.score.shape[1]} names)", flush=True)
