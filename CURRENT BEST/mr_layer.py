#!/usr/bin/env python3
"""
mr_layer.py — the MR (mean-reversion) sleeve  ✅ settled (research T1–T25, ../research/MR_research.md)

ONE object: a daily cross-sectional forecast of each OVERSOLD name's forward IDIOSYNCRATIC reversion, read out
as a long-only book — a ~0-correlation diversifier to the momentum sleeves (net SR ~0.60, corr ≈ −0.6 to DM).

The settled recipe (decided by END-TO-END evidence, not IC — see MR_research.md T22–T25):

  UNIVERSE      liquid tier (hub.elig); oversold EVENT trigger QP<15 (own-history rarity of the 3-day move).
  TARGET        Da-Liu-Schaumburg SECTOR-RESIDUAL forward 5-day reversion, ASYMMETRICALLY CLIPPED at −5%
                (floor knives = noise, keep bounce upside = signal). Structural equivalent: L1/median loss on
                the UNCLIPPED residual (T21). The raw return / its sign are unpredictable (AUC 0.50, T3); the
                idiosyncratic residual is the mean-reverting object (T5, Da-Liu-Schaumburg 2014 / Avellaneda-Lee).
  WHY CLIP/L1   MSE explodes on the −90-skew knife tail (T13 anatomy). Any robust loss (L1 / clip / winsor)
                lifts target IC +50–80% (T20/T21). The one-sidedness (floor knives, keep bounces) IS the DLS
                info-vs-liquidity economics; it belongs in the TARGET, not the loss tilt (asymmetric quantile
                loss went the wrong way, T21).
  FEATURES      reversion set (QP{3,5,10}, rvol{21,63}, turnover ts+cs, ROC{5,21,63,126,252}, dist-200SMA,
                RSI{2,14}, IBS, nATR)
                + SECTOR-RESIDUAL move (qp3_sr, roc5_sr, roc21_sr)                          [RESID_FEAT]
                + ENGINEERED DLS info-vs-liquidity: OVERNIGHT-gap vs INTRADAY split (on3/id3/gapfrac/gapdn =
                  news/KNIFE vs order-flow/BOUNCE), Amihud illiquidity, high-low spread, downside-dev,
                  worst-drop                                                                [ENGFEAT]
                + LONG-SIDE bounce-magnitude: Bollinger-z, short-stretch, lag-1 autocorr, variance-ratio,
                  volume-exhaustion (discriminate WHICH oversold names bounce most)         [LONGFEAT]
                All cross-sectionally Gaussian-ranked, PIT at close[t]. ENG+LONG are complementary — together
                they move GROSS SR (0.56→0.59); either alone does not (T25). We have OHLC (the "close-only"
                assumption was wrong) — the overnight/intraday split is the DLS separator finally operationalized.
  MODEL         XGBoost regressor, objective=reg:squarederror on the CLIPPED target (or reg:absoluteerror on the
                unclipped residual — same effect). Trained on the QP<15 oversold subset, walk-forward
                (train 1512d, annual retrain, embargo the H=5 label).
  CONSTRUCTION  sector-neutral scores → EWMA-smooth(3) the alpha (turnover: 25×→7× at zero SR loss, T12) → long
                top-150 EQUAL-weight (breadth, NOT conviction — reversion is a breadth premium, T22) → ~10-day
                hold. KNIFE-VETO on the long leg (drop most-idiosyncratic-drop pct<0.2 + top-vol pct>0.8, T14).
                LONG-ONLY: the short leg is the squeeze/borrow tax (T24, short-leg-is-the-tax).
  RESULT        net SR ~0.60, maxDD −47%, turnover ~4.8×/yr; corr ≈ −0.6 to DM (short-horizon reversion is
                anti-momentum) → a genuine diversifier. The layer emits the raw daily book; sizing/blend is the
                ERC/META layer's job.

CEILING (why we stopped, T24/T25): signal quality is NOT the binding constraint — IC tripled (0.017→0.049) with
ZERO net gain, because the extra signal was KNIFE-side (bottom) and a long-only book only trades the TOP. The
long side is near-featureless WITHIN the top; net is clamped by the top-N/cost/turnover structure (the parked
turnover-construction lever). MR is a ~0-corr net-0.60 diversifier — that is its job, not a standalone star.

REJECTED and therefore absent (../research/MR_research.md): raw/sign targets (AUC 0.50, T3/T4); OU s-score state
as features (redundant with price features, T19); model-free stationarity gates (T18); RMT/MP factor denoising as
a gross lever (T18); distributional Stop-Regressing / HL-Gauss on trees (loses to plain L1, T21); asymmetric
quantile loss (T21); conviction-weighting (breadth wins, T22); knife-side features to lift a long book (IC↑ but
gross flat, T24); the overnight-reversal SLEEVE (untradeable on the daily-close engine, T16 — the overnight
RETURN survives as a FEATURE); short / L-S leg (squeeze + borrow tax).

Run:  python3 "CURRENT BEST/mr_layer.py"      # builds features, runs the honest walk-forward, prints net+gross SR
"""
import warnings, os, sys
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))               # repo root
sys.path.insert(0, _ROOT); sys.path.insert(0, os.path.join(_ROOT, "research"))
# The settled feature set is selected by these flags (see docstring). Set BEFORE importing the research harness.
os.environ.setdefault("RESID_FEAT", "1"); os.environ.setdefault("ENGFEAT", "1"); os.environ.setdefault("LONGFEAT", "1")
import numpy as np, pandas as pd
import mr_research as R                                                            # the tested harness — single source of truth
from DATAHUB import DataHub

# ── settled hyper-parameters (do not tune without end-to-end evidence in MR_research.md) ──────────────────────
CLIP   = -0.05                                                                     # asymmetric knife-clip on the sector-residual target
N      = 150                                                                       # top-N long book (breadth, equal-weight)
HOLD   = 10                                                                        # ~10-day hold
SMOOTH = 3                                                                         # EWMA-smooth the alpha (turnover)
VETO_IDIO, VETO_VOL = 0.2, 0.8                                                     # long-leg knife veto (idio-drop pct<0.2 + vol pct>0.8)
LOSS   = "reg:squarederror"                                                        # on the CLIPPED target (== L1 on the unclipped residual)
HZ     = 5                                                                         # forward reversion horizon (days)


class MrLayer:
    """Settled MR sleeve. `.build()` prepares features+target and the walk-forward signal function; `.backtest()`
    runs the honest engine (lag=1, tiered cost+borrow) and returns the BACKTEST result dict (net returns, SR, DD)."""
    def __init__(self, hub: DataHub | None = None, start: str = "2000-01-01", tier: str = "liquid",
                 ustart: str = "2004-01-01"):
        self.hub = hub or DataHub(start=start, min_days=0)
        self.tier, self.ustart = tier, ustart

    def build(self):
        h = self.hub
        self.names, self.days = R._universe(h, self.tier, self.ustart)
        F = R.features(h, self.names, self.days)                                   # RESID+ENG+LONG via env flags above
        self.T = R.prep(h, F, self.names, self.days, self.tier)
        self.secs = R.sector_of(h, self.names)
        Ysec = R.resid_fwd(h, self.names, self.T["GIDX"], H=HZ, kind="sector")     # DLS idiosyncratic reversion target
        self.Ytarget = np.clip(Ysec, CLIP, 1e9)                                    # asymmetric knife-clip
        self.signal_fn = R.make_resid_signal(
            self.T, self.Ytarget, self.secs, N=N, hold=HOLD, test_len=252, sector_neutral=True,
            event_only=True, long_only=True, smooth=SMOOTH, veto_idio=VETO_IDIO, veto_vol=VETO_VOL, loss=LOSS)
        return self

    def backtest(self, train: int = 1512, test: int = 252, cost: bool = True, tag: str = "MR champion"):
        return R.run(self.hub, self.T, self.signal_fn, tag, train=train, test=test, cost=cost)


if __name__ == "__main__":
    L = MrLayer().build()
    print(f"[MR] settled champion · tier={L.tier} · {L.T['D']}d × {L.T['N']} names · features={L.T['Farr'].shape[2]}"
          f" · target=sector-residual reversion clipped@{CLIP} · N={N} hold={HOLD} smooth={SMOOTH} veto=({VETO_IDIO},{VETO_VOL})",
          flush=True)
    print(f"\n{'arm':28}{'SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>9}", flush=True)
    L.backtest(cost=False, tag="  gross")
    L.backtest(cost=True,  tag="  net (deployable)")
