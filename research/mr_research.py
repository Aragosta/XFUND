#!/usr/bin/env python3
"""
mr_research.py — MEAN-REVERSION harness. FAITHFUL implementation of Quantitativo "Long & Short Mean Reversion
Machine" (quantitativo.com/p/long-and-short-mean-reversion-machine) as the BASIC FRAMEWORK. ONE script, overwritten.

THE ARTICLE (the framework we implement — NOT the DM/MOM cross-sectional-ranker reframe):
  1. EVENT TRIGGER      3-day QP (own-history rarity of the 3-day move) < 15  → oversold long candidate.
                        Mirror: QP > 85 → overbought short candidate.
  2. ML = a FILTER      binary classifier P(bounce within H days). Enter a triggered name ONLY iff P > thr (~0.60).
                        Separate long model (P up-bounce after a drop) and short model (P down-fade after a pop).
                        Same feature set + training process both sides ("same features, same training").
  3. EVENT POSITION SIM enter on trigger+filter; EXIT on the FIRST of {close > yesterday's high (reversion touched),
                        6-bar time limit, −5% stop from entry}. One position per name at a time. ≤20 long + ≤20
                        short concurrent, prioritized by ML probability. → turnover-efficient (few names fire/day).
  4. SIZING             long book gross ~1.1× in bull / ~0.1× in bear; short book ~0.2× both regimes (shorts squeeze).
                        Names equal-weighted within a book. This is ~net-long BY DESIGN (article's SR carries beta).
  5. REGIME GATE        article uses VIX > 15d-SMA×1.15 (90th pct) = bear → cut long. We take this "from the STATE
                        layer" (user instruction): bear = STATE `surprise` (Mahalanobis turbulence) above its
                        PIT expanding 90th percentile → long scale 1.1→0.1. (GATE=REGIME, not stops.)

★ ENGINE: everything scored through BACKTEST.walk_forward (rolling train→test OOS) + BACKTEST.backtest (lag=1 ⇒
  trade next close / earn t+2; tiered per-name cost + borrow; honest squeeze/delist PnL grid). The signal_fn trains
  the two classifiers on the TRAIN block and runs the EVENT SIM over the TEST block to emit a daily weight panel.
  Reversion is a ~0-corr DIVERSIFIER — judge net-SR AND corr to MOM/DM. Honest prior ~0.2–0.4 net; Quantitativo's
  1.1–1.55 is a WITH-2020 / 2bp / net-long CEILING.
"""
import warnings, os, sys
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, "/Users/enzokreeft/XFUND")
import numpy as np, pandas as pd
from scipy.stats import norm
from xgboost import XGBClassifier, XGBRegressor
from DATAHUB import DataHub
import BACKTEST

CLF = dict(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           tree_method="hist", n_jobs=4, eval_metric="logloss")
STATE_FEAT = "/tmp/state_feat.parquet"                                # STATE layer read-out (monthly); ffill→daily

# ── DATA ────────────────────────────────────────────────────────────────────────────────────────────────────
def build(tier="liquid", start="2000-01-01"):
    hub = DataHub(start=start, min_days=0)
    if hub.px_d is None: raise RuntimeError("MR needs DAILY data (DataHub daily-native mode)")
    return hub, tier

def _grank_panel(F):                                                 # cross-sectional Gaussian rank per day (row-wise)
    R = F.rank(axis=1, pct=True); return pd.DataFrame(norm.ppf(R.clip(1e-4, 1-1e-4).values), index=F.index, columns=F.columns)

def _rsi(px, n):
    d = px.diff(); up = d.clip(lower=0).rolling(n).mean(); dn = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100/(1 + up/(dn + 1e-9))

def sector_of(hub, names):                                          # per-name sector label (for DLS residualization)
    return hub.sector.reindex(names).fillna("NA") if getattr(hub, "sector", None) is not None else pd.Series("NA", index=names)

def _sec_resid(df, secs):                                           # strip the within-day sector mean (idiosyncratic)
    return df - df.T.groupby(secs.values).transform("mean").T

def _universe(hub, tier, start, mincov=0.15):
    days = hub.px_d.index[hub.px_d.index >= pd.Timestamp(start)]
    elig = hub.elig(tier, "daily").reindex(days)
    names = elig.columns[(elig.mean(axis=0) >= mincov).values]
    return names, days

def qp_raw(px, k=3, win=252):
    """QP indicator: own-history rarity (0–100) of the k-day move — trailing-z → Gaussian CDF. Low = rare drop."""
    r = px/px.shift(k) - 1
    z = (r - r.rolling(win, min_periods=win//2).mean())/(r.rolling(win, min_periods=win//2).std() + 1e-9)
    return pd.DataFrame(norm.cdf(z.values)*100, index=px.index, columns=px.columns)

def statarb_state(hub, names, days, K=10, W=252, M=60, stride=5, denoise=True, Kcap=15):
    """★ FUSE the time-series stat-arb VIEW into the cross-sectional XGBoost book (unify the two MR engines).
    Per name, per day: strip PCA factors (MP-denoised) over trailing W days → residual returns → OU on the recent M →
    three PIT states: `sscore` (how stretched from equilibrium NOW = the AL dislocation), `skappa` (reversion speed =
    quality/tradeability), `sresmom` (cumulative-residual level = idiosyncratic residual-momentum). Computed on a stride
    grid over FULL return history (trailing window only ⇒ PIT-safe) and forward-filled; cached to /tmp."""
    import pickle
    cache = "/tmp/statarb_state.pkl"
    if os.path.exists(cache) and not os.environ.get("STATARB_RECOMPUTE"):
        d = pickle.load(open(cache, "rb"))
        if d["sscore"].shape == (len(days), len(names)) and list(d["sscore"].columns) == list(names):
            return {k: d[k] for k in ("sscore", "skappa", "sresmom")}
    Rfull = hub.ret_d.reindex(columns=names); idx = Rfull.index; Rv = Rfull.values.astype(np.float32); N = len(names)
    pos = idx.get_indexer(pd.DatetimeIndex(days))
    S = np.full((len(days), N), np.nan, np.float32); KA = np.full_like(S, np.nan); RM = np.full_like(S, np.nan)
    last = None
    for i, p in enumerate(pos):
        if not ((i % stride == 0) and p >= W):
            if last is not None: S[i], KA[i], RM[i] = S[last], KA[last], RM[last]   # ffill between stride points
            continue
        Rw = Rv[p - W + 1:p + 1]; cols = np.where(np.isfinite(Rw).all(0))[0]
        if len(cols) < 50:
            if last is not None: S[i], KA[i], RM[i] = S[last], KA[last], RM[last]
            continue
        Z0 = Rw[:, cols]; sd = Z0.std(0) + 1e-9; Z = (Z0 - Z0.mean(0)) / sd
        U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
        if denoise:
            ev = (Sv ** 2) / Z.shape[0]; lam = (1.0 + np.sqrt(len(cols) / Z.shape[0])) ** 2
            kk = int(np.clip(int((ev > lam).sum()), 1, Kcap))
        else: kk = K
        resid = (Z - (U[:, :kk] * Sv[:kk]) @ Vt[:kk]) * sd; Rm = resid[-M:]
        s, kappa, ok = _sscore(Rm); rm = np.cumsum(Rm, axis=0)[-1]
        S[i, cols] = np.where(ok, s, np.nan); KA[i, cols] = np.where(ok, kappa, np.nan); RM[i, cols] = rm
        last = i
    mk = lambda A: pd.DataFrame(A, index=pd.DatetimeIndex(days), columns=names)
    out = {"sscore": mk(S), "skappa": mk(KA), "sresmom": mk(RM)}
    try: pickle.dump(out, open(cache, "wb"))
    except Exception: pass
    return out

def features(hub, names, days, win=252):
    """Reversion feature set → dict name→daily panel, cross-sec Gaussian-ranked, known at close[t]."""
    px = hub.px_d.loc[:, names]; adv = hub.adv_d.loc[:, names]
    hi = hub.high_d.loc[:, names] if hub.high_d is not None else None
    lo = hub.low_d.loc[:, names] if hub.low_d is not None else None
    F = {}
    for k in (3, 5, 10): F[f"qp{k}"] = qp_raw(px, k, win).loc[:, names]           # rarity at multiple windows
    for w in (21, 63): F[f"rvol{w}"] = hub.rvol(w, "gk", "daily").loc[:, names]   # per-name rvol (Novy-Marx speed)
    F["turn_ts"] = adv/(adv.rolling(win, min_periods=win//2).mean() + 1e-9); F["turn_cs"] = adv
    for k in (5, 21, 63, 126, 252): F[f"roc{k}"] = px/px.shift(k) - 1
    F["d200"] = px/hub.sma_d(200).loc[:, names] - 1                              # dist-to-200SMA (article FEATURE)
    for n in (2, 14): F[f"rsi{n}"] = _rsi(px, n)
    if hi is not None:
        rng = (hi - lo); pc = px.shift()
        F["ibs"] = ((px - lo)/(rng + 1e-9)).clip(0, 1)
        tr = np.maximum(np.maximum(rng.values, (hi - pc).abs().values), (lo - pc).abs().values)
        F["natr"] = pd.DataFrame(tr, index=px.index, columns=names).rolling(14).mean()/px
    if os.environ.get("RESID_FEAT"):                                             # DLS: the IDIOSYNCRATIC recent move
        secs = sector_of(hub, names)                                             # (strip sector from the move features)
        F["qp3_sr"]  = _sec_resid(qp_raw(px, 3, win).loc[:, names], secs)        # sector-residual rarity
        F["roc5_sr"] = _sec_resid(px/px.shift(5) - 1, secs)                      # sector-residual 5-day move
        F["roc21_sr"] = _sec_resid(px/px.shift(21) - 1, secs)                    # sector-residual 21-day move
    if os.environ.get("STATARB_FEAT"):                                           # ★ FUSE: OU stat-arb state as cross-sec features
        for k, v in statarb_state(hub, names, days).items(): F[k] = v            # sscore + skappa + sresmom (time-series view)
    if os.environ.get("ENGFEAT") and hi is not None:                            # ★ ENGINEERED: DLS info-vs-liquidity separators (OHLC + volume)
        secs = sector_of(hub, names); ret = hub.ret_d.loc[:, names]; o = hub.open_d.loc[:, names]; pc = px.shift(1)
        on = np.log((o / pc).clip(0.5, 2.0)); idd = np.log((px / o).clip(0.5, 2.0))       # overnight gap vs intraday move
        F["on3"]  = _sec_resid(on.rolling(3, min_periods=2).sum(), secs)         # OVERNIGHT part of the recent drop = news/information (KNIFE)
        F["id3"]  = _sec_resid(idd.rolling(3, min_periods=2).sum(), secs)        # INTRADAY part = order-flow/liquidity (BOUNCE)
        tot3 = (on + idd).rolling(3, min_periods=2).sum()
        F["gapfrac"] = on.rolling(3, min_periods=2).sum() / (tot3.abs() + 1e-3)  # how much of the move was overnight (gap = info = knife)
        F["amihud"] = (ret.abs() / (adv + 1e-9)).rolling(21, min_periods=5).mean()   # Amihud illiquidity (proven reversal predictor, Nagel)
        F["hlspread"] = ((hi - lo) / ((hi + lo) / 2 + 1e-9)).rolling(5, min_periods=2).mean()  # high-low spread proxy
        F["downsemi"] = ((ret.clip(upper=0)) ** 2).rolling(21, min_periods=5).mean() ** 0.5    # downside deviation (knife proximity, finite)
        F["mindrop21"] = ret.rolling(21, min_periods=5).min()                   # worst single-day drop (jump/knife detector)
        F["gapdn"] = (on < -0.03).rolling(10, min_periods=3).sum()              # count of overnight gap-downs (news events = knives)
    if os.environ.get("LONGFEAT"):                                              # ★ LONG-SIDE: discriminate WHICH oversold names bounce MOST
        ret = hub.ret_d.loc[:, names]
        ma20 = px.rolling(20, min_periods=10).mean(); sd20 = px.rolling(20, min_periods=10).std()
        F["bbz20"] = (px - ma20) / (sd20 + 1e-9)                                # Bollinger z: how stretched below the short mean (room to revert)
        F["stretch10"] = px / px.rolling(10, min_periods=5).mean() - 1          # short-term stretch (mean-reversion distance)
        F["acf1"] = ret.rolling(63, min_periods=20).corr(ret.shift(1))          # lag-1 autocorr (NEGATIVE = this name mean-reverts)
        r5 = px / px.shift(5) - 1
        F["vr5"] = r5.rolling(63, min_periods=20).var() / (5 * ret.rolling(63, min_periods=20).var() + 1e-12)  # variance ratio <1 = reverting
        F["voldecel"] = adv.rolling(3, min_periods=2).mean() / (adv.rolling(21, min_periods=5).mean() + 1e-9)  # volume exhaustion (seller done → bounce)
        F["cs_spread"] = ((hi - lo) / px).rolling(21, min_periods=5).mean() if hi is not None else px * 0      # illiquidity/spread (bigger LP premium)
    if os.environ.get("NEWFEAT"):                                                # ★ NEW candidates (not in rejected set): market-axis idiosyncrasy + path shape
        ret = hub.ret_d.loc[:, names]; mkt = hub.ret_d.mean(axis=1)               # equal-weight market proxy
        F["mktresid5"] = (px / px.shift(5) - 1).sub(mkt.rolling(5, min_periods=3).sum(), axis=0)  # MARKET-residual 5d move (complements sector-resid)
        F["dropbeta"]  = ret.rolling(63, min_periods=20).corr(mkt)                # co-movement: high = systematic drop (liquidity→bounce); low = idiosyncratic (knife)
        F["choppy10"]  = (np.sign(ret).diff().abs() > 0).rolling(10, min_periods=5).sum()  # sign-flip count: choppy vs clean waterfall
    return {k: _grank_panel(v.reindex(days)) for k, v in F.items()}

# ── tensorize + trigger/exit panels (all known at close[t]; exits are causal price rules) ──────────────────────
def prep(hub, F, names, days, tier="liquid", qwin=252):
    GIDX = F["qp3"].index; Fn = list(F.keys())
    Farr = np.stack([F[k].values.astype(np.float32) for k in Fn], axis=2)        # D×N×Kf
    elig = hub.elig(tier, "daily").reindex(index=GIDX, columns=names).fillna(False).values
    sh   = hub.elig("shortable", "daily").reindex(index=GIDX, columns=names).fillna(False).values
    E = elig & np.isfinite(Farr).all(2)
    px  = hub.px_d.reindex(index=GIDX, columns=names)
    hi  = hub.high_d.reindex(index=GIDX, columns=names)
    QP3 = qp_raw(hub.px_d.loc[:, names], 3, qwin).reindex(GIDX)                   # raw QP for the event trigger
    brk = (px.values > hi.shift(1).values)                                        # close > yesterday's high → exit
    rv  = hub.rvol(21, "gk", "daily").reindex(index=GIDX, columns=names)          # per-asset realized vol
    RVP = rv.rank(axis=1, pct=True).fillna(0.0).values                            # cross-sec vol percentile (0..1)
    idr = _sec_resid(px / px.shift(3) - 1, sector_of(hub, names))                 # idiosyncratic part of the 3d drop
    IDP = idr.rank(axis=1, pct=True).fillna(0.5).values                           # low pct = most idiosyncratic crash (knife)
    return dict(GIDX=GIDX, names=list(names), Farr=Farr, E=E, sh=sh, QP=QP3.values,
                PX=px.values, BRK=brk, RVP=RVP, IDP=IDP, D=len(GIDX), N=len(names))

def labels(hub, names, GIDX, H=5):
    """Forward H-day close-to-close return per name (bounce/fade label source)."""
    px = hub.px_d.loc[:, names]; fwd = px.shift(-H)/px - 1
    return fwd.reindex(index=GIDX, columns=names).values

def resid_fwd(hub, names, GIDX, H=5, kind="sector"):
    """IDIOSYNCRATIC forward reversion target (Da-Liu-Schaumburg): raw fwd H-day return minus the sector (or market)
    mean of that day → the non-fundamental component, the part that actually reverts."""
    px = hub.px_d.loc[:, names]; fwd = (px.shift(-H)/px - 1).reindex(index=GIDX, columns=names)
    if kind == "xsec": return fwd.sub(fwd.mean(axis=1), axis=0).values
    return _sec_resid(fwd, sector_of(hub, names)).values

def leg_moves(hub, names, GIDX, k=3):
    """Decompose the recent k-day move into its OVERNIGHT (open/prev-close), INTRADAY (close/open) and CLOSE-CLOSE
    legs (log-returns, k-day cumulative). The literature (Liu-Liu-Wang-Zhou-Zhu 'Overnight-Intraday Reversal
    Everywhere'; Brogaard-Han-Kim 2024; Baltussen-Da-Soebhag 'End-of-Day Reversal') finds the OVERNIGHT leg reverts
    hardest — noise-trader/order-imbalance driven, a cleaner reversal signal than close-to-close. Uses hub OHLC."""
    c = hub.px_d.loc[:, names]; o = hub.open_d.loc[:, names]; pc = c.shift(1)
    on = np.log((o / pc).clip(0.5, 2.0)); idr = np.log((c / o).clip(0.5, 2.0)); cc = np.log((c / pc).clip(0.5, 2.0))
    def ks(x): return x.rolling(k).sum().reindex(index=GIDX, columns=names)
    return dict(ON=ks(on), ID=ks(idr), CC=ks(cc))

def make_direct_signal(T, SCORE, secs, N=150, hold=10, test_len=252, long_only=False, sector_neutral=True):
    """No-ML direct reversal book: rank a PRECOMPUTED score panel (higher = more long), long top-N / short bottom-N,
    hold `hold` days. Isolates the SIGNAL (which leg reverts) with no learned target/features confounding it."""
    GIDX, E, sh, names, D = T["GIDX"], T["E"], T["sh"], T["names"], T["D"]
    secv = pd.Series(secs.values, index=names)
    def signal_fn(train_px):
        te = GIDX.get_loc(train_px.index[-1]); first, last = te + 1, min(te + 1 + test_len, D)
        rebal = list(range(first, last, hold))
        sc = np.full((len(rebal), len(names)), np.nan, np.float32)
        for i, p in enumerate(rebal):
            m = E[p] & np.isfinite(SCORE[p])
            if m.sum() < 20: continue
            sc[i, m] = SCORE[p][m]
        ridx = GIDX[rebal]; S = pd.DataFrame(sc, index=ridx, columns=names)
        if sector_neutral: S = _sec_resid(S, secv)
        shT = pd.DataFrame(sh[rebal], index=ridx, columns=names)
        return _topn_hold(S, shT, N, short_frac=1.0, long_only=long_only, idx=GIDX[first:last], names=names)
    return signal_fn

def _topn_hold(S, shT, N, short_frac, long_only, idx, names, conv=False):
    """S = scores on REBALANCE days (rebal×names). Top-N long / bottom-N borrowable short, forward-filled to the daily
    index `idx` (piecewise-constant = genuine H-day hold). conv=True → CONVICTION-weight by rank within the top-N (top
    name gets weight N, Nth gets 1 → utilizes the ORDERING, not just membership); else equal-weight (breadth)."""
    Lr = S.rank(axis=1, ascending=False); sel = Lr <= N
    if conv: L = (N - Lr + 1).where(sel, 0.0).clip(lower=0)                        # linear conviction by rank
    else:    L = sel.astype(float)
    L = L.div(L.sum(1).replace(0, np.nan), axis=0)
    if long_only: B = L
    else:
        Br = S.rank(axis=1, ascending=True); ssel = (Br <= N) & shT
        Sh = ((N - Br + 1).where(ssel, 0.0).clip(lower=0) if conv else ssel.astype(float))
        Sh = Sh.div(Sh.sum(1).replace(0, np.nan), axis=0) * short_frac; B = L.sub(Sh, fill_value=0.0)
    return B.reindex(idx).ffill().fillna(0.0)

def state_regime(GIDX, long_bull=1.1, long_bear=0.1, pctl=0.90, col="surprise"):
    """LONG-book gross scaler from the STATE layer (replaces the article's VIX gate). bear = STATE `surprise`
    above its PIT expanding `pctl` percentile → long_bear, else long_bull. Monthly STATE read-out ffilled→daily."""
    try:
        s = pd.read_parquet(STATE_FEAT)[col]
    except Exception:
        print(f"  [warn] {STATE_FEAT} missing — run STATE.py; regime OFF (long_bull flat)", flush=True)
        return pd.Series(long_bull, index=GIDX)
    s = s.reindex(GIDX, method="ffill")
    thr = s.expanding(min_periods=252).quantile(pctl)                             # PIT 90th-pct, no look-ahead
    bear = (s > thr).fillna(False)
    return pd.Series(np.where(bear, long_bear, long_bull), index=GIDX)

def gross_gate(GIDX, col="surprise", on=1.0, off=0.2, pctl=0.90):
    """Symmetric gross scaler for a dollar-neutral book: cut gross to `off` when the regime driver `col` (STATE
    `surprise` turbulence, or `vix` from DATAHUB via the STATE feat store) exceeds its PIT expanding `pctl`. The DD
    lever (Nagel: liquidity-provision premium is riskiest in stress). Proof STATE `surprise` vs raw `vix` here."""
    return state_regime(GIDX, long_bull=on, long_bear=off, pctl=pctl, col=col)

# ── EVENT POSITION SIM — the article's engine (trigger→filter→hold→event/time/stop exit), ≤N per side by P ─────
def event_book(T, Plong, Pshort, first, last, thr=0.60, N=20, Hmax=6, stop=0.05, long_scale=None, short_frac=0.2,
               vol_min=0.0, vol_sort=False):
    """Deterministic causal sim over test positions [first,last). Plong/Pshort: (D×N) probability panels (NaN off
    trigger). Returns daily weight panel (test_days × names): equal-weight ≤N longs (gross long_scale[d]) minus
    equal-weight ≤N borrowable shorts (gross short_frac). Held name t..exit; exit = close>prev-high | Hmax | stop.
    vol_min: only enter names with cross-sec realized-vol percentile ≥ vol_min (article prioritizes high-vol names).
    vol_sort: prioritize entries by realized vol desc instead of by P (Novy-Marx: vol→faster/larger reversal)."""
    PX, BRK, E, sh, RVP, names, GIDX = T["PX"], T["BRK"], T["E"], T["sh"], T["RVP"], T["names"], T["GIDX"]
    tp = list(range(first, last)); Wl = np.zeros((len(tp), T["N"])); Ws = np.zeros((len(tp), T["N"]))
    heldL, heldS = {}, {}                                                         # name_ix -> (entry_pos, entry_px)
    for i, p in enumerate(tp):
        for held, sign in ((heldL, +1), (heldS, -1)):                            # ---- exits first (causal) ----
            for j in list(held):
                e_pos, e_px = held[j]; cp = PX[p, j]
                ret = (cp/e_px - 1) if np.isfinite(cp) and e_px else 0.0
                hit_stop = (sign*ret <= -stop)                                    # long: down -stop; short: up +stop
                if BRK[p, j] or (p - e_pos >= Hmax) or hit_stop or not np.isfinite(cp): held.pop(j)
        for held, P, borrow in ((heldL, Plong, False), (heldS, Pshort, True)):    # ---- entries by P (or vol) desc ----
            if len(held) >= N: continue
            row = P[p]; vr = RVP[p]
            cand = np.where(np.isfinite(row) & (row > thr) & E[p] & (vr >= vol_min))[0]  # high-vol subset only
            cand = [j for j in cand if j not in held and (sh[p, j] if borrow else True)]
            key = (lambda j: -vr[j]) if vol_sort else (lambda j: -row[j])         # prioritize high-vol or high-P
            for j in sorted(cand, key=key)[:N - len(held)]:
                if np.isfinite(PX[p, j]): held[j] = (p, PX[p, j])
        if heldL: Wl[i, list(heldL)] = 1.0 / len(heldL)                           # equal-weight within book
        if heldS: Ws[i, list(heldS)] = 1.0 / len(heldS)
    ls = (long_scale.values if long_scale is not None else np.ones(len(GIDX)))[tp][:, None]
    W = Wl * ls - Ws * short_frac
    return pd.DataFrame(W, index=GIDX[first:last], columns=names)

# ── SIGNAL FACTORIES → signal_fn(train_px) for BACKTEST.walk_forward ──────────────────────────────────────────
def _train_side(T, YB, side, tr):
    """Train one binary classifier on TRIGGERED events in the train positions `tr`. side=+1 long (trigger QP<15,
    label bounce = fwd>0); side=-1 short (trigger QP>85, label fade = fwd<0)."""
    Farr, E, QP = T["Farr"], T["E"], T["QP"]; Xl, Yl = [], []
    for p in tr:
        trig = (QP[p] < 15) if side > 0 else (QP[p] > 85)
        m = E[p] & trig & np.isfinite(YB[p])
        if m.sum() < 10: continue
        Xl.append(Farr[p][m]); Yl.append(((YB[p][m] > 0) if side > 0 else (YB[p][m] < 0)).astype(int))
    if not Xl: return None
    X = np.vstack(Xl); Y = np.concatenate(Yl)
    if Y.min() == Y.max(): return None
    return XGBClassifier(**CLF, random_state=0).fit(X, Y)

def make_article_signal(T, YB, long_scale, thr=0.60, N=20, Hmax=6, stop=0.05, short_frac=0.2, test_len=252,
                        stride=2, maxtrain=3800, use_filter=True, vol_min=0.0, vol_sort=False):
    """Faithful article: QP-event trigger + binary ML filter (P>thr) + event sim. use_filter=False → pure QP rule.
    vol_min/vol_sort → per-asset realized-vol prioritization (article prefers high-vol names)."""
    GIDX, D, QP, E, sh = T["GIDX"], T["D"], T["QP"], T["E"], T["sh"]
    ev = lambda Pl, Ps, f, l, th: event_book(T, Pl, Ps, f, l, thr=th, N=N, Hmax=Hmax, stop=stop,
                                             long_scale=long_scale, short_frac=short_frac, vol_min=vol_min, vol_sort=vol_sort)
    def signal_fn(train_px):
        te = GIDX.get_loc(train_px.index[-1]); tl = len(train_px); ts = max(0, te - tl + 1)
        tr = [p for p in range(ts, te - 6) if p % stride == 0][-(maxtrain // stride):]     # embargo the H horizon
        first, last = te + 1, min(te + 1 + test_len, D)
        if first >= last: return pd.DataFrame(columns=T["names"])
        if use_filter:
            mL = _train_side(T, YB, +1, tr); mS = _train_side(T, YB, -1, tr)
            Plong  = np.full((D, T["N"]), np.nan, np.float32); Pshort = np.full((D, T["N"]), np.nan, np.float32)
            for p in range(first, last):
                mask = E[p] & (QP[p] < 15)
                if mL is not None and mask.any(): Plong[p, mask] = mL.predict_proba(T["Farr"][p][mask])[:, 1]
                maskS = E[p] & (QP[p] > 85) & sh[p]
                if mS is not None and maskS.any(): Pshort[p, maskS] = mS.predict_proba(T["Farr"][p][maskS])[:, 1]
        else:                                                                     # pure rule: rarity IS the score
            Plong  = np.where((QP < 15), (15 - QP)/15.0, np.nan).astype(np.float32)   # in (0,1], bigger=rarer drop
            Pshort = np.where((QP > 85), (QP - 85)/15.0, np.nan).astype(np.float32)
            return ev(Plong, Pshort, first, last, -1.0)                            # rule has no prob threshold
        return ev(Plong, Pshort, first, last, thr)
    return signal_fn

# ── RESIDUAL-TARGET BOOK — the T5 lead: XGBRegressor on the IDIOSYNCRATIC fwd reversion, SECTOR-NEUTRAL construction ──
def make_resid_signal(T, Ytarget, secs, N=150, hold=10, test_len=252, stride=2, maxtrain=3800,
                      sector_neutral=True, long_only=False, gate=None, event_only=True, vol_min=0.0, smooth=1,
                      veto_idio=0.0, veto_vol=1.0, loss="reg:squarederror", conv=False):
    """Train XGBRegressor on the residual reversion target over the OVERSOLD subset (QP<15), predict, rank, long top-N /
    short bottom-N. event_only=True → rank ONLY within the QP<15 event (long-only makes sense: long the best-predicted
    reverters); False → rank the FULL eligible universe (Avellaneda-Lee stat-arb L/S: long high-resid / short low-resid).
    sector_neutral → demean scores within sector/day (trade the idiosyncratic axis). gate = daily gross scaler."""
    GIDX, Farr, E, QP, sh, names, D = T["GIDX"], T["Farr"], T["E"], T["QP"], T["sh"], T["names"], T["D"]
    secv = pd.Series(secs.values, index=names)
    def signal_fn(train_px):
        te = GIDX.get_loc(train_px.index[-1]); tl = len(train_px); ts = max(0, te - tl + 1)
        tr = [p for p in range(ts, te - 6) if p % stride == 0][-(maxtrain // stride):]
        Xl, Yl = [], []
        for p in tr:
            m = (E[p] & (QP[p] < 15) if event_only else E[p]) & np.isfinite(Ytarget[p])
            if m.sum() < 10: continue
            Xl.append(Farr[p][m]); Yl.append(Ytarget[p][m])
        first, last = te + 1, min(te + 1 + test_len, D)
        if not Xl or first >= last: return pd.DataFrame(columns=names)
        mdl = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                           tree_method="hist", n_jobs=4, random_state=0, objective=loss).fit(np.vstack(Xl), np.concatenate(Yl))
        RVP = T["RVP"]; IDP = T["IDP"]; rebal = list(range(first, last, hold))
        sc = np.full((len(rebal), len(names)), np.nan, np.float32)
        for i, p in enumerate(rebal):
            m = E[p] & (QP[p] < 15) if event_only else E[p]                       # rank event-subset or full universe
            m = m & (RVP[p] >= vol_min)                                           # vol tilt: trade where reversion is BIGGEST
            m = m & (IDP[p] >= veto_idio) & (RVP[p] <= veto_vol)                  # KNIFE-TAIL VETO: drop most-idiosyncratic crashes + highest-vol
            if m.sum() < 20: continue
            sc[i, m] = mdl.predict(Farr[p][m])
        ridx = GIDX[rebal]; S = pd.DataFrame(sc, index=ridx, columns=names)
        if sector_neutral: S = _sec_resid(S, secv)                               # trade the idiosyncratic axis
        if smooth > 1: S = S.ewm(span=smooth, min_periods=1).mean()              # SIGNAL-LEVEL persistence (EWMA the alpha → stabler ranks → less turnover)
        shT = pd.DataFrame(sh[rebal], index=ridx, columns=names)
        didx = GIDX[first:last]
        W = _topn_hold(S, shT, N, short_frac=1.0, long_only=long_only, idx=didx, names=names, conv=conv)
        if gate is not None: W = W.mul(gate.reindex(didx).values, axis=0)         # STATE/VIX regime scaler
        return W
    return signal_fn

# ── AVELLANEDA-LEE STAT-ARB — PCA factor residual → OU s-score (the SOTA construction; NO forward-return prediction) ──
def _sscore(resid):
    """Vectorized OU s-score per name from a window of residual RETURNS (W×N). Cumulate → AR(1) X_t=a+b·X_{t-1}+ζ →
    κ=−log(b)·252 (reversion speed), m=a/(1−b) (equilibrium), σ_eq=√(var(ζ)/(1−b²)); s=(X_last−m)/σ_eq (Avellaneda-Lee)."""
    X = np.cumsum(resid, axis=0); X0, X1 = X[:-1], X[1:]; n = X0.shape[0]
    m0 = X0.mean(0); m1 = X1.mean(0)
    cov = ((X0 - m0) * (X1 - m1)).mean(0); v0 = ((X0 - m0) ** 2).mean(0) + 1e-12
    b = cov / v0; a = m1 - b * m0
    zvar = ((X1 - (a + b * X0)) ** 2).mean(0) * n / max(n - 2, 1)
    with np.errstate(all="ignore"):
        kappa = -np.log(b) * 252.0; m = a / (1 - b); sigeq = np.sqrt(zvar / (1 - b ** 2))
        s = (X[-1] - m) / sigeq
    ok = (b > 0) & (b < 1) & np.isfinite(s) & (sigeq > 0)
    return s, kappa, ok

def make_statarb_signal(T, R, K=15, W=252, M=60, sbo=1.25, sclose=0.5, kappa_min=8.4, hold=5, test_len=252,
                        weight="sign", sector_neutral=False, secs=None, ADV=None, nliq=None,
                        denoise=False, Kcap=40, qgate="kappa", vr_thr=1.0):
    """PCA-residual OU s-score (Avellaneda-Lee 2010; the residual is the mean-reverting object, NOT a forecast).
    Each rebalance: strip K PCA factors over trailing W days (stable loadings) → residual returns; fit the OU on the
    RECENT M days (AL use ~60d — the residual reverts over weeks; 252d over-smooths) → s-score; enter s<−sbo long /
    s>+sbo short among FAST reverters (κ>kappa_min ⇒ reversion time<~30d); dollar-neutral, hold `hold` days.
    weight: 'sign' (±1 breadth) or 'score' (∝ −s, conviction). K = number of PCA factors stripped, M = OU window.

    ── #1 NOISE CONTROL: RMT/Marchenko-Pastur factor denoising (denoise=True) ──
    Instead of a FIXED K, strip only the eigenvectors ABOVE the MP noise-edge λ+=(1+√(N/T))² — i.e. the genuine common
    factors — and keep the noise bulk in the residual (Laloux-Bouchaud / Lopez de Prado). This is the principled fix for
    T17 (K>10 hurt because those eigenvectors are MP noise): a data-driven per-window factor count that never strips noise.
    ── #2 SIGNAL QUALITY: model-free stationarity gate (qgate) ──
    Beyond the OU κ-speed gate, filter which residuals we TRUST to be genuinely mean-reverting NOW:
      · 'vr'    — variance-ratio VR(5)<vr_thr (reject non-stationary/trending residuals = the knives)
      · 'emrt'  — Ning-Lee empirical mean-reversion time (crossing-run length): keep the FAST-reverting half
      · 'combo' — κ AND VR AND fast-EMRT. Pure signal/noise control, NO turnover machinery."""
    GIDX, E, names, D = T["GIDX"], T["E"], T["names"], T["D"]
    secv = pd.Series(secs.values, index=names) if (sector_neutral and secs is not None) else None
    keffs = []                                                                     # per-window realized factor count (denoise diagnostic)
    def signal_fn(train_px):
        te = GIDX.get_loc(train_px.index[-1]); first, last = te + 1, min(te + 1 + test_len, D)
        rebal = [p for p in range(first, last, hold) if p - W >= 0]
        if not rebal: return pd.DataFrame(columns=names)
        sc = np.full((len(rebal), len(names)), np.nan, np.float32)
        for i, p in enumerate(rebal):
            elig_p = E[p] & np.isfinite(R[p - W + 1:p + 1]).all(0)
            if nliq is not None and ADV is not None:                              # restrict to top-nliq MOST LIQUID (AL/Pelger ~500)
                av = np.where(elig_p, ADV[p], -np.inf)
                if np.isfinite(av).sum() > nliq: elig_p = elig_p & (av >= np.sort(av)[-nliq])
            cols = np.where(elig_p)[0]                                             # names with a full clean window
            if len(cols) < 50: continue
            Rw = R[p - W + 1:p + 1][:, cols]                                       # W×Nc residualization window
            mu = Rw.mean(0); sd = Rw.std(0) + 1e-9; Z = (Rw - mu) / sd            # standardize (correlation PCA)
            U, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
            if denoise:                                                             # #1 MP/RMT: strip only factors above the noise edge
                ev = (Sv ** 2) / Z.shape[0]                                         # eigenvalues of the correlation matrix
                lam_plus = (1.0 + np.sqrt(len(cols) / Z.shape[0])) ** 2            # Marchenko-Pastur upper edge (σ²=1, correlation)
                kk = int(np.clip(int((ev > lam_plus).sum()), 1, Kcap))            # data-driven factor count (never strip noise)
            else:
                kk = K
            keffs.append(kk)
            resid = (Z - (U[:, :kk] * Sv[:kk]) @ Vt[:kk]) * sd                      # strip kk factors → residual returns
            Rm = resid[-M:]                                                         # OU/quality window (AL ~60d)
            s, kappa, ok = _sscore(Rm)                                             # fit OU on the RECENT M days
            ok = ok & (kappa > kappa_min)                                          # trade only FAST-reverting residuals
            if qgate != "kappa":                                                    # #2 model-free signal-quality gate
                X = np.cumsum(Rm, axis=0); Xd = X - X.mean(0)                       # cumulative residual (the OU object)
                if qgate in ("vr", "combo"):                                        # variance ratio VR(5) < thr → stationary/reverting
                    qv = 5; d1 = np.diff(X, axis=0); dq = X[qv:] - X[:-qv]
                    vr = dq.var(0) / (qv * d1.var(0) + 1e-12); ok = ok & (vr < vr_thr)
                if qgate in ("emrt", "combo"):                                      # Ning-Lee empirical reversion time (run-length) → keep fast half
                    sg = np.sign(Xd); cross = (sg[1:] * sg[:-1] < 0).sum(0); runlen = Rm.shape[0] / (cross + 1.0)
                    if ok.any(): ok = ok & (runlen <= np.median(runlen[ok]))
            row = np.full(len(cols), np.nan); row[ok] = s[ok]                       # store RAW s-score
            sc[i, cols] = row
        Smat = sc                                                                  # rebal × N raw s-scores (NaN invalid)
        if weight == "band":                                                       # ── AL entry/exit HYSTERESIS ──
            pos = np.zeros(len(names)); POS = np.zeros_like(Smat)                   # open |s|>sbo, HOLD until |s|<sclose
            for i in range(Smat.shape[0]):
                si = Smat[i]; nanm = ~np.isfinite(si)
                pos[nanm] = 0.0                                                     # name left universe → flat
                pos[(pos > 0) & (si >= -sclose)] = 0.0                             # long reverted up → close
                pos[(pos < 0) & (si <= sclose)] = 0.0                              # short reverted down → close
                pos[(pos == 0) & (si < -sbo)] = 1.0                                # open long on deep neg dislocation
                pos[(pos == 0) & (si > sbo)] = -1.0                               # open short on deep pos dislocation
                POS[i] = pos
            S = pd.DataFrame(POS, index=GIDX[rebal], columns=names)
            L = (S > 0).astype(float); Sh = (S < 0).astype(float)
        else:
            S = pd.DataFrame(Smat, index=GIDX[rebal], columns=names)
            if secv is not None: S = _sec_resid(S, secv)
            if weight == "score": L = (-S).clip(lower=0); Sh = S.clip(lower=0)     # conviction ∝ −s (both legs)
            else: L = (S < -sbo).astype(float); Sh = (S > sbo).astype(float)       # sign: ±1 beyond bands
        L = L.div(L.sum(1).replace(0, np.nan), axis=0); Sh = Sh.div(Sh.sum(1).replace(0, np.nan), axis=0)
        B = L.sub(Sh, fill_value=0.0)
        return B.reindex(GIDX[first:last]).ffill().fillna(0.0)                     # hold between rebalances
    signal_fn.keffs = keffs                                                        # expose realized factor counts (denoise diagnostic)
    return signal_fn

def run(hub, T, signal_fn, tag, train=1512, test=252, lag=1, cost=True, flatbps=None):
    names = T["names"]; px = hub.pnl("daily").loc[T["GIDX"], names]               # honest squeeze/delist PnL grid
    # BUGFIX: the tiered cost/borrow schedules are calibrated on MONTHLY $-volume (every other sleeve feeds hub.mdv).
    # adv_d is the DAILY avg $-volume → feeding it raw understates liquidity ~21× and overcharges cost ~5×. Convert to
    # a monthly-equivalent ($/mo ≈ daily-avg × ~21 trading days) so a $50M/day name lands in the 5bp tier, not 25bp.
    dv = hub.adv_d.loc[T["GIDX"], names] * 21.0
    if flatbps is not None:                                                        # replicate article: flat X bp, no borrow tier
        tc = flatbps/1e4; bf = 0.0
    else:
        tc = BACKTEST.tiered_transaction_costs(dv) if cost else 0.0
        bf = BACKTEST.tiered_borrow_fees(dv) if cost else 0.0
    r = BACKTEST.walk_forward(lambda tp: signal_fn(tp), px, train=train, test=test, freq=252, lag=lag,
                              transaction_cost=tc, borrow_fee=bf)
    print(f"  {tag:32}{r['sharpe']:>8.2f}{r['ann_return']:>9.1%}{r['max_drawdown']:>9.1%}{r['ann_turnover']:>9.1%}", flush=True)
    return r


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT — faithful article build. Arms: pure QP-event rule → +ML filter → +STATE regime gate.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    tier = os.environ.get("TIER", "liquid"); start = os.environ.get("START", "2004-01-01")
    HZ = int(os.environ.get("HZ", 5)); N = int(os.environ.get("N", 20)); THR = float(os.environ.get("THR", 0.60))
    HMAX = int(os.environ.get("HMAX", 6)); STOP = float(os.environ.get("STOP", 0.05)); SFRAC = float(os.environ.get("SFRAC", 0.2))
    TEST = int(os.environ.get("TEST", 252)); TRAIN = int(os.environ.get("TRAIN", 1512))
    hub, tier = build(tier=tier)
    names, days = _universe(hub, tier, start)
    F = features(hub, names, days); T = prep(hub, F, names, days, tier)
    YB = labels(hub, names, T["GIDX"], H=HZ)
    reg  = state_regime(T["GIDX"], long_bull=1.1, long_bear=0.1)                  # STATE gate (article VIX analog)
    flat = pd.Series(1.1, index=T["GIDX"])                                        # no-gate control (long 1.1 always)
    print(f"[MR] ARTICLE framework · tier={tier} · {T['D']}d × {T['N']} names · OOS BACKTEST(train={TRAIN},test={TEST},lag=1,tiered)"
          f" · thr={THR} N={N} Hmax={HMAX} stop={STOP} short={SFRAC} · Hlabel={HZ}", flush=True)
    print(f"[regime] STATE surprise-gate: bear days = {(reg < 1.0).mean():.1%} (long 1.1→0.1)", flush=True)

    # ── MODE=corr : deployment test — correlation of the deployable MR book with the DM sleeve + combined-book value ──
    if os.environ.get("MODE") == "corr":
        import pickle
        secs = sector_of(hub, T["names"]); Ysec = resid_fwd(hub, T["names"], T["GIDX"], H=HZ, kind="sector")
        RN = int(os.environ.get("RN", 150)); HOLD = int(os.environ.get("HOLD", 10))
        print("\n[MODE=corr] deployable MR book (long-only · residual target · smooth=3 · knife-veto) vs DM sleeve", flush=True)
        print(f"\n{'arm':40}{'SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>9}", flush=True)
        sig = make_resid_signal(T, Ysec, secs, N=RN, hold=HOLD, test_len=TEST, sector_neutral=True, event_only=True,
                                long_only=True, smooth=3, veto_idio=0.2, veto_vol=0.8)
        r = run(hub, T, sig, "MR long-only (deployable)", train=TRAIN, test=TEST)
        mr_d = r["returns"]                                                        # daily net returns
        mr_m = (1 + mr_d).resample("ME").prod() - 1                                # → monthly
        try: dm_m = pickle.load(open("/tmp/dm_returns.pkl", "rb"))
        except Exception: print("  [warn] /tmp/dm_returns.pkl missing — run DM.py first"); sys.exit(0)
        J = pd.concat([mr_m.rename("MR"), dm_m.rename("DM")], axis=1).dropna()
        J = J[(J.index >= "2011-01-01")]                                           # OOS overlap
        pear = J["MR"].corr(J["DM"]); spear = J["MR"].corr(J["DM"], method="spearman")
        def sr(x): return x.mean() / x.std() * np.sqrt(12)
        vmr, vdm = J["MR"].std(), J["DM"].std()                                    # inverse-vol (risk-parity) blend
        blend = (J["MR"] / vmr + J["DM"] / vdm) / 2
        roll = J["MR"].rolling(24).corr(J["DM"])
        print(f"\n  overlap {len(J)} months ({J.index.min():%Y-%m}→{J.index.max():%Y-%m})", flush=True)
        print(f"  CORR(MR,DM):  Pearson {pear:+.3f} · Spearman {spear:+.3f} · rolling-24m range [{roll.min():+.2f},{roll.max():+.2f}]", flush=True)
        print(f"  standalone SR — MR {sr(J['MR']):.2f} · DM {sr(J['DM']):.2f}  →  50/50 risk-parity BLEND SR {sr(blend):.2f}", flush=True)
        print(f"  (a low corr means the blend SR exceeds both standalones = the diversification the sleeve is FOR.)", flush=True)
        sys.exit(0)

    # ── MODE=onx : OVERNIGHT vs INTRADAY vs CLOSE-CLOSE reversal decomposition (uses hub OHLC). MEASURE per-leg
    # reversal IC vs forward residual reversion, then run each leg as a direct reversal book through the honest engine.
    if os.environ.get("MODE") == "onx":
        secs = sector_of(hub, T["names"]); names, GIDX, E = T["names"], T["GIDX"], T["E"]
        KLEG = int(os.environ.get("KLEG", 3)); RN = int(os.environ.get("RN", 150)); HOLD = int(os.environ.get("HOLD", 10))
        LO = bool(os.environ.get("LONGONLY"))
        legs = leg_moves(hub, names, GIDX, k=KLEG)
        Yfwd = resid_fwd(hub, names, GIDX, H=HZ, kind="sector")                     # DLS idiosyncratic fwd reversion target
        print(f"\n[MODE=onx] overnight/intraday reversal decomposition · k={KLEG} · H={HZ} · N={RN} hold={HOLD} · {'LONG-ONLY' if LO else 'L/S'}", flush=True)
        # ── PART A · MEASURE: cross-sectional reversal IC (per-day corr of −leg_move with fwd residual reversion) ──
        print(f"\n  {'leg':6}{'IC(rev,fwd) all':>18}{'IC on QP<15':>14}", flush=True)
        for lab, mv in legs.items():
            R = _sec_resid(mv, pd.Series(secs.values, index=names)).values           # sector-residual leg move
            rev = -R                                                                  # reversal signal = fade the move
            ics, ics_e = [], []
            for p in range(len(GIDX)):
                m = E[p] & np.isfinite(rev[p]) & np.isfinite(Yfwd[p])
                if m.sum() < 30: continue
                a, b = rev[p][m], Yfwd[p][m]
                if a.std() > 0 and b.std() > 0: ics.append(np.corrcoef(a, b)[0, 1])
                me = m & (T["QP"][p] < 15)
                if me.sum() >= 30:
                    a2, b2 = rev[p][me], Yfwd[p][me]
                    if a2.std() > 0 and b2.std() > 0: ics_e.append(np.corrcoef(a2, b2)[0, 1])
            print(f"  {lab:6}{np.mean(ics):>18.4f}{np.mean(ics_e):>14.4f}", flush=True)
        # ── PART B · BOOK: each leg as a direct reversal book through the honest engine (no turnover control) ──
        print(f"\n  {'book':32}{'SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>9}", flush=True)
        books = {}
        for lab, mv in legs.items():
            SCORE = -mv.values.astype(np.float32)                                     # reversal = fade recent leg move
            sig = make_direct_signal(T, SCORE, secs, N=RN, hold=HOLD, test_len=TEST, long_only=LO, sector_neutral=True)
            r = run(hub, T, sig, f"{lab}-reversal", train=TRAIN, test=TEST)
            books[lab] = r["returns"]
        # ── corr to DM (if available) for the best diversifier read ──
        try:
            import pickle; dm_m = pickle.load(open("/tmp/dm_returns.pkl", "rb"))
            print(f"\n  {'leg':6}{'corr(MR,DM) monthly':>22}", flush=True)
            for lab, rd in books.items():
                mr_m = (1 + rd).resample("ME").prod() - 1
                J = pd.concat([mr_m.rename("MR"), dm_m.rename("DM")], axis=1).dropna(); J = J[J.index >= "2011-01-01"]
                if len(J) > 24: print(f"  {lab:6}{J['MR'].corr(J['DM']):>22.3f}", flush=True)
        except Exception as e: print(f"  [corr skipped: {e}]", flush=True)
        sys.exit(0)

    # ── MODE=anatomy : CHARACTERIZE the reversion trades — subdivide oversold events by dislocation type / vol /
    # idiosyncrasy / volume; report payoff (sector-residual fwd), hit-rate, IR-per-trade, and the LEFT-TAIL per bucket. ─
    if os.environ.get("MODE") == "anatomy":
        QP, E, RVP, D, names = T["QP"], T["E"], T["RVP"], T["D"], T["names"]
        secs = sector_of(hub, names)
        fwd_raw = labels(hub, names, T["GIDX"], H=HZ)
        fwd_res = resid_fwd(hub, names, T["GIDX"], H=HZ, kind="sector")            # what a neutral book earns
        px = hub.px_d.reindex(index=T["GIDX"], columns=names)
        drop3 = (px / px.shift(3) - 1).values                                     # the triggering move
        idrop = _sec_resid(px / px.shift(3) - 1, secs).values                     # idiosyncratic part of the drop (DLS)
        adv = hub.adv_d.reindex(index=T["GIDX"], columns=names)
        avol = (adv / adv.rolling(63, min_periods=20).mean()).values              # abnormal volume on the event
        rows = []
        for p in range(252, D - HZ, 3):                                           # sample long events (QP<15)
            m = E[p] & (QP[p] < 15) & np.isfinite(fwd_res[p]) & np.isfinite(RVP[p]) & np.isfinite(idrop[p]) & np.isfinite(avol[p])
            j = np.where(m)[0]
            if len(j) == 0: continue
            rows.append(np.column_stack([fwd_res[p, j], fwd_raw[p, j], RVP[p, j], drop3[p, j], idrop[p, j], avol[p, j]]))
        A = np.vstack(rows); F = A[:, 0]                                           # F = sector-residual fwd payoff (the P&L)
        print(f"\n[MODE=anatomy] {len(A):,} long oversold-events (QP<15) · payoff = sector-residual {HZ}d fwd return (bps)", flush=True)
        print(f"  ALL: hit {np.mean(F>0):.1%} · mean {F.mean()*1e4:+.1f}bp · med {np.median(F)*1e4:+.1f}bp · IR/trade {F.mean()/F.std():+.3f} · p1 {np.percentile(F,1)*1e4:.0f}bp · skew {float(pd.Series(F).skew()):+.2f}", flush=True)
        def table(label, key, asc=True):
            q = pd.qcut(pd.Series(key).rank(method="first"), 5, labels=False)
            print(f"\n  by {label} (Q0→Q4):", flush=True)
            print(f"    {'Q':4}{'n':>9}{'hit':>8}{'mean bp':>10}{'IR/trade':>10}{'left-tail p1 bp':>16}", flush=True)
            order = range(5) if asc else range(4, -1, -1)
            for b in order:
                f = F[q == b]
                if len(f) < 50: continue
                print(f"    Q{b:<3}{len(f):>9,}{np.mean(f>0):>8.1%}{f.mean()*1e4:>10.1f}{f.mean()/f.std():>10.3f}{np.percentile(f,1)*1e4:>16.0f}", flush=True)
        table("MAGNITUDE (3d drop; Q0=deepest)", A[:, 3])                          # deeper drop = bigger reversion or knife?
        table("IDIOSYNCRASY (idio drop; Q0=most idiosyncratic)", A[:, 4])         # DLS: idiosyncratic reverts, systematic doesn't
        table("REALIZED VOL (Q4=highest)", A[:, 2], asc=False)                     # high vol = bigger snap + fatter tail?
        table("ABNORMAL VOLUME (Q4=highest)", A[:, 5], asc=False)                  # high volume = informed knife (skip)?
        print("\n  READ: alpha lives where mean-bp & IR are high; RISK (squeeze/knife) lives where left-tail p1 is deeply negative.", flush=True)
        sys.exit(0)

    # ── MODE=dist : the forward-return DISTRIBUTION of oversold events across daily horizons h=1..7 (shape, not mean) ──
    if os.environ.get("MODE") == "dist":
        QP, E, D, names = T["QP"], T["E"], T["D"], T["names"]
        secs = sector_of(hub, names); px = hub.px_d.reindex(index=T["GIDX"], columns=names)
        kind = os.environ.get("DIST", "raw")                                      # raw cumret vs sector-residual
        print(f"\n[MODE=dist] forward-return distribution of QP<15 oversold events · {kind} cumulative return · h=1..7 days", flush=True)
        print(f"  {'h':3}{'n':>9}{'mean%':>8}{'med%':>8}{'hit':>7}{'IR':>7}{'skew':>7}{'kurt':>7}   {'p1':>7}{'p5':>7}{'p25':>7}{'p75':>7}{'p95':>7}{'p99':>7}", flush=True)
        for h in range(1, 8):
            cr = (px.shift(-h) / px - 1)
            if kind == "resid": cr = _sec_resid(cr, secs)                         # idiosyncratic (what a neutral book earns)
            crv = cr.values; rows = []
            for p in range(252, D - h, 3):
                m = E[p] & (QP[p] < 15) & np.isfinite(crv[p])
                if m.any(): rows.append(crv[p][m])
            f = np.concatenate(rows); s = pd.Series(f)
            pk = np.percentile(f, [1, 5, 25, 75, 95, 99]) * 100
            print(f"  {h:<3}{len(f):>9,}{f.mean()*100:>8.2f}{np.median(f)*100:>8.2f}{np.mean(f>0):>7.1%}"
                  f"{f.mean()/f.std():>7.3f}{s.skew():>7.1f}{s.kurt():>7.0f}   "
                  f"{pk[0]:>7.1f}{pk[1]:>7.1f}{pk[2]:>7.1f}{pk[3]:>7.1f}{pk[4]:>7.1f}{pk[5]:>7.1f}", flush=True)
        print("  (watch: does the CENTER (median/p25/p75) drift UP with h = reversion, while p1/p5 stay deeply negative = knife tail?)", flush=True)
        sys.exit(0)

    # ── MODE=statarb : Avellaneda-Lee PCA-residual OU s-score (SOTA construction; A+B from the Pelger literature) ──
    # Strip K PCA factors → residual returns → OU s-score → trade dislocations. NO forward-return prediction (sidesteps the
    # AUC-0.50 wall). Sweep K (factor count = the residual-quality lever) + sign vs conviction weighting. Gross → net.
    if os.environ.get("MODE") == "statarb":
        R = hub.ret_d.reindex(index=T["GIDX"], columns=T["names"]).values.astype(np.float32)
        secs = sector_of(hub, T["names"]); Kw = int(os.environ.get("K", 10)); WIN = int(os.environ.get("WIN", 252))
        SBO = float(os.environ.get("SBO", 1.25)); HOLD = int(os.environ.get("HOLD", 5)); Mw = int(os.environ.get("M", 60))
        print(f"\n[MODE=statarb] Avellaneda-Lee PCA-residual OU s-score · WIN={WIN} M={Mw} sbo={SBO} recheck={HOLD}d κ_min=8.4", flush=True)
        print(f"\n{'arm':44}{'SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>9}", flush=True)
        ADV = hub.adv_d.reindex(index=T["GIDX"], columns=T["names"]).values
        DN = bool(os.environ.get("DENOISE")); QG = os.environ.get("QGATE", "kappa"); VRT = float(os.environ.get("VRT", 1.0))
        mk = lambda nl, sc, k=Kw, dn=DN, qg=QG: make_statarb_signal(T, R, K=k, W=WIN, M=Mw, sbo=SBO, sclose=sc, hold=HOLD,
                                        test_len=TEST, weight="band", secs=secs, ADV=ADV, nliq=nl, denoise=dn, Kcap=k, qgate=qg, vr_thr=VRT)
        if os.environ.get("KSWEEP"):                                                # "weak factors matter" (Attention Factors ablation): does raising K help gross?
            tail = " +MP-DENOISE (Kcap=K, data-driven count)" if DN else " (fixed K)"
            print(f"-- #1 K-FACTOR SWEEP{tail} — RMT denoising should stop the K>10 decay (strip only real factors) --", flush=True)
            for k in (5, 10, 15, 20, 30, 40):
                sf = mk(None, 0.05, k); r = run(hub, T, sf, f"  K{'cap' if DN else ''}={k} BAND [gross]", train=TRAIN, test=TEST, cost=False)
                if DN and sf.keffs: print(f"       ↳ realized factor count: mean {np.mean(sf.keffs):.1f} (min {min(sf.keffs)}, max {max(sf.keffs)})", flush=True)
            sys.exit(0)
        if os.environ.get("QSWEEP"):                                                # #2 model-free signal-quality gate sweep (which residuals to trust)
            print(f"-- #2 SIGNAL-QUALITY GATE SWEEP (K={Kw}, denoise={DN}, all names, band exit) — does a stationarity filter lift GROSS? --", flush=True)
            for qg in ("kappa", "vr", "emrt", "combo"):
                run(hub, T, mk(None, 0.05, Kw, DN, qg), f"  qgate={qg:6} [gross]", train=TRAIN, test=TEST, cost=False)
            sys.exit(0)
        print("-- LIQUIDITY CONCENTRATION (AL/Pelger ~500 names) × band exit |s|<0.05 — the cost/turnover fix, gross→net --", flush=True)
        for nl in (None, 1000, 500, 300):
            tag = f"top-{nl}" if nl else "all(~3300)"
            run(hub, T, mk(nl, 0.05), f"  {tag} BAND [gross]", train=TRAIN, test=TEST, cost=False)
        for nl in (None, 1000, 500, 300):
            tag = f"top-{nl}" if nl else "all(~3300)"
            run(hub, T, mk(nl, 0.05), f"  {tag} BAND [net]",   train=TRAIN, test=TEST)
        sys.exit(0)

    # ── MODE=speed : VALIDATE Novy-Marx vol→speed. Bucket oversold (QP<15) events by realized-vol quintile; measure
    # median days-to-reversion (first close>prev-high within 20d) + reversion magnitude. High-vol ⇒ faster+bigger snap? ─
    if os.environ.get("MODE") == "speed":
        PX, BRK, QP, E, RVP, D = T["PX"], T["BRK"], T["QP"], T["E"], T["RVP"], T["D"]
        W = 20; days_b = [[] for _ in range(5)]; mag_b = [[] for _ in range(5)]; rev_b = [[] for _ in range(5)]
        for p in range(252, D - W, 5):                                            # stride 5 (sample)
            idx = np.where(E[p] & (QP[p] < 15) & np.isfinite(RVP[p]))[0]
            for j in idx:
                q = min(int(RVP[p, j] * 5), 4)
                k = next((kk for kk in range(1, W + 1) if BRK[p + kk, j]), W)      # days to first reversion (censored=W)
                reverted = k < W
                mag = (PX[p + k, j] / PX[p, j] - 1) if np.isfinite(PX[p + k, j]) and PX[p, j] else np.nan
                days_b[q].append(k); rev_b[q].append(reverted)
                if np.isfinite(mag): mag_b[q].append(mag)
        print("\n[MODE=speed] oversold events bucketed by realized-vol quintile (Q0=low vol → Q4=high vol):", flush=True)
        print(f"  {'vol quintile':14}{'n':>8}{'median days→rev':>18}{'% reverted≤20d':>16}{'mean rev mag':>14}", flush=True)
        for q in range(5):
            n = len(days_b[q])
            if not n: continue
            print(f"  Q{q:<13}{n:>8}{np.median(days_b[q]):>18.1f}{np.mean(rev_b[q]):>16.1%}{np.mean(mag_b[q]):>14.2%}", flush=True)
        print("  (Novy-Marx: high vol → FEWER days + BIGGER magnitude ⇒ vol ranks reversion SPEED.)", flush=True)
        sys.exit(0)

    # ── MODE=resid : THE DECISIVE TEST — sector-neutral RESIDUAL-target book (T5 lead) through the honest engine ──
    # Does IC +0.026 (residual target) convert to net SR? raw vs residual target · +STATE gate · proof STATE vs VIX gate.
    if os.environ.get("MODE") == "resid":
        secs = sector_of(hub, T["names"]); RN = int(os.environ.get("RN", 150)); HOLD = int(os.environ.get("HOLD", 10))
        Yraw = labels(hub, T["names"], T["GIDX"], H=HZ)                           # raw fwd return target
        Ysec = resid_fwd(hub, T["names"], T["GIDX"], H=HZ, kind="sector")         # DLS idiosyncratic target
        gS = gross_gate(T["GIDX"], col="surprise", on=1.0, off=0.1)              # STATE turbulence gate (crisis→10%)
        gV = gross_gate(T["GIDX"], col="vix", on=1.0, off=0.1)                   # DATAHUB VIX gate (article analog)
        gP = gross_gate(T["GIDX"], col="sig_port", on=1.0, off=0.1)              # STATE portfolio-VOL gate (vol gating)
        print(f"\n[MODE=resid] residual-target book · N={RN} hold={HOLD} · resid_feat={bool(os.environ.get('RESID_FEAT'))}", flush=True)
        print(f"  gate bear days — STATE-surprise {(gS<1.0).mean():.1%} · VIX {(gV<1.0).mean():.1%} · STATE-vol {(gP<1.0).mean():.1%}", flush=True)
        print(f"\n{'arm':50}{'SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>9}", flush=True)
        # integrate everything: residual target + smooth=3 (turnover) + KNIFE VETO (drop most-idio crash + top-vol tail)
        lo = lambda vi, vv: make_resid_signal(T, Ysec, secs, N=RN, hold=HOLD, test_len=TEST, sector_neutral=True, event_only=True,  long_only=True,  smooth=3, veto_idio=vi, veto_vol=vv)
        ls = lambda vi, vv: make_resid_signal(T, Ysec, secs, N=RN, hold=HOLD, test_len=TEST, sector_neutral=True, event_only=False, long_only=False, smooth=3, veto_idio=vi, veto_vol=vv)
        # ── T22 DECISIVE: does the IC lift (MSE 0.017 → L1 0.026 → clipL 0.030) CONVERT to net book SR? champion config +features ──
        Yclip = np.clip(Ysec, -0.05, 1e9)                                          # asymmetric knife-clip target (T20)
        mk = lambda Y, ls: make_resid_signal(T, Y, secs, N=RN, hold=HOLD, test_len=TEST, sector_neutral=True,
                                             event_only=True, long_only=True, smooth=3, veto_idio=0.2, veto_vol=0.8, loss=ls)
        print("-- LOSS × TARGET on the CHAMPION long-only book (resid · smooth=3 · veto BOTH) — does IC 0.017→0.026→0.030 convert to net SR? --", flush=True)
        run(hub, T, mk(Ysec,  "reg:squarederror"),    "    MSE · resid  (champion baseline)", train=TRAIN, test=TEST)
        run(hub, T, mk(Ysec,  "reg:absoluteerror"),   "    L1/median · resid  (STRUCTURAL)",  train=TRAIN, test=TEST)
        run(hub, T, mk(Ysec,  "reg:pseudohubererror"),"    Huber · resid",                    train=TRAIN, test=TEST)
        run(hub, T, mk(Yclip, "reg:squarederror"),    "    MSE · clipL−5%  (pragmatic)",       train=TRAIN, test=TEST)
        run(hub, T, mk(Yclip, "reg:absoluteerror"),   "    L1 · clipL−5%  (both)",             train=TRAIN, test=TEST)
        # ── T22b UTILIZATION: veto ON captures the knife-avoidance the robust loss provides → loss looks redundant. Turn veto
        # OFF (let the LOSS handle knives) + CONVICTION-weight + BREADTH → does the IC gain now convert? (user: "not utilizing it")
        mkv = lambda Y, ls, vi, vv, cv, n: make_resid_signal(T, Y, secs, N=n, hold=HOLD, test_len=TEST, sector_neutral=True,
                                             event_only=True, long_only=True, smooth=3, veto_idio=vi, veto_vol=vv, loss=ls, conv=cv)
        print("  [VETO OFF — let the LOSS handle knives; does robust loss/clip now beat MSE?]", flush=True)
        run(hub, T, mkv(Ysec,  "reg:squarederror",  0.0, 1.0, False, RN), "    MSE · resid · NO veto",       train=TRAIN, test=TEST)
        run(hub, T, mkv(Ysec,  "reg:absoluteerror", 0.0, 1.0, False, RN), "    L1 · resid · NO veto",        train=TRAIN, test=TEST)
        run(hub, T, mkv(Yclip, "reg:squarederror",  0.0, 1.0, False, RN), "    MSE · clipL−5% · NO veto",     train=TRAIN, test=TEST)
        print("  [CONVICTION-weight (utilize the ordering) + BREADTH (N=400), veto BOTH]", flush=True)
        run(hub, T, mkv(Ysec,  "reg:absoluteerror", 0.2, 0.8, True,  RN),  "    L1 · resid · conviction",     train=TRAIN, test=TEST)
        run(hub, T, mkv(Yclip, "reg:absoluteerror", 0.2, 0.8, True,  RN),  "    L1 · clipL · conviction",     train=TRAIN, test=TEST)
        run(hub, T, mkv(Yclip, "reg:absoluteerror", 0.2, 0.8, True,  400), "    L1 · clipL · conv · N=400",   train=TRAIN, test=TEST)
        sys.exit(0)

    # ── MODE=auc : DIAGNOSTIC — is the bounce classifier actually LEARNING? (strict-OOS AUC + IC per block) ──
    # Decisive check for "does ML help/hurt": AUC≈0.50 ⇒ label unpredictable (ML = noise+turnover ⇒ can only hurt);
    # AUC>0.53 ⇒ real signal (then any book-level hurt is a construction/cost problem, not a dead model).
    if os.environ.get("MODE") == "auc":
        from sklearn.metrics import roc_auc_score; from scipy.stats import spearmanr
        D, QP, E = T["D"], T["QP"], T["E"]; aucs, ics, base = [], [], []
        print("\n[MODE=auc] strict-OOS bounce classifier (train QP<15 subset, label 1[fwd>0]); article train=15y", flush=True)
        for s in range(TRAIN, D - TEST, TEST):
            tr = [p for p in range(s - TRAIN, s - HZ - 1) if p % 2 == 0]
            mdl = _train_side(T, YB, +1, tr)
            if mdl is None: continue
            Pv, Yb, Yr = [], [], []
            for p in range(s, min(s + TEST, D)):
                m = E[p] & (QP[p] < 15) & np.isfinite(YB[p])
                if m.sum() < 5: continue
                Pv.append(mdl.predict_proba(T["Farr"][p][m])[:, 1]); Yb.append((YB[p][m] > 0).astype(int)); Yr.append(YB[p][m])
            if not Pv: continue
            P = np.concatenate(Pv); B = np.concatenate(Yb); R = np.concatenate(Yr)
            if B.min() == B.max(): continue
            a = roc_auc_score(B, P); ic = spearmanr(P, R).correlation; aucs.append(a); ics.append(ic); base.append(B.mean())
            print(f"  {T['GIDX'][s].date()}  AUC {a:.3f}  IC(P,fwd) {ic:+.3f}  n={len(B)}  bounce-rate {B.mean():.1%}", flush=True)
        print("="*66)
        print(f"  mean AUC {np.mean(aucs):.4f} (0.50=no skill, >0.53=signal) · mean IC {np.mean(ics):+.4f} · base {np.mean(base):.1%}", flush=True)
        print("="*66); sys.exit(0)

    # ── MODE=hlgauss : STOP-REGRESSING (Farebrother 2024; Imani-White). MSE on the −90-skew knife target explodes → predict a
    # HISTOGRAM over the sector-residual reversion via classification (cross-entropy, tail-robust) → read out DM-style Σpₖμₖ AND
    # tail-aware functionals. The distributional target is the principled fix winsor only approximated (user: "stop-regression paper").
    if os.environ.get("MODE") == "hlgauss":
        from scipy.stats import spearmanr
        D, QP, E, GIDX, names = T["D"], T["QP"], T["E"], T["GIDX"], T["names"]
        r = hub.ret_d.reindex(index=GIDX, columns=names); RET5 = sum(r.shift(-d).values for d in range(1, HZ + 1))
        ret5_df = pd.DataFrame(RET5, index=GIDX, columns=names)
        secs = hub.sector.reindex(names).fillna("NA") if hub.sector is not None else pd.Series("NA", index=names)
        sres = (ret5_df - ret5_df.T.groupby(secs.values).transform("mean").T).values          # DLS sector-residual reversion
        KB = int(os.environ.get("KB", 10)); LAMP = float(os.environ.get("LAM_PEN", 3.0))       # buckets · knife-penalty weight
        rankpct = pd.DataFrame(sres, index=GIDX, columns=names).rank(axis=1, pct=True).values
        bins = np.clip(np.floor(rankpct * KB), 0, KB - 1)
        srank = norm.ppf(np.clip(rankpct, 1e-4, 1 - 1e-4))                                     # RANK-GAUSS target (kills the knife tail)

        # ── SOFT HL-Gauss over the RANK-GAUSS target (user idea): Gaussian soft labels over rank-space bins + custom soft
        # cross-entropy objective — the TRUE Stop-Regressing (soft labels keep the resolution hard bins throw away). ──
        if os.environ.get("SOFT"):
            import xgboost as xgb
            PROJ = os.environ.get("PROJ", "value"); SIG = float(os.environ.get("SIGMA", 1.0))
            def softmax(z): z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
            redges = np.linspace(-3, 3, KB + 1); rmids = 0.5 * (redges[:-1] + redges[1:])       # rank-space bins (PROJ=rank)
            def soft_rank(yg):
                c = norm.cdf((redges[None, :] - yg[:, None]) / SIG); L = c[:, 1:] - c[:, :-1]
                return L / (L.sum(1, keepdims=True) + 1e-12)
            def soft_value(y, C):                                                  # ★ C51/HL-Gauss VALUE-space 2-hot projection onto centers C
                n = len(y); L = np.zeros((n, len(C)), np.float64)                   # knives fall in the bottom bin → BOUNDED loss, NO clip
                j = np.clip(np.searchsorted(C, y) - 1, 0, len(C) - 2)
                w = np.clip((C[j + 1] - y) / (C[j + 1] - C[j] + 1e-12), 0, 1)
                ar = np.arange(n); L[ar, j] = w; L[ar, j + 1] = 1 - w; return L
            print(f"\n[MODE=hlgauss SOFT · PROJ={PROJ}] {KB}-bin soft-CE (custom obj) · {'value-space quantile centers (STRUCTURAL: no clip, bounded loss, asymmetry kept)' if PROJ=='value' else 'rank-gauss σ='+str(SIG)} · vs clipL 0.0300", flush=True)
            ic = {"Σpₖμₖ (mean)": [], "P(up)": [], f"−{LAMP}·p_worst": []}
            for s in range(TRAIN, D - TEST, TEST):
                tr = [p for p in range(s - TRAIN, s - HZ - 1) if p % 2 == 0]; Xl, Yl = [], []
                for p in tr:
                    m = E[p] & (QP[p] < 15) & np.isfinite(sres[p])
                    if m.sum() < 10: continue
                    Xl.append(T["Farr"][p][m]); Yl.append(sres[p][m])
                if not Xl: continue
                Xtr = np.vstack(Xl); ytr = np.concatenate(Yl)
                if PROJ == "value":
                    C = np.quantile(ytr, (np.arange(KB) + 0.5) / KB); C = np.maximum.accumulate(C)   # adaptive quantile centers (data-driven, no magic clip)
                    C[1:] += 1e-9 * np.arange(1, KB); L = soft_value(ytr, C); mids = C               # ties → strictly increasing
                else:
                    yg = norm.ppf(np.clip(pd.Series(ytr).rank(pct=True).values, 1e-4, 1 - 1e-4)); L = soft_rank(yg); mids = rmids
                dtr = xgb.DMatrix(Xtr)
                def obj(preds, d, L=L):                                            # xgboost≥2.1 wants 2D (n,K) grad/hess
                    z = preds if preds.ndim == 2 else preds.reshape(-1, KB); p = softmax(z)
                    return (p - L), np.maximum(p * (1 - p), 1e-6)
                bst = xgb.train({"num_class": KB, "max_depth": 5, "eta": 0.05, "subsample": 0.8,
                                 "colsample_bytree": 0.8, "tree_method": "hist", "nthread": 4,
                                 "base_score": 0.0, "disable_default_eval_metric": 1}, dtr, num_boost_round=150, obj=obj)
                for p in range(s, min(s + TEST, D)):
                    m = E[p] & (QP[p] < 15) & np.isfinite(RET5[p])
                    if m.sum() < 5: continue
                    P = softmax(bst.predict(xgb.DMatrix(T["Farr"][p][m]), output_margin=True)); rr = RET5[p][m]
                    mean_ro = P @ mids; ppos = P[:, mids > 0].sum(1) if (mids > 0).any() else np.zeros(len(rr)); pen = mean_ro - LAMP * P[:, 0]
                    for key, ro in zip(ic.keys(), [mean_ro, ppos, pen]):
                        if np.std(ro) > 0: ic[key].append(spearmanr(ro, rr).correlation)
            for k, v in ic.items(): print(f"  {k:22}{np.mean(v):>18.4f}", flush=True)
            print("  (STRUCTURAL test: does value-space soft-CE MATCH/BEAT the quick clip 0.0300 with NO magic threshold?)", flush=True)
            sys.exit(0)
        CC = dict(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=4)
        print(f"\n[MODE=hlgauss] DISTRIBUTIONAL residual target · {KB}-bucket histogram → read-out functionals · vs MSE-winsor 0.0261", flush=True)
        print(f"  {'readout':26}{'IC(readout,ret5)':>18}", flush=True)
        ic = {"Σpₖμₖ (DM mean)": [], "P(up buckets)": [], f"Σpₖμₖ − {LAMP}·p_worst (knife-pen)": [], "−E[bucket] (rank)": []}
        for s in range(TRAIN, D - TEST, TEST):
            tr = [p for p in range(s - TRAIN, s - HZ - 1) if p % 2 == 0]
            Xl, Yl, Cl = [], [], []
            for p in tr:
                m = E[p] & (QP[p] < 15) & np.isfinite(sres[p]) & np.isfinite(bins[p])
                if m.sum() < 10: continue
                Xl.append(T["Farr"][p][m]); Yl.append(bins[p][m].astype(int)); Cl.append(RET5[p][m])
            if not Xl: continue
            Xtr = np.vstack(Xl); ytr = np.concatenate(Yl); ctr = np.concatenate(Cl)
            centers = np.array([ctr[ytr == k].mean() if (ytr == k).any() else 0.0 for k in range(KB)])   # bucket→expected PnL
            clf = XGBClassifier(**CC, random_state=0).fit(Xtr, ytr); cls = clf.classes_; cen = centers[cls]
            for p in range(s, min(s + TEST, D)):
                m = E[p] & (QP[p] < 15) & np.isfinite(RET5[p])
                if m.sum() < 5: continue
                P = clf.predict_proba(T["Farr"][p][m]); rr = RET5[p][m]
                mean_ro = P @ cen                                                    # DM analog RET=Σpₖμₖ
                ppos = P[:, cen > 0].sum(1) if (cen > 0).any() else np.zeros(len(rr))
                pen = mean_ro - LAMP * P[:, 0]                                        # penalize prob mass in worst (knife) bucket
                erank = P @ cls.astype(float)                                         # expected bucket index (rank-space readout)
                for key, ro in zip(ic.keys(), [mean_ro, ppos, pen, erank]):
                    if np.std(ro) > 0: ic[key].append(spearmanr(ro, rr).correlation)
        for key, v in ic.items(): print(f"  {key:26}{np.mean(v):>18.4f}", flush=True)
        print("  (does the distribution's mean/tail-penalized readout BEAT MSE-on-winsor 0.0261? = the Stop-Regressing lever)", flush=True)
        sys.exit(0)

    # ── MODE=target : does a TRANSFORMED / SPEED-WEIGHTED target carry IC where the sign (AUC 0.50) does not? ──
    # sign is a wall, but 2nd-moment/speed is more predictable ([[target-snr-return-vs-secondmoment]]; Novy-Marx: vol→
    # fast reversal). Train an XGBRegressor on the QP<15 subset vs each target; report OOS IC(pred,target) AND — the one
    # that matters for PnL — IC(pred, plain 5d return). A speed target only WINS if it ranks the RETURN better, not just
    # itself. (λ<1 rewards a fast day-1 snap = "more desirable when reversion speed is high", the user's ask.)
    if os.environ.get("MODE") == "target":
        from scipy.stats import spearmanr
        D, QP, E, GIDX, names = T["D"], T["QP"], T["E"], T["GIDX"], T["names"]
        r = hub.ret_d.reindex(index=GIDX, columns=names)
        Rf = [r.shift(-d).values for d in range(1, HZ + 1)]                       # forward per-day returns r_{t+d}
        lam = float(os.environ.get("LAM", 0.7)); RET5 = sum(Rf)                   # plain cumulative fwd return
        ret5_df = pd.DataFrame(RET5, index=GIDX, columns=names)
        xres = (ret5_df.sub(ret5_df.mean(axis=1), axis=0)).values                # market-neutral residual
        if hub.sector is not None:                                               # sector-residual (DLS component 1)
            secs = hub.sector.reindex(names).fillna("NA")
            sres = (ret5_df - ret5_df.T.groupby(secs.values).transform("mean").T).values
        else: sres = xres
        rv21 = hub.rvol(21, "gk", "daily").reindex(index=GIDX, columns=names).values
        # constructed retrace-fraction: how much of the recent 3-day drop is retraced over H (bounded, reversion-native)
        drop = (hub.px_d.reindex(index=GIDX, columns=names) / hub.px_d.reindex(index=GIDX, columns=names).shift(3) - 1).values
        retrace = np.clip(RET5 / (-drop + 1e-6), -3, 3)                          # >0 when a drop is recovered
        TARGS = {"ret5 (raw signed)":     RET5,                                   # T4 baseline (sign is the wall)
                 "resid-XSEC (mkt-neut)":  xres,                                   # ── DLS: strip market ──
                 "resid-SECTOR (DLS c1)":  sres}                                   # ── DLS: strip industry (the big one) ──
        # ── T20 BATCH: the anatomy says the residual MEAN is flat + the risk is a −90-skew KNIFE tail → try target FORMS that
        # tame the tail / reframe the problem as knife-AVOIDANCE (DM analog: engineer the target, don't forecast raw fwd return).
        sres_df = pd.DataFrame(sres, index=GIDX, columns=names)
        rvol63 = hub.rvol(63, "gk", "daily").reindex(index=GIDX, columns=names).values
        h1 = Rf[0]; h1_sec = (pd.DataFrame(h1, index=GIDX, columns=names)                       # 1-day sector-residual (fast snap)
                              .pipe(lambda d: d - d.T.groupby(secs.values).transform("mean").T)).values if hub.sector is not None else h1
        TARGS.update({
            "resid-SEC winsor±10%":  np.clip(sres, -0.10, 0.10),                   # (a) clip knife tail → MSE fits the BULK reversion
            "resid-SEC winsor±5%":   np.clip(sres, -0.05, 0.05),
            "resid-SEC tanh(/5%)":   np.tanh(sres / 0.05),                         # (b) soft-squash tail (bounded, differentiable)
            "resid-SEC RANK(gauss)": norm.ppf(sres_df.rank(axis=1, pct=True).clip(1e-4, 1-1e-4).values),  # (c) rank-space (MOM champ)
            "knife-safe 1[r>-5%]":   (sres > -0.05).astype(float),                 # (d) REFRAME: classify NOT-a-knife (selection, not mean)
            "resid-SEC / rvol63":    sres / (rvol63 + 1e-9),                       # (e) risk-adj by slower vol (cleaner denom)
            "resid-SEC sign":        np.sign(sres),                                # (f) pure direction (control: is magnitude the wall?)
            "resid-1d snap (h1)":    h1_sec,                                       # (g) 1-day residual snap (fastest reversion, Novy-Marx)
        })
        # T20b: winsor±5% was the win → push tighter / asymmetric (kill the LEFT knife tail, keep bounce upside) / rank-of-winsor
        TARGS.update({
            "resid-SEC winsor±3%":  np.clip(sres, -0.03, 0.03),
            "resid-SEC winsor±2%":  np.clip(sres, -0.02, 0.02),
            "resid-SEC clipL−5%":   np.clip(sres, -0.05, 0.50),                    # ASYMMETRIC: clip knives only, keep bounce upside
            "resid-SEC clipL−3%":   np.clip(sres, -0.03, 0.50),
            "resid-SEC rank(w±5%)": norm.ppf(pd.DataFrame(np.clip(sres, -0.05, 0.05), index=GIDX, columns=names)
                                             .rank(axis=1, pct=True).clip(1e-4, 1-1e-4).values),  # rank the winsorized target
        })
        RC = dict(n_estimators=150, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=4)
        OBJ = os.environ.get("MODELOBJ")                                           # model-LOSS lever: robust loss = loss-side winsor
        if OBJ:
            RC["objective"] = OBJ                                                  # reg:pseudohubererror / reg:quantileerror / reg:absoluteerror
            if OBJ == "reg:quantileerror": RC["quantile_alpha"] = float(os.environ.get("QALPHA", 0.5))
        TONLY = os.environ.get("TONLY")                                            # run only matching target(s) for quick model probes
        if TONLY: TARGS = {k: v for k, v in TARGS.items() if TONLY in k}
        print(f"\n[MODE=target] XGBRegressor on QP<15 subset · train={TRAIN}d · IC vs own target AND vs 5d-return (PnL yardstick)"
              f"{' · obj=' + OBJ if OBJ else ''}", flush=True)
        print(f"  {'target':22}{'IC(pred,target)':>18}{'IC(pred,ret5)':>16}", flush=True)
        for tname, Y in TARGS.items():
            ics_t, ics_r = [], []
            for s in range(TRAIN, D - TEST, TEST):
                tr = [p for p in range(s - TRAIN, s - HZ - 1) if p % 2 == 0]
                Xl, Yl = [], []
                for p in tr:
                    m = E[p] & (QP[p] < 15) & np.isfinite(Y[p])
                    if m.sum() < 10: continue
                    Xl.append(T["Farr"][p][m]); Yl.append(Y[p][m])
                if not Xl: continue
                mdl = XGBRegressor(**RC, random_state=0).fit(np.vstack(Xl), np.concatenate(Yl))
                pr, tg, rr = [], [], []
                for p in range(s, min(s + TEST, D)):
                    m = E[p] & (QP[p] < 15) & np.isfinite(Y[p]) & np.isfinite(RET5[p])
                    if m.sum() < 5: continue
                    pr.append(mdl.predict(T["Farr"][p][m])); tg.append(Y[p][m]); rr.append(RET5[p][m])
                if not pr: continue
                P = np.concatenate(pr)
                ics_t.append(spearmanr(P, np.concatenate(tg)).correlation)
                ics_r.append(spearmanr(P, np.concatenate(rr)).correlation)
            print(f"  {tname:22}{np.mean(ics_t):>18.4f}{np.mean(ics_r):>16.4f}", flush=True)
        print("  (IC(pred,ret5) is what ranks PnL; a speed/transform target must WIN this column, not just its own.)", flush=True)
        sys.exit(0)

    # ── MODE=repl : reproduce the ARTICLE's headline conditions to locate the gap (net-long, flat 2bp, OOS from set START) ──
    # If this lands near their ~1.5, our engine is CORRECT and the whole gap = their assumptions (bull-beta + 2020 + 2bp +
    # microcaps), not a bug. Run with START=2008 so OOS≈2014+ (incl. 2020). Compare flat-2bp vs honest-tiered on the SAME book.
    if os.environ.get("MODE") == "repl":
        print(f"\n[MODE=repl] SAME faithful book, article conditions. net-long (long 1.1 / short {SFRAC}), OOS from {start}.", flush=True)
        print(f"{'arm':34}{'SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>9}", flush=True)
        sig = lambda ff: make_article_signal(T, YB, flat, THR, N, HMAX, STOP, SFRAC, TEST, use_filter=ff)
        run(hub, T, sig(True),  "QP+ML  flat 2bp  (their cost)", train=TRAIN, test=TEST, flatbps=2.0)
        run(hub, T, sig(False), "QP-rule flat 2bp (their cost)", train=TRAIN, test=TEST, flatbps=2.0)
        run(hub, T, sig(True),  "QP+ML  honest tiered",          train=TRAIN, test=TEST)
        run(hub, T, sig(False), "QP-rule honest tiered",         train=TRAIN, test=TEST)
        sys.exit(0)

    print(f"\n{'arm':32}{'SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>9}", flush=True)
    print("-- GROSS (cost-free: isolate signal from turnover-cost) --", flush=True)
    run(hub, T, make_article_signal(T, YB, flat, THR, N, HMAX, STOP, SFRAC, TEST, use_filter=False), "QP-event rule  [gross]", train=TRAIN, test=TEST, cost=False)
    run(hub, T, make_article_signal(T, YB, flat, THR, N, HMAX, STOP, SFRAC, TEST, use_filter=True),  "QP + ML filter [gross]", train=TRAIN, test=TEST, cost=False)
    print("-- NET (honest tiered cost + borrow) --", flush=True)
    run(hub, T, make_article_signal(T, YB, flat, THR, N, HMAX, STOP, SFRAC, TEST, use_filter=False), "QP-event rule (no filter)", train=TRAIN, test=TEST)
    run(hub, T, make_article_signal(T, YB, flat, THR, N, HMAX, STOP, SFRAC, TEST, use_filter=True),  "QP + ML filter (no gate)",  train=TRAIN, test=TEST)
    run(hub, T, make_article_signal(T, YB, reg,  THR, N, HMAX, STOP, SFRAC, TEST, use_filter=True),  "QP + ML filter + STATE gate", train=TRAIN, test=TEST)
