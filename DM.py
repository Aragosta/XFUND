#!/usr/bin/env python3
"""DM.py — the DM sleeve (CHAMPION): MULTI-HORIZON multi-label classification, tree-boosted vector-leaf.

XGBoost vector-leaf `multi_strategy="multi_output_tree"` on a MULTI-LABEL target — per stock, binary
[top-decile, bottom-decile] membership at each horizon h in H=(1,2,3). One shared tree emits the whole label
vector -> shared multi-horizon representation. Score = Σ_h [P(top@h) − P(bot@h)] -> rank -> top-q L/S,
dollar-neutral (gross 2.0). FEATURES = Han set (make_features) + TIME-SERIES features (52-wk-high, trend-R²,
tsmom). NO FFD (hurts MH-DM). Seeds=5, full broad universe, net via BACKTEST.py, OOS 2011+, 120-mo window.
Beta-neutralize the output book with BETANEUT.py before combining. Saves -> /tmp/dm_weights.pkl.
Leak-free: label at horizon h uses rets.iloc[t+h-1]; features frozen at t-1; pool excludes t>T-max(H)."""
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import numpy as np, pandas as pd
from xgboost import XGBClassifier, XGBRegressor
import BACKTEST
from DATAHUB import DataHub
from UNIVERSE import eligibility, ffd_scores
from features import make_features, MOM_WINDOWS

SEEDS = int(os.environ.get("SEEDS", 5)); TOP_Q, MIN_TRAIN_YRS = 0.05, 10    # champion = seeds=5
MAXTRAIN = int(os.environ.get("MAXTRAIN", 120)); DEC = 0.10                 # decile cutoffs for top/bottom labels
LONGONLY = int(os.environ.get("LONGONLY", 0))                              # 1 = long top-decile only, no short leg (matches MOM's native book)
XGB = dict(n_estimators=150, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           tree_method="hist", multi_strategy="multi_output_tree", objective="binary:logistic", verbosity=0)
REP = os.environ.get("REP", "hist")                                         # CHAMPION="hist" (K-bucket histogram -> RET reclassification); "dm"=coarse 2-bucket P(top)-P(bot)
KB  = int(os.environ.get("KB", 10))                                         # histogram buckets
SMOOTH = float(os.environ.get("SMOOTH", 0))                                 # 0=hard two-hot; >0=HL-Gauss σ (in bin widths)
TARGET = os.environ.get("TARGET", "ret")                                    # ret = fwd return (DM) | tval = slope/SE t-stat (MOM). (slope/cum/smooth/voladj removed — dominated, T10/T21)

print("[MH-DM] load from DataHub ...", flush=True)
hub = DataHub(start="2000-01-01", min_days=0)
pm = hub.delisted_prices("monthly")                                        # delisting-injected (global data eng)
rm = hub.clean_returns("monthly")                                          # 1/99-winsorized returns (global)
sm = hub.dollar_size("monthly")                                            # close × month-end volume
MINDVPCT = float(os.environ.get("MINDVPCT", 0.30)); MINDVABS = float(os.environ.get("MINDVABS", 1e6))  # liquidity filter (0/0 = full universe)
elig, short = eligibility(pm, sm, min_dollar_vol_pct=MINDVPCT, min_dollar_vol_abs=MINDVABS)
pnl = hub.pnl("monthly"); tcd = BACKTEST.tiered_transaction_costs(sm); bfd = BACKTEST.tiered_borrow_fees(sm)
me = rm.index; T = len(me); first_feat = max(MOM_WINDOWS) + 1

ADDFFD = int(os.environ.get("ADDFFD", 0))                                   # 1 = add FFD (frac-diff) features to make_features
FFD = None
if ADDFFD:
    print("[MH-DM] computing FFD scores ...", flush=True)
    FFD = ffd_scores(pm, MIN_TRAIN_YRS*12 + max(MOM_WINDOWS) + 1)
    for m in list(FFD): FFD[m] = FFD[m].reindex(rm.index)
ADDTS = int(os.environ.get("ADDTS", 1))                                     # DEFAULT ON: TIME-SERIES/trend features (52wk-high, trendR2, tsmom)
if ADDTS:
    logpm = np.log(pm); mi = pd.Series(np.arange(len(pm)), index=pm.index); wtr = 6
    xd = (mi - mi.rolling(wtr, min_periods=4).mean())
    yd = logpm.sub(logpm.rolling(wtr, min_periods=4).mean(), axis=0)
    cxy = (yd.mul(xd, axis=0)).rolling(wtr, min_periods=4).mean()
    r2 = (cxy**2).div((xd**2).rolling(wtr, min_periods=4).mean(), axis=0).div(yd.pow(2).rolling(wtr, min_periods=4).mean()+1e-12)
    TSF = {"hi52": pm/pm.rolling(12, min_periods=8).max() - 1, "trendR2": r2,
           "tsmom": (pm/pm.shift(6)-1).where(lambda z: z.abs()<5) * r2}

POOL  = int(os.environ.get("POOL", 0))                                      # 1 = add MOM's resmom + MACD features
RESID = int(os.environ.get("RESID", 0))                                     # 1 = RESIDUAL-return decile target (Blitz-HM)
RES = POOLF = None
if POOL or RESID:                                                           # residual (market-beta-neutralized) returns
    mktm = rm.mean(axis=1)
    bcov = rm.mul(mktm,axis=0).rolling(36,min_periods=24).mean() - rm.rolling(36,min_periods=24).mean().mul(mktm.rolling(36,min_periods=24).mean(),axis=0)
    bvar = (mktm**2).rolling(36,min_periods=24).mean() - mktm.rolling(36,min_periods=24).mean()**2
    betam = bcov.div(bvar,axis=0).shift(1); RES = rm.sub(betam.mul(mktm,axis=0))
if POOL:
    def _hl(s): return np.log(0.5)/np.log(1-1/s)
    resmom = RES.shift(1).rolling(11,min_periods=8).sum() / (RES.rolling(11,min_periods=8).std().shift(1)+1e-9)
    comp = 0.0                                                              # Baz MACD composite (monthly halflifes)
    for Sh,Lg in [(2,6),(4,12),(8,24)]:
        q=(pm.ewm(halflife=_hl(Sh)).mean()-pm.ewm(halflife=_hl(Lg)).mean())/pm.rolling(12,min_periods=6).std()
        y=q/q.rolling(24,min_periods=12).std(); comp = comp + y*np.exp(-y**2/4)/0.89
    POOLF = {"resmom": resmom, "macd": comp}
def labels(t, H):
    """multi-label binary matrix (idx x 2|H|): top/bottom decile of fwd (raw|resid) return t+h-1."""
    cols = {}
    src = RES if RESID else rm                                             # RESIDUAL-return target if RESID
    for h in H:
        r = src.iloc[t+h-1]; pr = r.rank(pct=True)
        cols[f"top{h}"] = (pr >= 1-DEC).astype(float); cols[f"bot{h}"] = (pr <= DEC).astype(float)
    return pd.DataFrame(cols)

def labels_hist(t, H, K):
    """multi-horizon HISTOGRAM target: soft TWO-HOT over K cross-sectional buckets of fwd return t+h-1, per h.
    Same target family as DM's top/bot deciles but the FULL distribution (DM = K=2 extremes). idx x (|H|*K)."""
    src = RES if RESID else rm; out = {}
    for h in H:
        if TARGET == "tval":                                              # slope/SE = t-stat = the MOM tval target (h+2 pts so h=1 valid)
            Y = np.log(pm.iloc[t-1:t+h+1].values.astype(float)); nn = Y.shape[0]; x = np.arange(nn).astype(float); xb = x.mean(); Sxx = max(np.sum((x-xb)**2), 1e-9)
            with np.errstate(all="ignore"):
                Ym = np.nanmean(Y, 0); sl = np.nansum((x-xb)[:, None] * (Y - Ym), 0) / Sxx
                resid = Y - (Ym + (x-xb)[:, None] * sl); se = np.sqrt(np.nansum(resid**2, 0) / max(nn-2, 1) / Sxx)
                r = pd.Series(sl / (se + 1e-9), index=pm.columns)
        else:                                                            # ret: single-month fwd return (DM default; residual if RESID)
            r = src.iloc[t+h-1]
        idx = r.index; pct = r.rank(pct=True).values; valid = np.isfinite(pct)
        pos = np.clip(np.nan_to_num(pct) * K - 0.5, 0, K-1)
        if SMOOTH > 0:                                                     # HL-Gauss: spread mass over bins (Gaussian)
            kk = np.arange(K)[None, :]; M = np.exp(-0.5 * ((kk - pos[:, None]) / SMOOTH) ** 2); M = M / (M.sum(1, keepdims=True) + 1e-9)
        else:                                                             # hard two-hot: mass on nearest two bins
            lo = np.floor(pos).astype(int); frac = pos - lo; hi = np.minimum(lo+1, K-1)
            M = np.zeros((len(r), K)); a = np.arange(len(r)); M[a, lo] += (1-frac); M[a, hi] += frac
        M[~valid] = np.nan
        for k in range(K): out[f"h{h}b{k}"] = pd.Series(M[:, k], index=idx)
    return pd.DataFrame(out)

def build(H):
    hmax = max(H); pool = {}
    for t in range(first_feat, T-hmax):
        F = make_features(rm, sm, t, ffd_scores=FFD).dropna()
        if ADDTS:
            for nm, df in TSF.items(): F[nm] = df.iloc[t-1].reindex(F.index)   # append TS feats (NaN -> XGB native)
        if POOL:
            for nm, df in POOLF.items(): F[nm] = df.iloc[t-1].reindex(F.index)  # append MOM resmom+MACD
        idx = F.index
        e = elig.iloc[t-1]; idx = idx.intersection(e.index[e.values])
        if len(idx) < 20: continue
        Y = (labels_hist(t, H, KB) if REP == "hist" else labels(t, H)).reindex(idx)
        ok = Y.notna().all(axis=1) & rm.iloc[t].reindex(idx).notna()
        idx = idx[ok.values]
        if len(idx) < 20: continue
        pool[t] = dict(F=F.loc[idx], Y=Y.loc[idx].values, fwd=rm.iloc[t].reindex(idx), dt=me[t])
    return pool

def walk(H):
    hmax = max(H); pool = build(H)
    first_pred = next(t for t in range(first_feat, T-hmax) if me[t].year >= 2011 and t in pool)
    store = {}; ics = []; model = None
    for year in sorted({me[t].year for t in range(first_pred, T-hmax)}):
        months = [t for t in range(first_pred, T-hmax) if me[t].year == year and t in pool]
        if not months: continue
        ats = sorted([t for t in pool if t < months[0]])[-MAXTRAIN:]
        if len(ats) >= 18:
            Xtr = pd.concat([pool[t]["F"] for t in ats]).values
            Ytr = np.vstack([pool[t]["Y"] for t in ats])
            model = [(XGBRegressor if REP == "hist" else XGBClassifier)(**XGB, random_state=s).fit(Xtr, Ytr) for s in range(SEEDS)]
        if model is None: continue
        nlab = len(H); centers = np.arange(KB) - (KB - 1) / 2.0
        for t in months:
            p = pool[t]
            if REP == "hist":                                              # read the tilt off the predicted histogram (law of total expectation)
                Q = np.mean([m.predict(p["F"].values) for m in model], axis=0).reshape(len(p["F"]), nlab, KB)
                Q = Q / (Q.sum(2, keepdims=True) + 1e-9); sc = (Q * centers).sum(2).mean(1)
            else:
                P = np.mean([np.asarray(m.predict_proba(p["F"].values)) for m in model], axis=0).reshape(len(p["F"]), 2*nlab)
                sc = P[:, :nlab].sum(1) - P[:, nlab:].sum(1)               # Σ P(top) − Σ P(bot)
            s = pd.Series(sc, index=p["F"].index)
            ics.append(np.corrcoef(pd.Series(sc).rank(), pd.Series(p["fwd"].values).rank())[0,1])
            sh = short.iloc[t-1]; shortable = s.index.intersection(sh.index[sh.values])
            q = 0.10 if LONGONLY else TOP_Q                                # long-decile (10%) when long-only, else 5% L/S
            n = max(1, int(len(s)*q)); w = pd.Series(0.0, index=s.index)
            w[s.nlargest(n).index] = 1.0/n
            if not LONGONLY:
                w[s.reindex(shortable).nsmallest(n).index] = -1.0/n        # short only borrowable
            store[me[t-1]] = w                                             # signal end of t-1, earns t
    W = pd.DataFrame(store).T.reindex(columns=pnl.columns).sort_index(); sig = list(W.index)
    net = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=sig, transaction_cost=tcd, borrow_fee=bfd)
    return np.nanmean(ics), net, W

print("="*70); print("DM SLEEVE — MULTI-HORIZON multi-label classification (Han + TS features, seeds=5) — net via BACKTEST.py")
print(f"  {'arm':16}{'IC':>9}{'net SR':>9}{'ann':>8}{'maxDD':>8}{'turn':>7}")
# CHAMPION ONLY: MH-DM (1,2,3) — 2-bucket P(top)-P(bot) on RETURNS, multi-horizon. (SH-DM single-horizon arm removed.)
nm, H = "MH-DM (1,2,3)", (1, 2, 3)
ic, r, W = walk(H)
print(f"  {nm:16}{ic:>9.4f}{r['sharpe']:>9.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}", flush=True)
pickle.dump(W, open("/tmp/dm_weights.pkl", "wb")); pickle.dump(r["returns"], open("/tmp/dm_returns.pkl", "wb"))
print("[done] DM champion book -> /tmp/dm_weights.pkl", flush=True)
