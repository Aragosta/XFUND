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
from xgboost import XGBClassifier
import BACKTEST
from DATAHUB import DataHub
from UNIVERSE import eligibility, ffd_scores
from features import make_features, MOM_WINDOWS

SEEDS = int(os.environ.get("SEEDS", 5)); TOP_Q, MIN_TRAIN_YRS = 0.05, 10    # champion = seeds=5
MAXTRAIN = int(os.environ.get("MAXTRAIN", 120)); DEC = 0.10                 # decile cutoffs for top/bottom labels
XGB = dict(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           tree_method="hist", multi_strategy="multi_output_tree", objective="binary:logistic", verbosity=0)

print("[MH-DM] load from DataHub ...", flush=True)
hub = DataHub(start="2000-01-01", min_days=0)
pm = hub.delisted_prices("monthly")                                        # delisting-injected (global data eng)
rm = hub.clean_returns("monthly")                                          # 1/99-winsorized returns (global)
sm = hub.dollar_size("monthly")                                            # close × month-end volume
elig, short = eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6)
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

VOLADJ = int(os.environ.get("VOLADJ", 0))                                   # 1 = target = fwd return / trailing vol
mvol = rm.rolling(6, min_periods=4).std() if VOLADJ else None               # trailing 6m monthly-return vol
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
    """multi-label binary matrix (idx x 2|H|): top/bottom decile of fwd (raw|resid|voladj) return t+h-1."""
    cols = {}
    sc = (mvol.iloc[t-1] + 1e-9) if VOLADJ else 1.0                         # vol known at signal date t-1 (point-in-time)
    src = RES if RESID else rm                                             # RESIDUAL-return target if RESID
    for h in H:
        r = src.iloc[t+h-1] / sc; pr = r.rank(pct=True)
        cols[f"top{h}"] = (pr >= 1-DEC).astype(float); cols[f"bot{h}"] = (pr <= DEC).astype(float)
    return pd.DataFrame(cols)

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
        Y = labels(t, H).reindex(idx)
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
            model = [XGBClassifier(**XGB, random_state=s).fit(Xtr, Ytr) for s in range(SEEDS)]
        if model is None: continue
        nlab = len(H)
        for t in months:
            p = pool[t]; P = np.mean([np.asarray(m.predict_proba(p["F"].values)) for m in model], axis=0)
            P = P.reshape(len(p["F"]), 2*nlab)                              # [top@h.., bot@h..]
            sc = P[:, :nlab].sum(1) - P[:, nlab:].sum(1)                    # Σ P(top) − Σ P(bot)
            s = pd.Series(sc, index=p["F"].index)
            ics.append(np.corrcoef(pd.Series(sc).rank(), pd.Series(p["fwd"].values).rank())[0,1])
            sh = short.iloc[t-1]; shortable = s.index.intersection(sh.index[sh.values])
            n = max(1, int(len(s)*TOP_Q)); w = pd.Series(0.0, index=s.index)
            w[s.nlargest(n).index] = 1.0/n
            w[s.reindex(shortable).nsmallest(n).index] = -1.0/n            # short only borrowable
            store[me[t-1]] = w                                             # signal end of t-1, earns t
    W = pd.DataFrame(store).T.reindex(columns=pnl.columns).sort_index(); sig = list(W.index)
    net = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=sig, transaction_cost=tcd, borrow_fee=bfd)
    return np.nanmean(ics), net, W

print("="*70); print("DM SLEEVE — MULTI-HORIZON multi-label classification (Han + TS features, seeds=5) — net via BACKTEST.py")
print(f"  {'arm':16}{'IC':>9}{'net SR':>9}{'ann':>8}{'maxDD':>8}{'turn':>7}")
ARMSEL = os.environ.get("ARM", "mh")                                       # default: MH champion only
out = {}
_arms = [("SH-DM (1)", (1,)), ("MH-DM (1,2,3)", (1,2,3))]
if ARMSEL == "mh": _arms = [("MH-DM (1,2,3)", (1,2,3))]
for nm, H in _arms:
    ic, r, W = walk(H); out[nm] = (ic, r["sharpe"])
    print(f"  {nm:16}{ic:>9.4f}{r['sharpe']:>9.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}", flush=True)
    if "MH" in nm:                                                          # the champion book -> canonical dm_weights
        pickle.dump(W, open("/tmp/dm_weights.pkl","wb")); pickle.dump(r["returns"], open("/tmp/dm_returns.pkl","wb"))
print("[done] DM book -> /tmp/dm_weights.pkl", flush=True)
