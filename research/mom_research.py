#!/usr/bin/env python3
"""
mom_research.py — THE reusable MOM-sleeve research harness. ONE script, OVERWRITTEN per experiment.

WORKFLOW (per RESEARCH_PROTOCOL.md Phase R → 4):
  1. Read MOM_research.md + the ⚠️ MANDATORY papers (papers/mom/).
  2. Edit ONLY the `EXPERIMENT` block at the bottom for the idea you're testing.
  3. Run:  PYTHONPATH=/Users/enzokreeft/XFUND python3 research/mom_research.py
  4. If the result is noteworthy → append a T-entry to MOM_research.md, then OVERWRITE this block for the next
     idea. DO NOT create research/<new_one_off>.py — that clutter is what this file replaces.

Reuses CURRENT BEST/mom_layer.MomLayer for the leak-free pool + full DM feature set, and provides the standard
scorers and a net() reporter (long-decile AND L/S books). Config via env: SEEDS, TOPN, MINDVPCT, MINDVABS
(0/0 = full universe, no liquidity filter = Han's setup).
"""
import warnings, os, sys
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, "/Users/enzokreeft/XFUND"); sys.path.insert(0, "/Users/enzokreeft/XFUND/CURRENT BEST")
import numpy as np, pandas as pd
from scipy.stats import norm, spearmanr
from xgboost import XGBClassifier, XGBRegressor
import BACKTEST
from mom_layer import MomLayer, KB, HZ, CEN, REGC, _bkt

def grank(a): a = np.asarray(a, float); r = pd.Series(a).rank(method="average"); return norm.ppf((r - 0.5) / (len(r) + 0.0))
def soft(y, K):
    r = pd.Series(y).rank(method="first").values; pos = np.clip((r - 0.5) / len(r) * K - 0.5, 0, K - 1)
    lo = np.floor(pos).astype(int); fr = pos - lo; hi = np.minimum(lo + 1, K - 1)
    M = np.zeros((len(y), K), np.float32); a = np.arange(len(y)); M[a, lo] += 1 - fr; M[a, hi] += fr; return M
def _tr(ml, k): return [j for j in ml.keys if j <= k - HZ][-ml.maxtrain:]        # embargoed rolling training months
def _oos(ml): return [k for k in ml.keys if ml.pool[k]["dt"].year >= 2011]
def _refit(ml, k, mdl, months=(1, 7)): return ml.pool[k]["dt"].month in months or mdl is None
def _finite(*arrs):
    m = np.ones(len(arrs[0]), bool)
    for a in arrs: m &= np.isfinite(a)
    return m

# ── STANDARD SCORERS (each returns dict: k -> Series(score, index=names)) ──────────────────────────────────
def single_hist(ml, field, centers=CEN):                     # histogram over field-buckets -> RET = Σpₖ·centerₖ
    pool, out, mdl = ml.pool, {}, None
    for k in _oos(ml):
        if _refit(ml, k, mdl):
            tr = _tr(ml, k)
            if len(tr) >= 24:
                Xl = [pool[j]["X"][_finite(pool[j][field])] for j in tr]
                yl = [_bkt(pool[j][field][_finite(pool[j][field])], KB) for j in tr]
                Xt = np.vstack(Xl); yt = np.concatenate(yl)
                mdl = [XGBClassifier(**REGC, objective="multi:softprob", num_class=KB, random_state=s).fit(Xt, yt) for s in range(ml.seeds)]
        if mdl is not None:
            P = np.mean([m.predict_proba(pool[k]["X"]) for m in mdl], axis=0); out[k] = pd.Series(P @ centers, index=pool[k]["idx"])
    return out

def dm_2bucket(ml, field):                                   # DM native: P(top-decile) − P(bottom-decile) of field
    pool, out, mdl = ml.pool, {}, None
    for k in _oos(ml):
        if _refit(ml, k, mdl):
            tr = _tr(ml, k)
            if len(tr) >= 24:
                Xt = np.vstack([pool[j]["X"][_finite(pool[j][field])] for j in tr])
                b = [_bkt(pool[j][field][_finite(pool[j][field])], KB) for j in tr]
                yt = np.concatenate([(bb == KB - 1).astype(int) for bb in b]); yb = np.concatenate([(bb == 0).astype(int) for bb in b])
                mdl = {"t": [XGBClassifier(**REGC, random_state=s).fit(Xt, yt) for s in range(ml.seeds)],
                       "b": [XGBClassifier(**REGC, random_state=s).fit(Xt, yb) for s in range(ml.seeds)]}
        if mdl is not None:
            pt = np.mean([m.predict_proba(pool[k]["X"])[:, 1] for m in mdl["t"]], axis=0)
            pb = np.mean([m.predict_proba(pool[k]["X"])[:, 1] for m in mdl["b"]], axis=0)
            out[k] = pd.Series(pt - pb, index=pool[k]["idx"])
    return out

def multi_hist(ml, fields):                                  # ONE multi_output_tree over [soft(f) for f in fields]
    """Returns dict k -> DataFrame(N × len(fields)) of per-field RET (columns aligned to `fields`)."""
    pool, out, mdl, H = ml.pool, {}, None, len(fields)
    for k in _oos(ml):
        if _refit(ml, k, mdl, months=(1,)):
            tr = _tr(ml, k)
            if len(tr) >= 24:
                Xl, Yl = [], []
                for j in tr:
                    m = _finite(*[pool[j][f] for f in fields])
                    if m.sum() < 50: continue
                    Xl.append(pool[j]["X"][m]); Yl.append(np.hstack([soft(pool[j][f][m], KB) for f in fields]))
                Xt = np.vstack(Xl); Yt = np.vstack(Yl)
                mdl = XGBRegressor(**REGC, multi_strategy="multi_output_tree", objective="binary:logistic", random_state=0).fit(Xt, Yt)
        if mdl is not None:
            Q = mdl.predict(pool[k]["X"]).reshape(len(pool[k]["idx"]), H, KB); Q = Q / (Q.sum(2, keepdims=True) + 1e-9)
            out[k] = pd.DataFrame(Q @ CEN, index=pool[k]["idx"], columns=list(fields))
    return out

# ── REPORTER: long-decile OR L/S dollar-neutral (BORROWABLE shorts only), net of cost, IC vs fwd6 ───────────
def net(ml, Slong, tag, Sshort=None, ls=False, decile=0.10, evalfield="fwd6"):
    pool = ml.pool; tc = BACKTEST.tiered_transaction_costs(ml.sm); bf = BACKTEST.tiered_borrow_fees(ml.sm)
    Sshort = Sshort or Slong; rows, ics = {}, []
    for k in Slong:
        sl, ss = Slong[k].dropna(), Sshort[k].dropna()
        if len(sl) < 20: continue
        fwd = pd.Series(pool[k][evalfield], index=pool[k]["idx"]).reindex(sl.index)
        ics.append(spearmanr(sl.values, fwd.values, nan_policy="omit").correlation)
        n = max(1, int(len(sl) * decile)); w = pd.Series(0.0, index=sl.index); w[sl.nlargest(n).index] = 1.0 / n
        if ls:                                               # short BORROWABLE names only (DATAHUB global filter)
            sh = pd.Series(pool[k]["short_ok"], index=pool[k]["idx"]).reindex(ss.index).fillna(False)
            cand = ss[sh.values]; ns = max(1, int(len(cand) * decile))
            w[cand.nsmallest(ns).index] = -1.0 / ns
        rows[pool[k]["dt"]] = w
    W = pd.DataFrame(rows).T.reindex(columns=ml.pnl.columns)
    r = BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
    print(f"  {tag:20}{np.nanmean(ics):>8.4f}{r['sharpe']:>8.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}", flush=True)
    return r

def build(**kw):                                             # convenience: MomLayer on a DATAHUB global tier
    seeds = int(os.environ.get("SEEDS", kw.pop("seeds", 2))); topn = os.environ.get("TOPN"); topn = int(topn) if topn else kw.pop("topn", None)
    tier = os.environ.get("TIER", kw.pop("tier", "liquid"))  # 'liquid'|'relaxed'|'base'; shorts always borrowable
    ml = MomLayer(seeds=seeds, topn=topn, tier=tier, **kw); ml.build_pool()
    print(f"[pool] tier={tier} · {len(ml.keys)} months, avg {np.mean([len(ml.pool[k]['idx']) for k in ml.keys if ml.pool[k]['dt'].year>=2011]):.0f} names, {len(ml.feat_names)} feats · shorts=borrowable", flush=True)
    return ml


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT — edit ONLY below this line, run, then log noteworthy results to MOM_research.md and overwrite.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # T25 re-test: vector [return-hist, tval-hist] with BORROWABLE shorts enforced (DATAHUB global filter).
    # Does asym (long tval / short return) or sym L/S survive now that we don't short un-borrowable names?
    ml = build(tier="liquid")                                # env: SEEDS/TOPN/TIER
    V = multi_hist(ml, ["fwd6", "tval"])                     # V[k]['fwd6']=E[return], V[k]['tval']=E[tval]
    RETv = {k: V[k]["fwd6"] for k in V}; TVAL = {k: V[k]["tval"] for k in V}
    SYM = {k: pd.Series((grank(V[k]["fwd6"].values) + grank(V[k]["tval"].values)) / 2.0, index=V[k].index) for k in V}
    print(f"\n{'book':20}{'IC6m':>8}{'net SR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    net(ml, TVAL, "tval_LS",  Sshort=TVAL, ls=True)
    net(ml, RETv, "ret_LS",   Sshort=RETv, ls=True)
    net(ml, TVAL, "asym_LS",  Sshort=RETv, ls=True)          # long tval, short return
    net(ml, SYM,  "sym_LS",   Sshort=SYM,  ls=True)
    net(ml, TVAL, "tval_long")
    net(ml, RETv, "ret_long")
