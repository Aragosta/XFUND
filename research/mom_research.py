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
    r["rankIC"] = float(np.nanmean(ics)); return r

def build(**kw):                                             # convenience: MomLayer on a DATAHUB global tier
    seeds = int(os.environ.get("SEEDS", kw.pop("seeds", 2))); topn = os.environ.get("TOPN"); topn = int(topn) if topn else kw.pop("topn", None)
    tier = os.environ.get("TIER", kw.pop("tier", "liquid"))  # 'liquid'|'relaxed'|'base'; shorts always borrowable
    ml = MomLayer(seeds=seeds, topn=topn, tier=tier, **kw); ml.build_pool()
    print(f"[pool] tier={tier} · {len(ml.keys)} months, avg {np.mean([len(ml.pool[k]['idx']) for k in ml.keys if ml.pool[k]['dt'].year>=2011]):.0f} names, {len(ml.feat_names)} feats · shorts=borrowable", flush=True)
    return ml


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT — edit ONLY below this line, run, then log noteworthy results to MOM_research.md and overwrite.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# ── EXTRA SCORERS for this experiment (distributional readout variants) ──────────────────────────────────
def mean_reg(ml, field):                                     # BASE: XGBRegressor on grank(field) — point regression
    pool, out, mdl = ml.pool, {}, None
    for k in _oos(ml):
        if _refit(ml, k, mdl):
            tr = _tr(ml, k)
            if len(tr) >= 24:
                Xl = [pool[j]["X"][_finite(pool[j][field])] for j in tr]
                yl = [grank(pool[j][field][_finite(pool[j][field])]) for j in tr]
                mdl = [XGBRegressor(**REGC, random_state=s).fit(np.vstack(Xl), np.concatenate(yl)) for s in range(ml.seeds)]
        if mdl is not None:
            out[k] = pd.Series(np.mean([m.predict(pool[k]["X"]) for m in mdl], axis=0), index=pool[k]["idx"])
    return out

def softg(y, K, sigma):                                      # HL-Gauss soft labels: Gaussian mass spread over K bins
    r = pd.Series(y).rank(method="first").values; pos = np.clip((r - 0.5) / len(r) * K - 0.5, 0, K - 1)
    kk = np.arange(K)[None, :]; M = np.exp(-0.5 * ((kk - pos[:, None]) / sigma) ** 2)
    return (M / (M.sum(1, keepdims=True) + 1e-9)).astype(np.float32)

def hlgauss_hist(ml, field, sigma=1.0):                      # HL-Gauss histogram (soft-bin CE proxy) -> RET = Σpₖ·centerₖ
    pool, out, mdl = ml.pool, {}, None
    for k in _oos(ml):
        if _refit(ml, k, mdl):
            tr = _tr(ml, k)
            if len(tr) >= 24:
                Xl, Yl = [], []
                for j in tr:
                    m = _finite(pool[j][field]); Xl.append(pool[j]["X"][m]); Yl.append(softg(pool[j][field][m], KB, sigma))
                mdl = [XGBRegressor(**REGC, multi_strategy="multi_output_tree", objective="binary:logistic", random_state=s).fit(np.vstack(Xl), np.vstack(Yl)) for s in range(ml.seeds)]
        if mdl is not None:
            Q = np.mean([m.predict(pool[k]["X"]) for m in mdl], axis=0); Q = Q / (Q.sum(1, keepdims=True) + 1e-9)
            out[k] = pd.Series(Q @ CEN, index=pool[k]["idx"])
    return out

# ── FACTOR MOMENTUM (Ehsani-Linnainmaa 2022): PCA the return panel, momentum-time the PCs, project to stocks ──
def facmom(ml, K=8, win=60, mom_lb=11, skip=1):
    """score_i(t) = Σ_k loading_i,k · (Σ past factor-k returns).  PIT: window ends at iloc[k-1]; PC-sign cancels
    (loading & factor-return flip together). Leak-free. Higher = more bullish (factor-momentum tailwind)."""
    R = ml.rm.values; cols = ml.cols; out = {}
    for k in _oos(ml):
        idx = ml.pool[k]["idx"]; lo = max(0, k - win); M = R[lo:k]           # window months × N, ≤ iloc[k-1]
        if M.shape[0] < mom_lb + skip + 2: continue
        valid = np.isfinite(M).mean(0) > 0.8                                 # names with data through the window
        Ms = np.where(np.isfinite(M[:, valid]), M[:, valid], 0.0)
        Ms = Ms - Ms.mean(0, keepdims=True)                                  # demean each name over the window
        try: U, S, Vt = np.linalg.svd(Ms, full_matrices=False)
        except np.linalg.LinAlgError: continue
        Kk = min(K, Vt.shape[0]); V = Vt[:Kk].T                              # loadings  (Nsub × K)
        F = Ms @ V                                                           # factor returns (win × K), variance-ordered
        fm = F[-(mom_lb + skip):-skip].sum(0) if skip else F[-mom_lb:].sum(0)  # 12-1 factor momentum per PC
        out[k] = pd.Series(V @ fm, index=cols[valid]).reindex(idx)          # project factor-momentum back to stocks
    return out


def hist_P(ml, field):                                       # like single_hist but RETURN the full bin-probability matrix P
    pool, out, mdl = ml.pool, {}, None
    for k in _oos(ml):
        if _refit(ml, k, mdl):
            tr = _tr(ml, k)
            if len(tr) >= 24:
                Xt = np.vstack([pool[j]["X"][_finite(pool[j][field])] for j in tr])
                yt = np.concatenate([_bkt(pool[j][field][_finite(pool[j][field])], KB) for j in tr])
                mdl = [XGBClassifier(**REGC, objective="multi:softprob", num_class=KB, random_state=s).fit(Xt, yt) for s in range(ml.seeds)]
        if mdl is not None:
            P = np.mean([m.predict_proba(pool[k]["X"]) for m in mdl], axis=0)
            out[k] = (pool[k]["idx"], P)
    return out


def _hist_veto():
    # ============================================================================================================
    # BIMODAL TAIL-MASS VETO — can the risk layer SALVAGE the L/S short leg? (user 2026-07-24). The short leg is the
    # tax ([[short-leg-is-the-tax]]); [[short-vol-thesis-proven]] showed HARD-capping squeezes flipped a short book
    # 0.28→1.69. RISK ACTION 3 currently uses a SOFT realized-semivol down-weight → L/S+veto = −0.08 (didn't salvage).
    # TEST: does a FORWARD-LOOKING squeeze forecast (the return-histogram's UPPER-tail mass) beat semi-vol, and does a
    # HARD exclusion beat a soft down-weight, at salvaging the short leg? Ranking score = MOM(tval) champion, gross held
    # constant (dollar-neutral). Borrowable shorts only. top-1000 seeds=1 screen.
    # ============================================================================================================
    import RISK
    os.environ.setdefault("TOPN", "1000"); ml = build(tier="liquid", pool=True); pool = ml.pool
    MOM = single_hist(ml, "tval")                            # ranking score (champion)
    PH = hist_P(ml, "fwd6"); nb = 2                          # return-histogram → tail masses (bottom/top nb of KB bins)
    SQ = {k: pd.Series(PH[k][1][:, -nb:].sum(1), index=PH[k][0]) for k in PH}   # forecast SQUEEZE prob (upper tail)
    try:
        up, down = RISK.tail_props(ml.hub, "monthly", 63)    # realized semi-vol proxy (up = squeeze axis)
        SV = {k: up.iloc[k - 1].reindex(pool[k]["idx"]) for k in MOM} if up is not None else None
    except Exception:
        SV = None
    tc = BACKTEST.tiered_transaction_costs(ml.sm); bf = BACKTEST.tiered_borrow_fees(ml.sm)

    def book(ls=False, sq=None, mode="down"):                # sq: dict k→Series squeeze-propensity for the SHORT leg
        rows = {}
        for k in MOM:
            s = MOM[k].dropna()
            if len(s) < 20: continue
            n = max(1, int(len(s) * 0.10)); w = pd.Series(0.0, index=s.index); w[s.nlargest(n).index] = 1.0 / n
            if ls:
                sh = pd.Series(pool[k]["short_ok"], index=pool[k]["idx"]).reindex(s.index).fillna(False)
                cand = s[sh.values]
                if len(cand) >= 20:
                    bn = max(1, int(len(cand) * 0.10)); names = cand.nsmallest(bn).index; sw = pd.Series(1.0 / bn, index=names)
                    if sq is not None and sq.get(k) is not None:
                        pr = sq[k].reindex(names).rank(pct=True).fillna(0.5)     # 1.0 = worst (most squeeze-prone)
                        f = (1.0 - 0.8 * pr).clip(lower=0.2) if mode == "down" else (pr <= 0.75).astype(float)  # hard = drop top-quartile
                        sw = sw * f; tot = sw.sum(); sw = sw / tot if tot > 0 else sw   # renorm → leg gross held at 1 (dollar-neutral)
                    w = w.sub(sw, fill_value=0.0)
            rows[pool[k]["dt"]] = w
        W = pd.DataFrame(rows).T.reindex(columns=ml.pnl.columns)
        return BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)

    print(f"\n[BIMODAL VETO · can risk salvage L/S?] top-{ml.topn or 'full'} · seeds={ml.seeds} · net · borrowable shorts")
    print(f"  {'book':32}{'netSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    def show(tag, r): print(f"  {tag:32}{r['sharpe']:>7.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}", flush=True)
    show("long-only (reference)",           book(ls=False))
    show("L/S no veto (baseline)",          book(ls=True))
    if SV is not None:
        show("L/S semi-vol squeeze down-wt", book(ls=True, sq=SV, mode="down"))
    show("L/S hist-squeeze down-wt",        book(ls=True, sq=SQ, mode="down"))
    show("L/S hist-squeeze HARD-exclude",   book(ls=True, sq=SQ, mode="hard"))


def _twomodel():
    # ============================================================================================================
    # TWO-MODEL (asymmetric) MOMENTUM (user 2026-07-24: "separate short & long models into one combined book"). Per
    # [[design-philosophy]]: two heads, same machinery, differing only in TARGET. Long head = tval champion (wins long).
    # Short head = a DEDICATED model with a short-appropriate target, shorting ITS OWN bottom-decile. DECISIVE metric =
    # each short head's STANDALONE GROSS SR — T39 showed the short leg's problem is ~ZERO GROSS ALPHA; a separate model
    # only helps if a different target manufactures gross short edge the momentum features actually contain.
    #   short heads: (a) tval (baseline = share the long model) ; (b) ret1 reclassification (Han squeeze-protected) ;
    #               (c) squeeze-aware P(bot1m)−P(top1m) [short high-decline / low-squeeze]. top-1000 seeds=1 net.
    # ============================================================================================================
    os.environ.setdefault("TOPN", "1000"); ml = build(tier="liquid", pool=True); pool, pnl, sm = ml.pool, ml.pnl, ml.sm
    tc = BACKTEST.tiered_transaction_costs(sm); bf = BACKTEST.tiered_borrow_fees(sm)
    def bt(W, t, b): return BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=t, borrow_fee=b)
    LONG   = single_hist(ml, "tval")                         # long head = champion
    SH_ret = single_hist(ml, "fwd")                          # short head B: 1-month RETURN reclassification
    SH_sqz = dm_2bucket(ml, "fwd")                           # short head C: P(top1m)−P(bot1m); short LOWEST = decline w/o squeeze
    def shortgross(Ss):                                      # short-only book: gross SR / net SR of shorting Ss bottom-decile
        rows = {}
        for k in Ss:
            s = Ss[k].dropna(); sh = pd.Series(pool[k]["short_ok"], index=pool[k]["idx"]).reindex(s.index).fillna(False)
            cand = s[sh.values]
            if len(cand) < 20: continue
            n = max(1, int(len(cand) * 0.10)); w = pd.Series(0.0, index=s.index); w[cand.nsmallest(n).index] = -1.0 / n
            rows[pool[k]["dt"]] = w
        W = pd.DataFrame(rows).T.reindex(columns=pnl.columns); g = bt(W, 0.0, 0.0); n = bt(W, tc, bf)
        return g["sharpe"], g["ann_return"], n["sharpe"]
    print(f"\n[TWO-MODEL asymmetric] top-{ml.topn or 'full'} · seeds={ml.seeds} · long=tval / short=head · net")
    print(f"  {'book':30}{'combSR':>8}{'combDD':>8}  |  {'SHORT-only: grossSR':>20}{'grossAnn':>9}{'netSR':>7}", flush=True)
    rL = net(ml, LONG, "long-only (ref)", ls=False)
    for tag, Ss in [("tval (shared model)", LONG), ("ret1 (reclassif)", SH_ret), ("sqz-aware P(b)−P(t)", SH_sqz)]:
        rc = net(ml, LONG, f"L/S short={tag}", Sshort=Ss, ls=True)
        sg, sga, sn = shortgross(Ss)
        print(f"  {'  → '+tag:30}{rc['sharpe']:>8.2f}{rc['max_drawdown']:>8.1%}  |  {sg:>20.2f}{sga:>9.1%}{sn:>7.2f}", flush=True)


def _decomp():
    # ============================================================================================================
    # FULL L/S ANATOMY (user 2026-07-24: "why is the L/S book not producing; what is RISK really doing; analyze the
    # liquidity × squeeze interaction"). Three parts:
    #   PART A — LEG × COST decomposition of the REAL DM book (/tmp/dm_weights.pkl): long vs short leg, and for each
    #            attribute gross → −transaction-cost → −borrow-fee → net. Answers "WHERE and WHY it loses".
    #   PART B — RISK ACTION LADDER on the MOM score: equal → mdv(liq) → +band → +tail-veto → +beta-neut. Net SR +
    #            turnover + cost at each step → "what each RISK action DOES".
    #   PART C — LIQUIDITY × SQUEEZE interaction: among shorted names, corr(size, squeeze-propensity). If negative,
    #            mdv-weighting ALREADY avoids squeezers → the veto is redundant (the two RISK actions overlap).
    # ============================================================================================================
    import pickle, RISK
    os.environ.setdefault("TOPN", "1000"); ml = build(tier="liquid", pool=True)
    pnl, sm = ml.pnl, ml.sm
    tc = BACKTEST.tiered_transaction_costs(sm); bf = BACKTEST.tiered_borrow_fees(sm)
    def bt(W, t, b): return BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=t, borrow_fee=b)

    # ── PART A — decompose the real DM book into legs, attribute cost ──
    print("\n[PART A · LEG × COST decomposition of the REAL DM book]  (gross → −tc → −borrow → net)")
    print(f"  {'leg':12}{'grossSR':>8}{'grossAnn':>9}{'netSR':>7}{'netAnn':>8}{'turn':>6}{'tc_drag':>9}{'borrow_drag':>12}", flush=True)
    try:
        W = pickle.load(open("/tmp/dm_weights.pkl", "rb"))
        for tag, Wx in [("long", W.clip(lower=0)), ("short", W.clip(upper=0)), ("combined", W)]:
            g = bt(Wx, 0.0, 0.0); tconly = bt(Wx, tc, 0.0); net = bt(Wx, tc, bf)
            print(f"  {tag:12}{g['sharpe']:>8.2f}{g['ann_return']:>9.1%}{net['sharpe']:>7.2f}{net['ann_return']:>8.1%}"
                  f"{net['ann_turnover']:>6.1f}{g['ann_return']-tconly['ann_return']:>9.1%}{tconly['ann_return']-net['ann_return']:>12.1%}", flush=True)
    except FileNotFoundError:
        print("  /tmp/dm_weights.pkl missing — run DM.py first", flush=True)

    # ── PART B — RISK ACTION LADDER on the MOM(tval) score (L/S) ──
    Sd = single_hist(ml, "tval"); S = pd.DataFrame({ml.pool[k]["dt"]: Sd[k] for k in Sd}).T.reindex(columns=ml.cols)
    print("\n[PART B · RISK ACTION LADDER]  MOM(tval) L/S · each row ADDS one RISK action")
    print(f"  {'risk stack':32}{'netSR':>7}{'grossSR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    ladder = [
        ("equal-weight (no RISK)",        dict(weighting="equal", band=None)),
        ("+ mdv sizing (liquidity)",      dict(weighting="mdv",   band=None)),
        ("+ banding (.10/.20)",           dict(weighting="mdv",   band=(0.10, 0.20))),
        ("+ tail-veto (squeeze)",         dict(weighting="mdv",   band=(0.10, 0.20), tail_veto=True, veto_lam=1.0, veto_floor=0.0)),
        ("+ beta-neut",                   dict(weighting="mdv",   band=(0.10, 0.20), beta_neut=True)),
    ]
    for tag, kw in ladder:
        W = RISK.risk_book(S, ml.hub, tier="liquid", ls=True, **kw)
        g = bt(W, 0.0, 0.0); net = bt(W, tc, bf)
        print(f"  {tag:32}{net['sharpe']:>7.2f}{g['sharpe']:>8.2f}{net['ann_return']:>8.1%}{net['max_drawdown']:>8.1%}{net['ann_turnover']:>7.1f}", flush=True)

    # ── PART C — liquidity × squeeze interaction among shorted names ──
    up, down = RISK.tail_props(ml.hub, "monthly", 63)
    corrs = []
    for k in Sd:
        s = Sd[k].dropna(); sh = pd.Series(ml.pool[k]["short_ok"], index=ml.pool[k]["idx"]).reindex(s.index).fillna(False)
        cand = s[sh.values]; names = cand.nsmallest(max(1, int(len(cand) * 0.10))).index      # the SHORT book
        size = np.log(sm.iloc[k - 1].reindex(names) + 1); sq = up.iloc[k - 1].reindex(names) if up is not None else None
        if sq is None: continue
        corrs.append(spearmanr(size.values, sq.values, nan_policy="omit").correlation)
    print(f"\n[PART C · LIQUIDITY × SQUEEZE]  corr(log-size, squeeze-propensity) among shorted names = {np.nanmean(corrs):+.2f}")
    print("  (strongly NEGATIVE → big/liquid shorts are LOW-squeeze → mdv-weighting already avoids squeezers → veto redundant)", flush=True)


def _mh_short():
    # ============================================================================================================
    # MULTI-HORIZON reclassification for the SHORT + tval TERM-STRUCTURE (user 2026-07-24). Two ideas:
    #  (1) Han single-horizon ret1 reclassification already rescued the short (−0.22→−0.04, T36). The REAL DM is
    #      MULTI-HORIZON return reclassification (H=1,2,3) — consensus RET across horizons → squeeze-protected short.
    #  (2) ★ USER'S IDEA: multi-horizon reveals the TERM STRUCTURE — if predicted return/tval RISES across horizons the
    #      trend is accelerating (long); if it FALLS the trend is rolling over (short). Slope = RET@late − RET@early =
    #      a DIRECTIONAL short signal (deteriorating trend), not just "weak trend". Squeeze-protected (return reclassif).
    # long ranking = tval champion; short scores compared. top-1000 seeds=1 net.
    # ============================================================================================================
    os.environ.setdefault("TOPN", "1000"); ml = build(tier="liquid", pool=True); rm = ml.rm; pool = ml.pool
    for k in list(pool):                                     # single-month forward returns at horizons h=1,2,3 (DM labels)
        idx = pool[k]["idx"]
        for h in (1, 2, 3):
            kk = k + h - 1
            pool[k][f"r{h}"] = (rm.iloc[kk].reindex(idx).values if kk < len(rm) else np.full(len(idx), np.nan))
    TVAL = single_hist(ml, "tval")
    V = multi_hist(ml, ["r1", "r2", "r3"])                   # ONE multi_output tree → per-horizon RET (N×3), squeeze-protected
    LVL = {k: pd.Series(V[k].mean(axis=1).values, index=V[k].index) for k in V}          # MH-DM consensus level (real DM short)
    SLP = {k: pd.Series((V[k]["r3"] - V[k]["r1"]).values, index=V[k].index) for k in V}  # ★ term-structure slope: accel(+)/decel(−)
    print(f"\n[MH RECLASSIFICATION + TERM-STRUCTURE] top-{ml.topn or 'full'} · seeds={ml.seeds} · net")
    print(f"  {'book':38}{'IC6m':>8}{'netSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    net(ml, TVAL, "long-only (tval)",                    ls=False)
    net(ml, LVL,  "L/S MH-DM both (consensus RET)",      ls=True)          # multi-horizon return reclassification
    net(ml, TVAL, "L/S long tval / short MH-DM",         Sshort=LVL, ls=True)
    net(ml, SLP,  "L/S term-structure slope both",       ls=True)          # long accel, short decel
    net(ml, TVAL, "L/S long tval / short decel-slope",   Sshort=SLP, ls=True)
    # score-corr of the two short signals (are consensus-level and slope different bets?)
    kk = sorted(set(LVL) & set(SLP))
    sc = np.nanmean([spearmanr(LVL[k].reindex(SLP[k].index).values, SLP[k].values, nan_policy="omit").correlation for k in kk])
    print(f"  score-corr(MH-DM level, term-slope) = {sc:+.2f}", flush=True)


def _han_short():
    # ============================================================================================================
    # HAN'S RECLASSIFICATION AS THE SQUEEZE FILTER (user 2026-07-24: "losers bounce is EXACTLY what the bimodal dist
    # addresses — start with the Han paper"). RE-READ Han pp.1-8 (MOM_research.md L27,L80-91): the reclassification
    # RET=Σpₖμₖ on the RETURN target pushes bounce/squeeze-prone losers OUT of the short extreme (short-book size
    # $153M→$1589M). tval predicts E[tval] not E[return] → LOSES the short protection ("tval L/S blows up; return-RET
    # L/S survives"). And the squeeze is a SHORT-horizon (t+1) event. T34/T35 shorted on TVAL = the wrong score.
    # THIS test: long on tval (wins long), SHORT on the 1-month RETURN reclassification (Han's actual squeeze-protected
    # short). Prediction (Han): return-reclassified short SURVIVES where tval-short blows up. top-1000 seeds=1 net.
    # ============================================================================================================
    os.environ.setdefault("TOPN", "1000"); ml = build(tier="liquid", pool=True)
    TVAL = single_hist(ml, "tval")                           # long ranking + tval-short baseline (the one that "blows up")
    RET1 = single_hist(ml, "fwd")                            # 1-month RETURN reclassification = Han DM (squeeze-protected)
    RET6 = single_hist(ml, "fwd6")                           # 6-month return reclassification (wrong horizon for squeeze)
    print(f"\n[HAN RECLASSIFICATION SHORT] top-{ml.topn or 'full'} · seeds={ml.seeds} · long-decile / short-decile · net")
    print(f"  {'book':34}{'IC6m':>8}{'netSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    net(ml, TVAL, "long-only (tval)",                 ls=False)
    net(ml, TVAL, "L/S  short=tval (baseline, blows up)", ls=True)
    net(ml, RET1, "L/S  both=ret1 (pure Han DM)",     ls=True)
    net(ml, TVAL, "L/S  long tval / short ret1 (Han)", Sshort=RET1, ls=True)
    net(ml, TVAL, "L/S  long tval / short ret6",       Sshort=RET6, ls=True)


def _risk_ls():
    # ============================================================================================================
    # CAN THE **FULL** RISK LAYER SALVAGE L/S? (user 2026-07-24: "solvable in this exact RISK setup"; stop-loss=META not RISK).
    # T34's veto FAILED because it hand-rolled an EQUAL-WEIGHT short book — discarding mdv-weighting + banding, the two
    # actions that ARE the RISK layer's power (long EW 0.14 → mdv+band 0.70). THIS test uses the real RISK.risk_book
    # (SELECT band → SIZE mdv → TAIL-VETO bimodal → NEUTRALIZE beta) on a FRESH champion score, ls=True, veto-strength +
    # beta-neut sweep. The faithful "exact RISK setup" test the user asked for. top-1000 seeds=1 screen.
    # ============================================================================================================
    import RISK
    os.environ.setdefault("TOPN", "1000"); ml = build(tier="liquid", pool=True)
    Sd = single_hist(ml, "tval")                             # fresh champion score (NOT the stale Jul-18 parquet)
    S = pd.DataFrame({ml.pool[k]["dt"]: Sd[k] for k in Sd}).T.reindex(columns=ml.cols)
    tc = BACKTEST.tiered_transaction_costs(ml.sm); bf = BACKTEST.tiered_borrow_fees(ml.sm)
    print(f"\n[FULL RISK LAYER · can it salvage L/S?] top-{ml.topn or 'full'} · seeds={ml.seeds} · mdv+band · net")
    print(f"  {'risk actions':36}{'netSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    def run(tag, **kw):
        W = RISK.risk_book(S, ml.hub, tier="liquid", weighting="mdv", band=(0.10, 0.20), **kw)
        r = BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
        print(f"  {tag:36}{r['sharpe']:>7.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}", flush=True)
        return r
    run("long-only (mdv+band)",              ls=False)
    run("L/S no veto",                       ls=True)
    run("L/S tail-veto (lam.5, floor.2)",    ls=True, tail_veto=True, veto_lam=0.5, veto_floor=0.2)
    run("L/S tail-veto HARD (lam1, floor0)", ls=True, tail_veto=True, veto_lam=1.0, veto_floor=0.0)
    run("L/S beta-neut",                     ls=True, beta_neut=True)
    run("L/S tail-veto HARD + beta-neut",    ls=True, tail_veto=True, veto_lam=1.0, veto_floor=0.0, beta_neut=True)


def _ic_at(ml, S, field):                                    # mean monthly rank-IC of score vs a forward-return field
    ics = []
    for k in S:
        s = S[k].dropna();
        if len(s) < 20: continue
        fwd = pd.Series(ml.pool[k][field], index=ml.pool[k]["idx"]).reindex(s.index)
        ics.append(spearmanr(s.values, fwd.values, nan_policy="omit").correlation)
    return np.nanmean(ics)


def _facmom_deep():
    # ============================================================================================================
    # FACTOR MOMENTUM — optimally harvest the momentum premium as ONE model (user 2026-07-24: "one model per premium,
    # optimally harvest it"). Ehsani-Linnainmaa (2022 JF): factor momentum SUBSUMES stock momentum, lives in the
    # HIGH-EIGENVALUE PCs, is lower-turnover/higher-capacity. T31 confirmed it's genuinely orthogonal (0.15) but weak
    # in the naive K=8/60m/6m form. This run finds the OPTIMAL harvest, then tests whether it beats/absorbs stock-MOM.
    #   PART A — SWEEP window×#PC×horizon: which config best harvests the premium? IC at native 1/3/6/12m + net SR.
    #   PART B — COMBINE & SUBSUMPTION: does facmom ⊕ MOM raise COMBINED SR (corr 0.15), or does one span the other?
    # Screen top-1000 seeds=1 (Protocol Step 3); promote the winning config to full/seeds=5 next.
    # ============================================================================================================
    os.environ.setdefault("TOPN", "1000"); ml = build(tier="liquid", pool=True)
    pm, pool = ml.pm, ml.pool
    for k in list(pool):                                     # native-horizon forward returns for IC (fwd6 already in pool)
        for h in (1, 3, 12):
            kk = k + h
            pool[k][f"fwd{h}"] = ((pm.iloc[kk] / pm.iloc[k] - 1).reindex(pool[k]["idx"]).values
                                  if kk < len(pm) else np.full(len(pool[k]["idx"]), np.nan))
    hdr = f"  {'config':20}{'IC1':>7}{'IC3':>7}{'IC6':>7}{'IC12':>7}{'netSR':>7}{'maxDD':>7}{'turn':>6}"

    # ── PART A: sweep — find the optimal factor-momentum harvest ──
    print(f"\n[PART A · FACMOM SWEEP] top-{ml.topn or 'full'} · seeds={ml.seeds} · long-decile · net"); print(hdr, flush=True)
    tc = BACKTEST.tiered_transaction_costs(ml.sm); bf = BACKTEST.tiered_borrow_fees(ml.sm)
    def book_long(S):
        rows = {}
        for k in S:
            s = S[k].dropna()
            if len(s) < 20: continue
            n = max(1, int(len(s) * 0.10)); w = pd.Series(0.0, index=s.index); w[s.nlargest(n).index] = 1.0 / n
            rows[pool[k]["dt"]] = w
        W = pd.DataFrame(rows).T.reindex(columns=ml.pnl.columns)
        return BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
    best = (None, -9); FMcache = {}
    for win in (36, 60):
        for K in (3, 5, 10, 20):
            FM = facmom(ml, K=K, win=win); FMcache[(win, K)] = FM
            r = book_long(FM); ics = {h: _ic_at(ml, FM, f"fwd{h}") for h in (1, 3, 6, 12)}
            print(f"  win{win} K{K:<2}         {ics[1]:>7.3f}{ics[3]:>7.3f}{ics[6]:>7.3f}{ics[12]:>7.3f}{r['sharpe']:>7.2f}{r['max_drawdown']:>7.0%}{r['ann_turnover']:>6.1f}", flush=True)
            if r["sharpe"] > best[1]: best = ((win, K), r["sharpe"])
    bwin, bK = best[0]
    print(f"  → best config: win{bwin} K{bK}  (net SR {best[1]:.2f})", flush=True)

    # ── PART B: combine the best facmom with the MOM(tval) champion + subsumption (spanning) test ──
    FM = FMcache[best[0]]; MOM = single_hist(ml, "tval")
    kk = sorted(set(FM) & set(MOM))
    UNI = {k: pd.Series(0.5 * (grank(FM[k].reindex(MOM[k].index).values) + grank(MOM[k].values)), index=MOM[k].index) for k in kk}
    print(f"\n[PART B · COMBINE + SUBSUMPTION] best facmom(win{bwin},K{bK}) ⊕ MOM(tval) · long-decile · net"); print(hdr, flush=True)
    rF = book_long({k: FM[k] for k in kk}); rM = book_long(MOM); rU = book_long(UNI)
    for tag, r in [("facmom", rF), ("MOM(tval)", rM), ("z-sum combine", rU)]:
        print(f"  {tag:20}{'':>28}{r['sharpe']:>7.2f}{r['max_drawdown']:>7.0%}{r['ann_turnover']:>6.1f}", flush=True)
    B = pd.DataFrame({"facmom": rF["returns"], "MOM": rM["returns"]}).dropna()
    bl = 0.5 * B["facmom"] + 0.5 * B["MOM"]; blSR = bl.mean() / bl.std() * np.sqrt(12)
    # spanning: regress each book on the other; leftover mean (alpha) tells us if one absorbs the other
    def alpha(y, x):
        x1 = np.column_stack([np.ones(len(x)), x]); b = np.linalg.lstsq(x1, y, rcond=None)[0]
        resid = y - x1 @ b; return b[0] * 12, b[0] / (resid.std() / np.sqrt(len(resid)))     # ann alpha, t-stat
    aM, tM = alpha(B["MOM"].values, B["facmom"].values)      # MOM alpha AFTER controlling for facmom
    aF, tF = alpha(B["facmom"].values, B["MOM"].values)      # facmom alpha AFTER controlling for MOM
    print(f"  50/50 strategy blend SR = {blSR:.2f}   book-corr = {B.corr().iloc[0,1]:+.2f}", flush=True)
    print(f"  SPANNING: MOM alpha|facmom = {aM:+.1%}/yr (t={tM:+.1f}) · facmom alpha|MOM = {aF:+.1%}/yr (t={tF:+.1f})", flush=True)
    print(f"    (E-L claim = facmom SUBSUMES MOM → MOM alpha|facmom ≈ 0; our 0.15 corr predicts NEITHER spans the other)", flush=True)


def _stage0():
    # STAGE 0 — IC→SR LEAK DECOMPOSITION (user 2026-07-24). Same tval champion score, full universe; vary ONLY construction.
    # Q: the champion scored IC 0.117 / net SR 0.14 on full univ but 0.43 on top-1000 — where does the Sharpe leak between
    # a strong ranking and a weak book? EW-full (the 0.14) vs liquid-restricted (rank only top-1000 by $vol) vs liquidity-
    # weighted decile. If liquid-restrict recovers ~0.43 → the leak is UNIVERSE/CONSTRUCTION (junk small-caps), not the signal.
    ml = build(tier="liquid", pool=True); pm, sm = ml.pm, ml.sm
    S = single_hist(ml, "tval")                                        # ONE champion train; reuse for all construction variants
    tc = BACKTEST.tiered_transaction_costs(sm); bf = BACKTEST.tiered_borrow_fees(sm)
    def book(mode, dec=0.10, topliq=1000):
        rows, ics, junk = {}, [], []
        for k in S:
            s = S[k].dropna(); idx = ml.pool[k]["idx"]; sz = sm.iloc[k - 1].reindex(idx)
            liq = sz.nlargest(topliq).index
            if mode == "liqrestrict": s = s.reindex(liq).dropna()
            fwd = pd.Series(ml.pool[k]["fwd6"], index=idx).reindex(s.index); ics.append(spearmanr(s.values, fwd.values, nan_policy="omit").correlation)
            n = max(1, int(len(s) * dec)); top = s.nlargest(n).index; w = pd.Series(0.0, index=s.index)
            junk.append(np.mean([t not in liq for t in top]))         # frac of longs OUTSIDE tradable-liquid line
            if mode == "mdvwt": wv = sz.reindex(top).clip(lower=0).fillna(0); w[top] = wv / (wv.sum() + 1e-9)
            else: w[top] = 1.0 / n
            rows[ml.pool[k]["dt"]] = w
        W = pd.DataFrame(rows).T.reindex(columns=ml.pnl.columns)
        r = BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
        print(f"  {mode:16}{np.nanmean(ics):>8.4f}{r['sharpe']:>8.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}   junk-frac {np.nanmean(junk):.0%}", flush=True)
        return r
    print(f"\n[STAGE 0 · IC→SR LEAK] full liquid ~{int(np.mean([len(ml.pool[k]['idx']) for k in ml.keys if ml.pool[k]['dt'].year>=2011]))} names · tval champion · long-decile · net · seeds={ml.seeds}", flush=True)
    print(f"{'construction':16}{'IC6m':>8}{'net SR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    book("ewfull"); book("liqrestrict"); book("mdvwt")


def _stages123():
    # STAGES 1-3 on top-1000 (clean, no junk → isolates objective/architecture from the Stage-0 construction leak).
    ml = build(tier="liquid", pool=True, topn=1000)
    pm, rm, sm = ml.pm, ml.rm, ml.sm; OOSK = _oos(ml)
    def z(x): x = np.asarray(x, float); return (x - np.nanmean(x)) / (np.nanstd(x) + 1e-9)
    hdr = f"{'arm':24}{'IC6m':>8}{'net SR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}"
    print(f"\n[STAGES 1-3] top-1000 · seeds={ml.seeds} · long-decile · net", flush=True)

    # ── STAGE 1: objective alignment — full-rank CE (IC-max) vs TOP-DECILE-focused loss (trade what you train) ──
    def top_binary(field, dec=0.10):
        pool, out, mdl = ml.pool, {}, None
        for k in OOSK:
            if _refit(ml, k, mdl):
                tr = _tr(ml, k)
                if len(tr) >= 24:
                    Xl, yl = [], []
                    for j in tr:
                        v = pool[j][field]; m = _finite(v); vv = v[m]
                        Xl.append(pool[j]["X"][m]); yl.append((vv >= np.quantile(vv, 1 - dec)).astype(int))
                    mdl = [XGBClassifier(**REGC, random_state=s).fit(np.vstack(Xl), np.concatenate(yl)) for s in range(ml.seeds)]
            if mdl is not None:
                out[k] = pd.Series(np.mean([m.predict_proba(pool[k]["X"])[:, 1] for m in mdl], axis=0), index=pool[k]["idx"])
        return out
    print("\n-- Stage 1: objective (rank-CE vs top-decile-focused) --"); print(hdr, flush=True)
    net(ml, single_hist(ml, "tval"), "S1 base_rankCE")
    net(ml, top_binary("tval"),      "S1 top_focused")

    # ── STAGE 2: unify DM+MOM as two readouts of ONE multi_output tree; combine + measure redundancy ──
    V = multi_hist(ml, ["tval", "fwd6"])
    MOMr = {k: V[k]["tval"] for k in V}; DMr = {k: V[k]["fwd6"] for k in V}
    UNI = {k: pd.Series(0.5 * (grank(V[k]["tval"].values) + grank(V[k]["fwd6"].values)), index=V[k].index) for k in V}
    print("\n-- Stage 2: DM+MOM as readouts of ONE model --"); print(hdr, flush=True)
    rM = net(ml, MOMr, "S2 MOM(tval)"); rD = net(ml, DMr, "S2 DM(ret)"); rU = net(ml, UNI, "S2 UNIFIED")
    cc = pd.DataFrame({"M": rM["returns"], "D": rD["returns"]}).dropna().corr().iloc[0, 1]
    print(f"   corr(MOM book, DM book) = {cc:+.2f}   (unification target: UNIFIED ≥ best single, lower turn)", flush=True)

    # ── STAGE 3: >sum of inputs — model OVER the signal library (+ state) vs best-single & equal-blend (T26 bar) ──
    MOM = pm / pm.shift(11) - 1; RESM = ml.POOLF["resmom"]; HI = ml.TSF["hi52"]; MACD = ml.POOLF["macd"]
    lp = np.log(pm.values); nP = 7; xc = np.arange(nP) - (nP - 1) / 2; Sxx = (xc ** 2).sum()
    TVP = pd.DataFrame(np.nan, index=pm.index, columns=pm.columns)
    for t in range(nP - 1, len(pm)):
        Y = lp[t - nP + 1:t + 1]; Ym = np.nanmean(Y, 0); sl = np.nansum(xc[:, None] * (Y - Ym), 0) / Sxx
        rd = Y - (Ym + xc[:, None] * sl); se = np.sqrt(np.nansum(rd ** 2, 0) / max(nP - 2, 1) / Sxx); TVP.iloc[t] = sl / (se + 1e-9)
    fpos = (rm > 0).rolling(11, min_periods=8).mean(); fneg = (rm < 0).rolling(11, min_periods=8).mean(); IDd = np.sign(MOM) * (fneg - fpos)
    disp = rm.std(axis=1)                                              # cross-sectional dispersion = regime STATE (broadcast per month)
    def sigmat(k, withstate):
        idx = ml.pool[k]["idx"]; m = MOM.iloc[k - 1].reindex(idx).values
        fip = z(m) - z(IDd.iloc[k - 1].reindex(idx).values)
        cols = [m, RESM.iloc[k - 1].reindex(idx).values, HI.iloc[k - 1].reindex(idx).values,
                MACD.iloc[k - 1].reindex(idx).values, TVP.iloc[k - 1].reindex(idx).values, fip]
        if withstate: cols.append(np.full(len(idx), disp.iloc[k - 1]))
        return np.column_stack(cols).astype(np.float32)
    baseX = {k: ml.pool[k]["X"].copy() for k in ml.keys}
    def model_over_signals(withstate, tag):
        for k in ml.keys: ml.pool[k]["X"] = sigmat(k, withstate)
        r = net(ml, single_hist(ml, "tval"), tag);
        for k in ml.keys: ml.pool[k]["X"] = baseX[k]
        return r
    Sbest = {k: TVP.iloc[k - 1].reindex(ml.pool[k]["idx"]) for k in OOSK}
    def comp(k):
        idx = ml.pool[k]["idx"]; m = MOM.iloc[k - 1].reindex(idx).values
        cols = [z(m), z(RESM.iloc[k - 1].reindex(idx).values), z(HI.iloc[k - 1].reindex(idx).values),
                z(MACD.iloc[k - 1].reindex(idx).values), z(TVP.iloc[k - 1].reindex(idx).values), z(z(m) - z(IDd.iloc[k - 1].reindex(idx).values))]
        return pd.Series(np.nanmean(np.vstack(cols), axis=0), index=idx)
    Scomp = {k: comp(k) for k in OOSK}
    print("\n-- Stage 3: model-over-signals vs best-single & equal-blend (must beat BOTH) --"); print(hdr, flush=True)
    net(ml, Sbest, "S3 best_single(tvp)")
    net(ml, Scomp, "S3 equal_blend")
    model_over_signals(False, "S3 model_signals")
    model_over_signals(True,  "S3 model_sig+state")


def _weights_from_scores(S, pool, pnl_cols, sm, VOL, *, weighting="equal", band=None, decile=0.10, ls=False, short_ok=None):
    """SHARED sleeve-agnostic construction prototype: scores dict k→Series  →  weights DataFrame.
    weighting ∈ equal|mdv|sqrt_mdv|volparity (liquidity/risk sizing). band=(enter_q,exit_q) buy/hold hysteresis (NMV).
    Any sleeve (MOM/DM/MR) plugs its own S in — this is the candidate CONSTRUCT layer."""
    def wt(sel, k):
        sel = list(sel)
        if weighting == "mdv":       w = sm.iloc[k - 1].reindex(sel).clip(lower=0)
        elif weighting == "sqrt_mdv": w = np.sqrt(sm.iloc[k - 1].reindex(sel).clip(lower=0))
        elif weighting == "volparity": w = 1.0 / (VOL.iloc[k - 1].reindex(sel) + 1e-9)
        else:                        w = pd.Series(1.0, index=sel)
        w = w.fillna(0.0); s = w.sum(); return w / s if s > 0 else pd.Series(0.0, index=sel)
    heldL, heldS, rows = set(), set(), {}
    for k in sorted(S):
        s = S[k].dropna();
        if len(s) < 20: continue
        rlong = s.rank(ascending=False, pct=True)                     # 0 = best (buy)
        if band:
            en, ex = band; topL = set(rlong[rlong <= en].index) | (heldL & set(rlong[rlong <= ex].index))
        else:
            topL = set(s.nlargest(max(1, int(len(s) * decile))).index)
        topL = [t for t in topL if t in s.index]; heldL = set(topL)
        w = wt(topL, k)
        if ls and short_ok is not None:                               # dollar-neutral short leg, borrowable only
            sh = pd.Series(short_ok[k], index=pool[k]["idx"]).reindex(s.index).fillna(False)
            cand = s[sh.values]; rsh = cand.rank(ascending=True, pct=True)
            if band: botS = set(rsh[rsh <= en].index) | (heldS & set(rsh[rsh <= ex].index))
            else: botS = set(cand.nsmallest(max(1, int(len(cand) * decile))).index)
            botS = [t for t in botS if t in cand.index]; heldS = set(botS)
            ws = wt(botS, k); w = w.sub(ws, fill_value=0.0)           # long +, short −
        rows[pool[k]["dt"]] = w
    return pd.DataFrame(rows).T.reindex(columns=pnl_cols)


def _construction_ab():
    # SHARED CONSTRUCTION A/B (user 2026-07-24: "this should work for all our sleeves"). One tval score, full universe;
    # vary weighting × banding. Validates the Stage-0 sizing win + tests banding (NMV's #1 cost technique) before we
    # promote this to a sleeve-agnostic CONSTRUCT.py that MOM/DM/MR all route through.
    ml = build(tier="liquid", pool=True); pm, rm, sm = ml.pm, ml.rm, ml.sm
    S = single_hist(ml, "tval")                                       # ← any sleeve's scores plug in here
    VOL = rm.rolling(6, min_periods=4).std()
    tc = BACKTEST.tiered_transaction_costs(sm); bf = BACKTEST.tiered_borrow_fees(sm)
    def run(weighting, band, tag):
        W = _weights_from_scores(S, ml.pool, ml.pnl.columns, sm, VOL, weighting=weighting, band=band)
        r = BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
        print(f"  {tag:26}{r['sharpe']:>7.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}", flush=True)
        return r
    print(f"\n[CONSTRUCTION A/B] full liquid · tval champion · long-decile · net · seeds={ml.seeds}", flush=True)
    print(f"  {'weighting × band':26}{'netSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    run("equal", None, "equal / no-band")
    run("mdv", None, "mdv / no-band")
    run("sqrt_mdv", None, "sqrt_mdv / no-band")
    run("volparity", None, "volparity / no-band")
    run("mdv", (0.10, 0.20), "mdv / band(.10,.20)")
    run("sqrt_mdv", (0.10, 0.20), "sqrt_mdv / band(.10,.20)")
    run("sqrt_mdv", (0.10, 0.30), "sqrt_mdv / band(.10,.30)")


def _unified_mh():
    # UNIFIED MULTI-HORIZON tval model + return-reclassification (user 2026-07-24). MECE, construction FIXED at
    # CONSTRUCT(mdv,band). ONE multi_output tree over [tval@6, r1, r2, r3] → read all arms off its bucket-probs.
    #  AXIS 1 (horizon): does a short-{1,2,3} readout blended with the 6mo tval offense add net? (keep-DM decision)
    #  AXIS 2 (centers): gauss → static-return → ROLLING-return μₖ; crash-adaptive check in crash months. (idea 2)
    import CONSTRUCT
    ml = build(tier="liquid", pool=True); pm = ml.pm
    for k in ml.keys:                                              # add short-horizon fwd single-month returns
        for h in (1, 2, 3):
            ml.pool[k][f"r{h}"] = (pm.iloc[k + h] / pm.iloc[k + h - 1] - 1).reindex(ml.pool[k]["idx"]).values
    tc = BACKTEST.tiered_transaction_costs(ml.sm); bf = BACKTEST.tiered_borrow_fees(ml.sm)
    def book(score, tag, weighting="mdv", band=(0.10, 0.20)):
        S = pd.DataFrame({ml.pool[k]["dt"]: score[k] for k in score}).T
        W = CONSTRUCT.construct(S, ml.hub, tier="liquid", weighting=weighting, band=band)
        r = BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
        print(f"  {tag:24}{r['sharpe']:>7.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}", flush=True)
        return r
    # ── ONE multi_output tree over [tval, r1, r2, r3] → keep per-field bucket-prob distributions ──
    fields = ["tval", "r1", "r2", "r3"]; pool, mdl, PQ = ml.pool, None, {}
    for k in _oos(ml):
        if _refit(ml, k, mdl, months=(1,)):
            tr = _tr(ml, k)
            if len(tr) >= 24:
                Xl, Yl = [], []
                for j in tr:
                    m = _finite(*[pool[j][f] for f in fields])
                    if m.sum() < 50: continue
                    Xl.append(pool[j]["X"][m]); Yl.append(np.hstack([soft(pool[j][f][m], KB) for f in fields]))
                mdl = XGBRegressor(**REGC, multi_strategy="multi_output_tree", objective="binary:logistic", random_state=0).fit(np.vstack(Xl), np.vstack(Yl))
        if mdl is not None:
            Q = mdl.predict(pool[k]["X"]).reshape(len(pool[k]["idx"]), len(fields), KB); Q = Q / (Q.sum(2, keepdims=True) + 1e-9)
            PQ[k] = {f: Q[:, i, :] for i, f in enumerate(fields)}
    idxof = {k: pool[k]["idx"] for k in PQ}
    def ret_centers(k, roll=None):                                # μₖ = causal per-tval-bucket mean fwd6 return (static | rolling)
        tr = _tr(ml, k) if roll is None else _tr(ml, k)[-roll:]
        sums = np.zeros(KB); cnts = np.zeros(KB)
        for j in tr:
            v = pool[j]["tval"]; rr = pool[j]["fwd6"]; m = _finite(v, rr)
            if m.sum() == 0: continue
            b = _bkt(v[m], KB); sums += np.bincount(b, weights=rr[m], minlength=KB); cnts += np.bincount(b, minlength=KB)
        return np.where(cnts > 0, sums / np.maximum(cnts, 1), 0.0)
    O_g  = {k: pd.Series(PQ[k]["tval"] @ CEN, index=idxof[k]) for k in PQ}                       # offense, gaussian (baseline)
    SH   = {k: pd.Series(np.mean([grank(PQ[k][f] @ CEN) for f in ("r1", "r2", "r3")], axis=0), index=idxof[k]) for k in PQ}  # short
    O_sr = {k: pd.Series(PQ[k]["tval"] @ ret_centers(k, None), index=idxof[k]) for k in PQ}      # static return centers
    O_rr = {k: pd.Series(PQ[k]["tval"] @ ret_centers(k, 12), index=idxof[k]) for k in PQ}        # rolling(12m) return centers
    print(f"\n[UNIFIED MH] full liquid ~{int(np.mean([len(idxof[k]) for k in idxof])) } names · CONSTRUCT(mdv,band) · net · seeds=1")
    print(f"  {'arm':24}{'netSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}")
    print("-- AXIS 1: HORIZON — does short {1,2,3} add to 6mo tval offense? --")
    rO = book(O_g, "O tval@6 (baseline)"); rS = book(SH, "S short{1,2,3}")
    bl = pd.DataFrame({"O": rO["returns"], "S": rS["returns"]}).dropna(); cOS = bl.corr().iloc[0, 1]
    blend = 0.5 * bl["O"] + 0.5 * bl["S"]; blSR = blend.mean() / blend.std() * np.sqrt(12)
    print(f"  {'O+S blend 50/50':24}{blSR:>7.2f}                        corr(O,S)={cOS:+.2f}  (bar: blend ≥ O+0.10 & corr<0.8)")
    print("-- AXIS 2: RECLASSIFICATION centers — gauss vs static-ret vs ROLLING-ret (crash-adaptive) --")
    rB0 = book(O_g, "B0 gauss (baseline)"); rB1 = book(O_sr, "B1 static-ret"); rB2 = book(O_rr, "B2 rolling-ret12")
    al = pd.DataFrame({"b0": rB0["returns"], "b2": rB2["returns"]}).dropna()
    cr = al[al["b0"] <= al["b0"].quantile(0.10)].index                                          # momentum-crash months
    print(f"  crash months (worst-10% of B0, n={len(cr)}): B0 {al.loc[cr,'b0'].mean():+.2%}/mo  B2 {al.loc[cr,'b2'].mean():+.2%}/mo"
          f"  Δ(B2−B0) {(al.loc[cr,'b2']-al.loc[cr,'b0']).mean():+.2%}  (bar: B2 loses less in crashes)")


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# 2026-07-25 EXPERIMENTS — T42 signatures / T43 cost-aware objective / T44 beta management
# Motivated by T41: the sleeve has ZERO CAPM alpha at beta ~0.9, and rankIC rises while net SR falls.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════

def _sig_panels(hub, me, cols):
    """PATH-SIGNATURE features (T42). The champion target `tval` is a normalised LEVEL-1 signature term of the
    log-price path: it is invariant to the ORDER of the moves. Level-2 signature terms are exactly the order
    information it discards. We compute the tractable, economically-named level-2 coordinates on DAILY paths:

      sig_tasym{W}   time-asymmetry  ∫(s−s₀)dX  centred & vol-normalised. Distinguishes "rose then flat" from
                     "flat then rose" — SAME cumulative return, SAME tval, opposite forward distribution.
                     (This is the signature-space generalisation of Schmerling's centre-weighted trend.)
      sig_levm{W}    Lévy area of the 2-D path (stock, market) = ½∮(XdM − MdX). Sign = did the name LEAD or LAG
                     the market through the window. Pure ordering info; zero correlation with either increment.
      sig_levv{W}    Lévy area of (log-price, log dollar-volume) = did volume arrive BEFORE or AFTER the move.
      sig_snr{W}     ΔX / √(W·Σ dX²) = daily-resolution trend t-stat (the champion's tval computed on 126 daily
                     points instead of 7 monthly ones — a strictly finer estimate of the same object).

    All are rolling-sum identities, so this is fully vectorised:  ∫(X−X₀)dY over a window
    = rollsum(X·dY) − X_{t−W}·rollsum(dY).  Every value at month-end m uses only data through m (leak-free);
    the pool then reads them at iloc[t−1], so they enter with the same 1-month lag as every other feature.
    """
    px = hub.px_d.reindex(columns=cols); dv = (px * hub.vol_d.reindex(columns=cols))
    X = np.log(px.where(px > 0))
    M = np.log((1 + hub.ret_d.reindex(columns=cols).mean(axis=1).fillna(0)).cumprod())   # equal-wt market path
    V = np.log1p(dv.clip(lower=0))
    dX = X.diff(); dV = V.diff(); dM = M.diff()
    n = pd.Series(np.arange(len(X), dtype=float), index=X.index)
    out = {}
    for W in (63, 126):
        rs = lambda D: D.rolling(W, min_periods=W // 2).sum()
        sdX = rs(dX); qv = rs(dX ** 2)
        sX = dX.rolling(W, min_periods=W // 2).std(); sM = dM.rolling(W, min_periods=W // 2).std()
        sV = dV.rolling(W, min_periods=W // 2).std()
        # time-asymmetry: ∫(s−s₀)dX − (W/2)∫dX , vol-normalised
        tw = rs(dX.mul(n, axis=0)).sub(n.shift(W) * sdX, axis=0)
        out[f"sig_tasym{W}"] = (tw - (W / 2.0) * sdX) / (sX * W ** 1.5 + 1e-12)
        # Lévy area vs market
        a1 = rs(dM.mul(X.shift(1))).sub(X.shift(W) * rs(dM), axis=0)
        a2 = rs(dX.mul(M.shift(1), axis=0)).sub(M.shift(W) * rs(dX), axis=0)
        out[f"sig_levm{W}"] = 0.5 * (a1 - a2) / (sX * sM * W + 1e-12)
        # Lévy area vs volume
        b1 = rs(dV.mul(X.shift(1))).sub(X.shift(W) * rs(dV), axis=0)
        b2 = rs(dX.mul(V.shift(1))).sub(V.shift(W) * rs(dX), axis=0)
        out[f"sig_levv{W}"] = 0.5 * (b1 - b2) / (sX * sV * W + 1e-12)
        # daily-resolution trend t-stat
        out[f"sig_snr{W}"] = sdX / (np.sqrt(qv * W) + 1e-12)
    return {k: v.reindex(me, method="ffill").reindex(columns=cols).replace([np.inf, -np.inf], np.nan)
            for k, v in out.items()}


def _build_with(extra=None, **kw):
    """MomLayer whose TSF feature dict is augmented BEFORE the pool is built (so X carries the extra columns)."""
    seeds = int(os.environ.get("SEEDS", kw.pop("seeds", 1))); topn = os.environ.get("TOPN")
    topn = int(topn) if topn else kw.pop("topn", None); tier = os.environ.get("TIER", kw.pop("tier", "liquid"))
    ml = MomLayer(seeds=seeds, topn=topn, tier=tier, **kw)
    ml._prep()                                              # builds pm/rm/sm/TSF/target
    if extra:
        ml.TSF.update(extra)
    ml._prep = lambda: None                                 # already prepped — stop build_pool re-running it
    ml.build_pool()
    print(f"[pool] tier={tier} · {len(ml.keys)} months · {len(ml.feat_names)} feats", flush=True)
    return ml



def _xattn_panels(ml):
    """HAND-ROLLED CROSS-SECTIONAL ATTENTION for trees (T47).

    A tree cannot see other rows, so learned attention over the cross-section is impossible in XGBoost.
    But attention's FIRST-ORDER effect is a weighted aggregate of peers — and a FIXED aggregation pattern
    can be precomputed as features. That is exactly what Han's M_MOM features are (cross-sectional MEANS of
    each momentum, broadcast to every stock = rank-1 mean-field attention), and DM.py showed we never had
    them. Here we give the tree the aggregation an attention layer would otherwise learn:

      xs_mean{w} / xs_disp{w}   cross-sectional mean and dispersion of w-month momentum  (Han's M_MOM + breadth)
      sec_rel{w}                stock momentum MINUS its SECTOR mean  (peer-group attention, hub.sector)
      sec_mean{w}               the sector mean itself
      siz_rel{w}                stock momentum MINUS its SIZE-decile mean (peer group = size cohort)
      xs_skew{w}                cross-sectional skew — the bimodality/crowding shape Han's Fig.1 is about

    If these help, "attention" on this problem is mostly mean-field aggregation and trees can have it for free.
    If they do not, learned attention has to earn its keep on higher-order interactions.
    """
    pm, sm = ml.pm, ml.sm
    sec = getattr(ml.hub, "sector", None)
    out = {}
    for w in (1, 6, 12):
        M = pm / pm.shift(w) - 1.0
        M = M.where(M.abs() < 5)
        mu = M.mean(axis=1); sd = M.std(axis=1)
        out[f"xs_mean{w}"] = pd.DataFrame(np.repeat(mu.values[:, None], M.shape[1], 1), index=M.index, columns=M.columns)
        out[f"xs_disp{w}"] = pd.DataFrame(np.repeat(sd.values[:, None], M.shape[1], 1), index=M.index, columns=M.columns)
        out[f"xs_skew{w}"] = pd.DataFrame(np.repeat(M.skew(axis=1).values[:, None], M.shape[1], 1), index=M.index, columns=M.columns)
        if sec is not None:
            g = sec.reindex(M.columns)
            secmean = M.T.groupby(g).transform("mean").T                 # peer-group (sector) mean
            out[f"sec_mean{w}"] = secmean
            out[f"sec_rel{w}"] = M - secmean
        szdec = sm.rank(axis=1, pct=True).mul(10).round().clip(1, 10)    # size cohort
        szmean = pd.DataFrame(np.nan, index=M.index, columns=M.columns)
        for d in range(1, 11):
            msk = szdec == d
            rowmean = M.where(msk).mean(axis=1)
            szmean = szmean.where(~msk, pd.DataFrame(np.repeat(rowmean.values[:, None], M.shape[1], 1),
                                                     index=M.index, columns=M.columns))
        out[f"siz_rel{w}"] = M - szmean
    return {k: v.replace([np.inf, -np.inf], np.nan) for k, v in out.items()}


def _t47_xattn():
    """T47 — can XGBoost have 'cross-sectional attention' as precomputed aggregation features?"""
    print("\n[T47 HAND-ROLLED CROSS-SEC ATTENTION] champion ± peer-aggregate features · long-decile · net", flush=True)
    base = _build_with(None)
    X = _xattn_panels(base)
    print(f"  added {len(X)} panels: {', '.join(sorted(X))}", flush=True)
    aug = _build_with(X)
    print(f"\n{'arm':26}{'IC6m':>8}{'net SR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    net(base, single_hist(base, "tval"), f"base ({len(base.feat_names)}f)")
    net(aug,  single_hist(aug,  "tval"), f"+xattn ({len(aug.feat_names)}f)")
    k0 = [k for k in aug.keys if aug.pool[k]["dt"].year >= 2011][:1]
    tr = _tr(aug, k0[0]) if k0 else []
    if tr:
        Xt = np.vstack([aug.pool[j]["X"] for j in tr]); yl = np.concatenate([_bkt(aug.pool[j]["tval"], KB) for j in tr])
        m = XGBClassifier(**REGC, objective="multi:softprob", num_class=KB, random_state=0).fit(Xt, yl)
        imp = pd.Series(m.feature_importances_, index=aug.feat_names).sort_values(ascending=False)
        new = [c for c in imp.index if c.startswith(("xs_", "sec_", "siz_"))]
        print(f"\n  attention-feature share of importance: {imp[new].sum():.1%}")
        print(f"  top-10 overall: {', '.join(f'{i}={v:.3f}' for i, v in imp.head(10).items())}")


def _t42_signature():
    """T42 — do LEVEL-2 path-signature coordinates add over the champion's level-1 (tval) feature set?"""
    print("\n[T42 PATH SIGNATURES] champion features ± level-2 signature coordinates · long-decile · net", flush=True)
    base = _build_with(None)
    from DATAHUB import DataHub
    SIG = _sig_panels(base.hub, base.rm.index, base.rm.columns)
    print(f"  signature panels: {', '.join(SIG)}", flush=True)
    aug = _build_with(SIG)
    print(f"\n{'arm':22}{'IC6m':>8}{'net SR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    rB = net(base, single_hist(base, "tval"), f"base ({len(base.feat_names)}f)")
    rA = net(aug,  single_hist(aug,  "tval"), f"+signature ({len(aug.feat_names)}f)")
    # is the ADDED information used, and is it orthogonal to the level-1 features?
    k0 = [k for k in aug.keys if aug.pool[k]["dt"].year >= 2011][:1]
    tr = _tr(aug, k0[0]) if k0 else []
    if tr:
        Xt = np.vstack([aug.pool[j]["X"] for j in tr]); yl = np.concatenate([_bkt(aug.pool[j]["tval"], KB) for j in tr])
        m = XGBClassifier(**REGC, objective="multi:softprob", num_class=KB, random_state=0).fit(Xt, yl)
        imp = pd.Series(m.feature_importances_, index=aug.feat_names).sort_values(ascending=False)
        print(f"\n  top-12 feature importance (first fit): {', '.join(f'{i}={v:.3f}' for i, v in imp.head(12).items())}")
        print(f"  signature share of total importance: {imp[[c for c in imp.index if c.startswith('sig_')]].sum():.1%}")
    # signature-ONLY arm: is there standalone information, or only interaction?
    return rB, rA


def _t43_cost_objective():
    """T43 — is the LOSS optimising the wrong object? T41 found rankIC↑ / netSR↓ three times over.
    MECE: hold FEATURES + POOL fixed, vary ONLY the objective. Arms C/D share one torch architecture so the
    comparison isolates the OBJECTIVE from the ARCHITECTURE (the flaw in T29-Stage-1, which used a tree proxy).
      A  champion  XGB histogram cross-entropy on tval            (rank-quality loss)
      B  A + liquidity SAMPLE WEIGHTS                             (same loss, capacity spent where we trade)
      C  torch MLP, DIFFERENTIABLE NET-SHARPE                     (softmax book − turnover cost → −Sharpe)
      D  torch MLP, same net, rank/MSE loss                       (architecture control for C)
    """
    import torch, torch.nn as nn
    torch.manual_seed(0)
    print("\n[T43 COST-AWARE OBJECTIVE] features & pool FIXED · only the loss varies · long-decile · net", flush=True)
    ml = _build_with(None); pool = ml.pool
    print(f"\n{'arm':26}{'IC6m':>8}{'net SR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)

    rA = net(ml, single_hist(ml, "tval"), "A champion hist-CE")

    # --- B: liquidity sample weights -----------------------------------------------------------------
    def hist_w(field):
        out, mdl = {}, None
        for k in _oos(ml):
            if _refit(ml, k, mdl):
                tr = _tr(ml, k)
                if len(tr) >= 24:
                    Xl, yl, wl = [], [], []
                    for j in tr:
                        m = _finite(pool[j][field])
                        Xl.append(pool[j]["X"][m]); yl.append(_bkt(pool[j][field][m], KB))
                        sz = ml.sm.iloc[j - 1].reindex(pool[j]["idx"]).values[m]
                        wl.append(pd.Series(sz).rank(pct=True).fillna(0.5).values)      # ∝ liquidity rank
                    mdl = [XGBClassifier(**REGC, objective="multi:softprob", num_class=KB, random_state=s)
                           .fit(np.vstack(Xl), np.concatenate(yl), sample_weight=np.concatenate(wl))
                           for s in range(ml.seeds)]
            if mdl is not None:
                P = np.mean([m.predict_proba(pool[k]["X"]) for m in mdl], axis=0)
                out[k] = pd.Series(P @ CEN, index=pool[k]["idx"])
        return out
    rB = net(ml, hist_w("tval"), "B + liquidity weights")

    # --- C/D: one torch architecture, two objectives ---------------------------------------------------
    NF = len(ml.feat_names)
    def mknet():
        return nn.Sequential(nn.Linear(NF, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def torch_run(objective):
        out, net_, prev = {}, None, None
        for k in _oos(ml):
            if _refit(ml, k, net_):
                tr = _tr(ml, k)
                if len(tr) >= 24:
                    net_ = mknet(); opt = torch.optim.Adam(net_.parameters(), lr=1e-3, weight_decay=1e-4)
                    # per-month tensors so the Sharpe loss can be formed on real cross-sections
                    Ms = []
                    for j in tr:
                        m = _finite(pool[j]["tval"], pool[j]["fwd6"])
                        if m.sum() < 50: continue
                        Xj = np.nan_to_num(pool[j]["X"][m], nan=0.0, posinf=0.0, neginf=0.0)
                        cost = BACKTEST.tiered_transaction_costs(ml.sm).iloc[j - 1].reindex(pool[j]["idx"]).values[m] \
                               if hasattr(BACKTEST.tiered_transaction_costs(ml.sm), "iloc") else np.zeros(int(m.sum()))
                        Ms.append((torch.tensor(Xj, dtype=torch.float32),
                                   torch.tensor(grank(pool[j]["tval"][m]), dtype=torch.float32),
                                   torch.tensor(np.nan_to_num(pool[j]["fwd6"][m]), dtype=torch.float32),
                                   torch.tensor(np.nan_to_num(cost, nan=0.002), dtype=torch.float32)))
                    for ep in range(40):
                        opt.zero_grad(); rets, loss_r = [], 0.0
                        for Xj, yj, fj, cj in Ms:
                            s = net_(Xj).squeeze(-1)
                            if objective == "sharpe":
                                w = torch.softmax(s * 8.0, dim=0)             # differentiable top-weighted book
                                rets.append((w * fj).sum() - (w * cj).sum())  # net of per-name trading cost
                            else:
                                loss_r = loss_r + ((s - yj) ** 2).mean()
                        if objective == "sharpe":
                            R = torch.stack(rets); loss = -(R.mean() / (R.std() + 1e-6))
                        else:
                            loss = loss_r / max(len(Ms), 1)
                        loss.backward(); opt.step()
            if net_ is not None:
                with torch.no_grad():
                    Xk = torch.tensor(np.nan_to_num(pool[k]["X"], nan=0.0, posinf=0.0, neginf=0.0), dtype=torch.float32)
                    out[k] = pd.Series(net_(Xk).squeeze(-1).numpy(), index=pool[k]["idx"])
        return out

    rC = net(ml, torch_run("sharpe"), "C torch NET-SHARPE")
    rD = net(ml, torch_run("rank"),   "D torch rank-MSE (ctrl)")
    print("\n  C vs D isolates the OBJECTIVE (same net, same features); A vs C is objective+architecture.")
    return rA, rB, rC, rD


def _t44_beta():
    """T44 — T41 showed the sleeve is a 0.9-beta carrier with 0.00% alpha. So MANAGE THE BETA.
    Overlays applied to the champion long book's NET monthly returns; every dial is PIT (lagged one month).
      A base                    champion long-decile, mdv + band (production construction)
      B static hedge            r − β̂ₜ₋₁·SPYₜ ,  β̂ = trailing 36m OLS beta of the book
      C STATE-timed hedge       hedge only when the macro stress PC is elevated (credit + VIX, PIT)
      D vol-managed             scale exposure to a 12% vol target on trailing 12m realised vol (Barroso-
                                Santa-Clara / Moreira-Muir), capped [0,2]
      E four-state exposure     Goulding-Harvey-Mazzoleni: fast(1m) vs slow(12m) MARKET trend agreement →
                                Bull(+,+)=1.0  Correction(−,+)=0.5  Bear(−,−)=0.5  Rebound(+,−)=1.0
                                (fixed prior exposures — NOT fitted in-sample)
      F  B + D                  hedge and vol-manage
    Judged on ALPHA and IR vs SPY, not on SR — that was T41's whole lesson.
    """
    print("\n[T44 BETA MANAGEMENT] overlays on the champion long book · net · judged vs SPY", flush=True)
    ml = _build_with(None)
    S = single_hist(ml, "tval")
    VOL = ml.rm.rolling(6, min_periods=4).std()
    W = _weights_from_scores(S, ml.pool, ml.pnl.columns, ml.sm, VOL, weighting="mdv", band=(0.10, 0.20))
    tc = BACKTEST.tiered_transaction_costs(ml.sm); bf = BACKTEST.tiered_borrow_fees(ml.sm)
    r0 = BACKTEST.backtest(W, ml.pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
    r = r0["returns"].dropna(); r = r[r.index >= W.index.min()]
    r.index = pd.PeriodIndex(r.index, freq="M")

    spy = ml.hub.spy_m.copy()
    spy = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
    spy.index = pd.PeriodIndex(spy.index, freq="M"); spy = spy[~spy.index.duplicated()]
    D = pd.concat({"r": r, "m": spy}, axis=1).dropna()
    if len(D) < 60: print("  [!] too few aligned months"); return

    beta = D.r.rolling(36, min_periods=24).cov(D.m) / D.m.rolling(36, min_periods=24).var()
    beta = beta.shift(1).fillna(1.0).clip(0, 2)                          # PIT: known at t−1
    rv   = D.r.rolling(12, min_periods=6).std().shift(1)
    volsc = (0.12 / (rv * np.sqrt(12) + 1e-9)).clip(0, 2).fillna(1.0)

    mac = ml.hub.macro_m
    if mac is not None and {"credit", "vix"} <= set(mac.columns):
        mm = mac.copy(); mm.index = pd.PeriodIndex(mm.index, freq="M"); mm = mm[~mm.index.duplicated()]
        z = ((mm[["credit", "vix"]] - mm[["credit", "vix"]].expanding(36).mean())
             / (mm[["credit", "vix"]].expanding(36).std() + 1e-9)).mean(axis=1)   # expanding = PIT stress PC
        stress = z.reindex(D.index).shift(1).fillna(0.0)
    else:
        stress = pd.Series(0.0, index=D.index)
    hedge_on = (stress > 0.5).astype(float)                              # elevated-stress dial

    lvl = (1 + D.m).cumprod()
    fast = np.sign(D.m); slow = np.sign(lvl / lvl.shift(12) - 1)
    fast, slow = fast.shift(1).fillna(1), slow.shift(1).fillna(1)        # PIT
    expo = pd.Series(1.0, index=D.index)
    expo[(fast > 0) & (slow > 0)] = 1.0                                  # Bull
    expo[(fast < 0) & (slow > 0)] = 0.5                                  # Correction
    expo[(fast < 0) & (slow < 0)] = 0.5                                  # Bear
    expo[(fast > 0) & (slow < 0)] = 1.0                                  # Rebound

    arms = {
        "A base":              D.r,
        "B static hedge":      D.r - beta * D.m,
        "C STATE-timed hedge": D.r - beta * hedge_on * D.m,
        "D vol-managed":       volsc * D.r,
        "E four-state expo":   expo * D.r,
        "F  B + D":            volsc * (D.r - beta * D.m),
    }
    print(f"\n{'arm':22}{'ann':>8}{'vol':>8}{'SR':>7}{'maxDD':>9}{'beta':>7}{'alpha':>9}{'t(a)':>7}{'IR':>7}", flush=True)
    for tag, x in list(arms.items()) + [("   SPY", D.m)]:
        x = x.dropna(); dd = ((1 + x).cumprod() / (1 + x).cumprod().cummax() - 1).min()
        mm2 = D.m.reindex(x.index)
        X = np.c_[np.ones(len(x)), mm2.values]
        c, *_ = np.linalg.lstsq(X, x.values, rcond=None); e = x.values - X @ c
        se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(x) - 2, 1)))
        act = x - c[1] * mm2
        print(f"  {tag:20}{x.mean()*12:>8.2%}{x.std()*np.sqrt(12):>8.2%}"
              f"{x.mean()*12/(x.std()*np.sqrt(12)+1e-9):>7.2f}{dd:>9.1%}{c[1]:>7.2f}"
              f"{c[0]*12:>+9.2%}{c[0]/(se[0]+1e-12):>7.2f}{act.mean()*12/(act.std()*np.sqrt(12)+1e-9):>7.2f}", flush=True)
    print("\n  Judge on alpha/IR. SR alone is what made a 0.9-beta index tracker look like a sleeve (T41).")


if __name__ == "__main__":
    _stg = os.environ.get("STAGES")
    if _stg:                                                  # STAGES set → run that experiment only, then exit
        {"FAC": _facmom_deep, "VETO": _hist_veto, "RISKLS": _risk_ls, "HAN": _han_short, "MHSHORT": _mh_short, "DECOMP": _decomp, "TWO": _twomodel, "UNI": _unified_mh, "CONS": _construction_ab, "123": _stages123, "0": _stage0,
         "SIG": _t42_signature, "XATTN": _t47_xattn, "COST": _t43_cost_objective, "BETA": _t44_beta}[_stg]()
        sys.exit(0)
    # (default, no STAGES) HORSE RACE — signals + a DISCIPLINED equal-weight combiner match/beat the per-sleeve ML? idea ③
    # (residual momentum as the real offense), frog-in-the-pan (Da-Gurun-Warachka info-discreteness), and
    # the "drop ML / let the portfolio sort out signals" thesis. Also emits the redundancy audit: score-corr
    # + book-return-corr matrices (are MOM/DM/these signals the same bet?). Long-only decile, net, top-1000 screen.
    ml = build(tier="liquid", pool=True)                     # pool=True → resmom + macd available; env SEEDS/TOPN
    pm, rm = ml.pm, ml.rm; OOSK = _oos(ml)
    def z(x):                                                # cross-sectional z (nan-safe)
        x = np.asarray(x, float); m, s = np.nanmean(x), np.nanstd(x); return (x - m) / (s + 1e-9)

    # ── raw signal DataFrames (dates × names), all read at iloc[k-1] = last fully-known month (no look-ahead) ──
    MOM  = pm / pm.shift(11) - 1                              # 11-month price momentum (skips trade month by reading k-1)
    RESM = ml.POOLF["resmom"]                                # residual (idiosyncratic) momentum — the certified orthogonal alpha
    HI52 = ml.TSF["hi52"]                                    # 52-week-high proximity (George-Hwang)
    MACD = ml.POOLF["macd"]                                  # Baz MACD trend composite
    lp = np.log(pm.values); nP = 7; xc = np.arange(nP) - (nP - 1) / 2; Sxx = (xc ** 2).sum()   # past 6m tval (slope/SE)
    TVP = pd.DataFrame(np.nan, index=pm.index, columns=pm.columns)
    for t in range(nP - 1, len(pm)):
        Y = lp[t - nP + 1:t + 1]; Ym = np.nanmean(Y, 0); sl = np.nansum(xc[:, None] * (Y - Ym), 0) / Sxx
        resd = Y - (Ym + xc[:, None] * sl); se = np.sqrt(np.nansum(resd ** 2, 0) / max(nP - 2, 1) / Sxx)
        TVP.iloc[t] = sl / (se + 1e-9)
    fpos = (rm > 0).rolling(11, min_periods=8).mean(); fneg = (rm < 0).rolling(11, min_periods=8).mean()
    IDd  = np.sign(MOM) * (fneg - fpos)                      # info-discreteness: LOW (neg) ID = continuous ("frog in the pan")

    RAW = {"mom11": MOM, "resmom": RESM, "hi52": HI52, "macd": MACD, "tvalpast": TVP}   # each: higher = more bullish
    def dfsig(DF): return {k: DF.iloc[k - 1].reindex(ml.pool[k]["idx"]) for k in OOSK}
    S = {nm: dfsig(DF) for nm, DF in RAW.items()}
    # frog-in-the-pan momentum = z(mom) − z(ID) within the cross-section (buy continuous winners)
    S["fip"] = {k: pd.Series(z(MOM.iloc[k - 1].reindex(ml.pool[k]["idx"]).values)
                             - z(IDd.iloc[k - 1].reindex(ml.pool[k]["idx"]).values), index=ml.pool[k]["idx"]) for k in OOSK}
    # EQUAL-WEIGHT composite = mean z over all 6 signals (the disciplined "let the portfolio sort it out")
    parts = ["mom11", "resmom", "hi52", "macd", "tvalpast", "fip"]
    S["composite"] = {k: pd.Series(np.nanmean(np.vstack([z(S[p][k].values) for p in parts]), axis=0),
                                   index=ml.pool[k]["idx"]) for k in OOSK}
    # ML comparator = the actual champion estimator (tval histogram → RET) + return-target DM
    S["ML_tval"] = single_hist(ml, "tval"); S["ML_ret"] = single_hist(ml, "fwd6")

    order = parts + ["composite", "ML_tval", "ML_ret"]
    print(f"\n[HORSE RACE] tier=liquid · {len(ml.feat_names)} feats · long-only decile · net · seeds={ml.seeds}", flush=True)
    print(f"{'book':22}{'IC6m':>8}{'net SR':>8}{'ann':>8}{'maxDD':>8}{'turn':>7}", flush=True)
    rets = {}
    for nm in order:
        r = net(ml, S[nm], nm); rets[nm] = r["returns"]

    # ── REDUNDANCY AUDIT 1: book net-return correlation (are these the same bet in $?) ──
    RB = pd.DataFrame(rets).dropna()
    print("\n[book-return corr]"); print(RB.corr().round(2).to_string())
    # ── REDUNDANCY AUDIT 2: cross-sectional score overlap (avg monthly Spearman) ──
    keys = sorted(set.intersection(*[set(S[nm]) for nm in order]))
    C = pd.DataFrame(index=order, columns=order, dtype=float)
    for i, a in enumerate(order):
        for b in order[i:]:
            cs = [spearmanr(S[a][k].reindex(S[b][k].index).values, S[b][k].values, nan_policy="omit").correlation for k in keys]
            C.loc[a, b] = C.loc[b, a] = np.nanmean(cs)
    print("\n[score cross-sec corr]"); print(C.astype(float).round(2).to_string())
