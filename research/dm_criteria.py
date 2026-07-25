#!/usr/bin/env python3
"""dm_criteria.py — search for a PROFITABLE momentum book from the Han-DM model, WITHOUT retraining.

Uses the saved class-probability matrix (CURRENT BEST/out/han_dm_prob.pkl) so every ranking criterion and
every construction can be compared on ONE fixed model. Three families of lever, motivated by T46 (gross alpha
+26% t=5.6, but ~85% paid away and zero gross alpha in the liquid universe):

A. RANKING CRITERION — "put Sharpe / cost IN the model"
   Return   (Han §3.3.4, our current default)  μ̂ = Σ ŷ_k μ̂_k
   Sharpe   (Han §3.3.5 — WE NEVER IMPLEMENTED THIS, and Han says it is the criterion for investors who
            "face short-sale constraints", i.e. exactly our long-biased situation). Law of total variance:
            σ̂² = Σ ŷ_k(σ̂²_k + μ̂²_k) − μ̂² ;  rank on μ̂/σ̂.
   PrDf1/5  (Han §3.3.3)  PD(h) = Σ_{k≤h} (ŷ_k − ŷ_{K+1−k})(K/2 + 1 − k)
   netRet   COST IN THE OBJECTIVE, at the ranking step: rank on μ̂ − round-trip cost of THIS name.
            The model then prefers cheap names among equal-alpha names — the correct place for cost in a
            cross-sectional ranker (cost in the LOSS was T43; Gârleanu-Pedersen puts it in the PORTFOLIO).
   netSharpe  (μ̂ − rt cost)/σ̂ — both levers at once.

B. HOLDING PERIOD — overlapping (Jegadeesh-Titman) portfolios. Form a book each month, hold h months,
   run h staggered tranches at 1/h capital. Turnover falls ~1/h. Han/our DM used h=1 (turnover 7.2).

C. BANDING — the buy/hold spread ((s,S) rule). Novy-Marx & Velikov (RFS 2016): the single most effective
   cost-mitigation technique, and "most anomalies with <50% one-sided monthly turnover generate significant
   net spreads; few with higher turnover do."

Run: python research/dm_criteria.py
"""
import warnings, sys, os, pickle
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np, pandas as pd, BACKTEST
from DATAHUB import DataHub

hub = DataHub(start="1990-01-01", min_days=0)
PROB = pickle.load(open("CURRENT BEST/out/han_dm_prob.pkl", "rb"))
pnl = hub.pnl("monthly"); sm = hub.dollar_size("monthly")
tc = BACKTEST.tiered_transaction_costs(sm); bf = BACKTEST.tiered_borrow_fees(sm)
shortable = hub.elig("shortable", "monthly"); liq = hub.elig("liquid", "monthly")
spy = hub.spy_m; spy = spy.iloc[:, 0] if isinstance(spy, pd.DataFrame) else spy
spy.index = pd.PeriodIndex(spy.index, freq="M"); spy = spy[~spy.index.duplicated()]
K = len(next(iter(PROB.values()))["mu"])


def _cost(dt, idx):
    """Per-name ROUND-TRIP proportional cost at dt (tiered, liquidity-dependent)."""
    if isinstance(tc, pd.DataFrame):
        row = tc.reindex(index=[dt]).iloc[0] if dt in tc.index else tc.iloc[-1]
        return 2.0 * row.reindex(idx).astype(float).fillna(row.median()).values
    return np.full(len(idx), 2.0 * float(tc))


def scores(kind):
    out = {}
    for k, d in PROB.items():
        P, mu, var = d["P"], d["mu"], d["var"]
        m = P @ mu
        if kind in ("sharpe", "netsharpe"):
            s2 = P @ (var + mu ** 2) - m ** 2
            sd = np.sqrt(np.clip(s2, 1e-12, None))
        if kind == "ret":        v = m
        elif kind == "sharpe":   v = m / sd
        elif kind == "netret":   v = m - _cost(d["dt"], d["idx"])
        elif kind == "netsharpe": v = (m - _cost(d["dt"], d["idx"])) / sd
        elif kind.startswith("prdf"):
            h = int(kind[4:]); w = np.array([(K / 2 + 1 - (i + 1)) for i in range(h)])
            v = ((P[:, :h] - P[:, K - h:][:, ::-1]) * w).sum(axis=1)
        out[d["dt"]] = pd.Series(v, index=d["idx"])
    return out


def book(S, dec=0.10, hold=1, band=None, uni=None, borrow=True):
    """Decile book with optional overlapping tranches (hold) and buy/hold banding."""
    dates = sorted(S)
    tranche, held_hi, held_lo = [], set(), set()
    W = {}
    for d in dates:
        s = S[d].dropna()
        if uni is not None and d in uni.index:
            keep = uni.loc[d]; s = s[s.index.intersection(keep[keep].index)]
        if len(s) < 50: continue
        r_hi = s.rank(ascending=False, pct=True)
        cand = s
        if borrow and d in shortable.index:
            sh = shortable.loc[d]; cand = s[s.index.isin(sh[sh].index)]
        if len(cand) < 20: continue
        r_lo = cand.rank(ascending=True, pct=True)
        if band:
            en, ex = band
            hi = set(r_hi[r_hi <= en].index) | (held_hi & set(r_hi[r_hi <= ex].index))
            lo = set(r_lo[r_lo <= en].index) | (held_lo & set(r_lo[r_lo <= ex].index))
        else:
            hi = set(s.nlargest(max(1, int(len(s) * dec))).index)
            lo = set(cand.nsmallest(max(1, int(len(cand) * dec))).index)
        hi = [t for t in hi if t in s.index]; lo = [t for t in lo if t in cand.index]
        if not hi or not lo: continue
        held_hi, held_lo = set(hi), set(lo)
        w = pd.Series(0.0, index=s.index)
        w[hi] = 1.0 / len(hi); w[lo] = w[lo] - 1.0 / len(lo)
        tranche.append(w); tranche = tranche[-hold:]                 # overlapping portfolios
        agg = pd.concat(tranche, axis=1).fillna(0.0).mean(axis=1)
        W[d] = agg
    return pd.DataFrame(W).T.reindex(columns=pnl.columns).fillna(0.0)


def rep(tag, W):
    if W.abs().sum().sum() == 0: print(f"  {tag:34} empty"); return
    res = {}
    for lbl, kw in (("g", dict(transaction_cost=0.0, borrow_fee=0.0)),
                    ("n", dict(transaction_cost=tc, borrow_fee=bf))):
        try:
            r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index), **kw)
            x = r["returns"].dropna(); x = x[x.index >= W.index.min()]
            x.index = pd.PeriodIndex(x.index, freq="M")
            D = pd.concat({"r": x, "m": spy}, axis=1).dropna()
            X = np.c_[np.ones(len(D)), D.m.values]; c, *_ = np.linalg.lstsq(X, D.r.values, rcond=None)
            e = D.r.values - X @ c
            se = np.sqrt(np.diag(np.linalg.inv(X.T @ X) * (e @ e) / max(len(D) - 2, 1)))
            eq = (1 + D.r).cumprod()
            res[lbl] = (D.r.mean() * 12, D.r.std() * np.sqrt(12), (eq / eq.cummax() - 1).min(),
                        c[0] * 12, c[0] / (se[0] + 1e-12), r["ann_turnover"])
        except Exception as ex:
            print(f"  {tag:34} FAILED {type(ex).__name__}"); return
    (ga, _, _, gal, gt, _), (na, nv, nd, nal, nt, turn) = res["g"], res["n"]
    print(f"  {tag:34}{ga:>8.1%}{gal:>+9.2%}{gt:>6.2f} |{na:>9.1%}{nv:>7.1%}"
          f"{na/(nv+1e-9):>6.2f}{nd:>8.1%}{nal:>+9.2%}{nt:>6.2f}{turn:>7.1f}", flush=True)


HDR = (f"  {'book':34}{'gANN':>8}{'gALPHA':>9}{'t':>6} |{'nANN':>9}{'nVOL':>7}"
       f"{'nSR':>6}{'nDD':>8}{'nALPHA':>9}{'t':>6}{'turn':>7}")
if __name__ == "__main__":
    print(f"[dm_criteria] {len(PROB)} months of class probabilities · K={K} · EW · borrowable shorts")

    print("\nA. RANKING CRITERION (hold=1, no band) — 'Sharpe/cost in the model'"); print(HDR)
    S = {}
    for c in ("ret", "sharpe", "prdf1", "prdf5", "netret", "netsharpe"):
        S[c] = scores(c); rep(c, book(S[c]))

    print("\nB. HOLDING PERIOD — overlapping Jegadeesh-Titman tranches"); print(HDR)
    for c in ("ret", "sharpe"):
        for h in (1, 3, 6, 12):
            rep(f"{c} hold={h}", book(S[c], hold=h))

    print("\nC. BANDING — buy/hold spread (Novy-Marx-Velikov's top cost lever)"); print(HDR)
    for c in ("ret", "sharpe"):
        for bd in ((0.10, 0.20), (0.10, 0.30), (0.05, 0.30)):
            rep(f"{c} band={bd}", book(S[c], band=bd))

    print("\nD. BEST-OF combinations + LIQUID universe (the capacity test)"); print(HDR)
    for c in ("ret", "sharpe", "netsharpe"):
        rep(f"{c} hold=6 band=(.10,.30)", book(S[c], hold=6, band=(0.10, 0.30)))
        rep(f"{c} hold=6 band LIQUID",    book(S[c], hold=6, band=(0.10, 0.30), uni=liq))
