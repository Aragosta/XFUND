#!/usr/bin/env python3
"""RISK.py — the per-sleeve RISK layer (alpha score → risk-managed book). Consolidates CONSTRUCT + BETANEUT.

WHERE IT SITS:  DATA(hub) > STATE(measure) > SLEEVES(alpha score) > RISK[ per-sleeve ] > ERC(allocate) > META > EXEC.
STATE *measures* risk (Σ_t, betas, tail propensities); RISK *acts* on it. This module is the PER-SLEEVE tier of the
risk layer: it turns one sleeve's SCORE panel into a tradable, risk-managed WEIGHTS panel. The PORTFOLIO tier
(ERC allocate; gross dial — currently a STATE diagnostic, NOT applied) runs once on the combined book, above here.

THE FOUR PER-SLEEVE RISK ACTIONS (applied in this order):
  1. SELECT     — cross-sectional decile / buy-hold banding (hysteresis). Which names are in the book.
  2. SIZE       — liquidity/vol weighting (mdv down-weights illiquid; volparity behind a mandatory liquidity floor).
  3. TAIL-VETO  — direction-aware bimodal tail control (Han's crash/squeeze modes as a RISK object, not alpha):
                  LONG position → down-weight DOWN-tail (crash) propensity; SHORT → down-weight UP-tail (squeeze).
                  Same function, tail side flipped by leg. Sourced from return-space semi-vol (universal; no
                  probability needed from the sleeve). This is the shared home for the short-squeeze research
                  (capping squeeze flipped a short book SR 0.28→1.69) generalized to both modes / either leg.
  4. NEUTRALIZE — strip market beta (Daniel-Moskowitz crash fix): w' = w − β·(w·β)/(β·β), renorm gross. Opt-in.

WHY (research/MOM_research.md T29 + construction A/B, 2026-07-24): construction was the single biggest momentum lever
(equal-weight decile net SR 0.14 → mdv+banded 0.70). Beta-neut is the crash tool (MOM 1.03→1.20, DM 1.26→1.50).
Tail-veto is the last unfilled risk cell (per-name idiosyncratic tail); A/B veto-vs-no-veto is the test.

DEFAULTS reproduce the CONSTRUCT champion: weighting="mdv", band=(0.10, 0.20), tail_veto=False, beta_neut=False
(strict superset — flip the flags on to test the new risk actions). CAPACITY: weighting="sqrt_mdv", band=(0.10, 0.30).
"""
import numpy as np, pandas as pd

WEIGHTINGS = ("equal", "mdv", "sqrt_mdv", "volparity")


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION 3 — tail-risk propensities (return-space, direction-aware; STATE-measured, RISK-acted)
# ══════════════════════════════════════════════════════════════════════════════
def tail_props(hub, grid="monthly", win=63):
    """Per-name UP-tail (squeeze) and DOWN-tail (crash) propensity on `grid`, from daily semi-deviation.

    up   = std of positive daily returns  (upside semi-vol)  → the SQUEEZE axis (veto for SHORT positions).
    down = std of negative daily returns  (downside semi-vol) → the CRASH  axis (veto for LONG  positions).
    These are the cheap, vectorized, proven-monotone tail predictors (asymmetric-feature-map: ivol/upvol/MAX5/hi52).
    Returned aligned to the sleeve grid (month-end or daily); the caller lags 1 bar (PIT), same as liquidity/vol.
    """
    rd = hub.ret_d
    if rd is None:
        return None, None
    up = rd.where(rd > 0).rolling(win, min_periods=win // 2).std()
    down = rd.where(rd < 0).rolling(win, min_periods=win // 2).std()
    if grid == "monthly":
        return hub.M(up), hub.M(down)
    return up.reindex(hub.days), down.reindex(hub.days)


def _veto_factor(prop_d, sel, lam, floor):
    """Direction-aware down-weight: names with the highest adverse-tail propensity get the most cut.
    factor = clip(1 − lam·rank_pct(prop), floor, 1). rank within the selected set (cross-sectional). lam=0 → no veto."""
    if lam <= 0 or prop_d is None:
        return pd.Series(1.0, index=sel)
    p = prop_d.reindex(sel)
    r = p.rank(pct=True)                                         # 1.0 = worst (most tail-prone) → most vetoed
    return (1.0 - lam * r).clip(lower=floor).fillna(1.0)


# ══════════════════════════════════════════════════════════════════════════════
#  ACTION 4 — beta neutralization (moved verbatim from BETANEUT.py; the crash tool)
# ══════════════════════════════════════════════════════════════════════════════
def rolling_beta(mret, elig, bw=60, min_periods=36):
    """Leak-free rolling market beta per stock (regress stock return on eligible-universe equal-weight market)."""
    mkt = mret.where(elig).mean(axis=1)
    mr = mret.rolling(bw, min_periods=min_periods).mean(); mm = mkt.rolling(bw, min_periods=min_periods).mean()
    mrm = mret.mul(mkt, axis=0).rolling(bw, min_periods=min_periods).mean()
    vm = (mkt ** 2).rolling(bw, min_periods=min_periods).mean() - mm ** 2
    return mrm.sub(mr.mul(mm, axis=0)).div(vm, axis=0).shift(1)             # shift(1) -> known at signal date


def neutralize(w, beta_row, gross=2.0):
    """Beta-neutralize one date's weight Series; renorm to `gross` (missing betas -> 1.0)."""
    b = beta_row.reindex(w.index).fillna(1.0).values
    wv = w.values - b * ((w.values @ b) / (b @ b + 1e-9)); g = np.abs(wv).sum()
    return pd.Series(wv * (gross / g) if g > 0 else wv, index=w.index)


def betaneut(W, BETA, gross=2.0):
    """Beta-neutralize a whole weight book (dates x tickers)."""
    cols = W.columns
    rows = {dt: neutralize(W.loc[dt].dropna(), BETA.loc[dt], gross) for dt in W.index if dt in BETA.index}
    return pd.DataFrame(rows).T.reindex(columns=cols).fillna(0.0)


# ══════════════════════════════════════════════════════════════════════════════
#  THE PER-SLEEVE RISK LAYER — score → risk-managed book (all four actions)
# ══════════════════════════════════════════════════════════════════════════════
def risk_book(scores, hub, *, tier="liquid", grid="monthly", weighting="mdv", band=(0.10, 0.20),
              decile=0.10, ls=False, liq_floor_q=0.5, vol_window=None,
              tail_veto=False, veto_lam=0.5, veto_floor=0.2, veto_win=63,
              beta_neut=False, beta_bw=60, beta_gross=2.0):
    """Turn a sleeve's SCORE panel (dates × names, higher = more bullish) into a risk-managed WEIGHTS panel.

    ACTIONS: SELECT (band/decile) · SIZE (weighting) · TAIL-VETO (tail_veto) · NEUTRALIZE (beta_neut). See module docstring.
    tail_veto : direction-aware bimodal down-weight. veto_lam=strength (0→off), veto_floor=max cut, veto_win=lookback days.
    beta_neut : strip market beta after the book is built (renorm gross `beta_gross`). Off = long-only book unchanged.
    """
    if isinstance(scores, dict):
        scores = pd.DataFrame({d: s for d, s in scores.items()}).T
    liq = hub.dollar_size(grid)                                          # size proxy = adj-close × period volume
    ret = hub.clean_returns(grid)
    vw = vol_window or (6 if grid == "monthly" else 21)
    vol = ret.rolling(vw, min_periods=max(3, vw // 2)).std()
    elig = hub.elig(tier, grid); shortable = hub.elig("shortable", grid)
    up_prop, down_prop = tail_props(hub, grid, veto_win) if tail_veto else (None, None)
    gidx = liq.index; pos = {d: i for i, d in enumerate(gidx)}
    en, ex = (band if band else (decile, decile))

    def _wt(sel, liq_d, vol_d):
        sel = [t for t in sel if t in liq_d.index]
        if not sel: return pd.Series(dtype=float)
        if weighting == "mdv":        w = liq_d.reindex(sel).clip(lower=0)
        elif weighting == "sqrt_mdv": w = np.sqrt(liq_d.reindex(sel).clip(lower=0))
        elif weighting == "volparity":                                  # MANDATORY liquidity floor, else it blows up
            lq = liq_d.reindex(sel); keep = lq[lq >= lq.quantile(liq_floor_q)].index
            w = (1.0 / (vol_d.reindex(keep) + 1e-12)).reindex(sel).fillna(0.0)
        else:                          w = pd.Series(1.0, index=sel)     # equal
        w = w.fillna(0.0); tot = w.sum()
        return w / tot if tot > 0 else pd.Series(0.0, index=sel)

    def _sized(sel, liq_d, vol_d, prop_d):
        """SIZE then TAIL-VETO (direction-aware), renormalized to unit leg. prop_d = adverse-tail propensity."""
        w = _wt(sel, liq_d, vol_d)
        if tail_veto and len(w):
            w = w * _veto_factor(prop_d, w.index, veto_lam, veto_floor)
            tot = w.sum(); w = w / tot if tot > 0 else w
        return w

    heldL, heldS, rows = set(), set(), {}
    for d in scores.index:
        i = pos.get(d)
        if i is None or i < 1: continue
        el = elig.iloc[i - 1]; ok = el.index[el.values.astype(bool)]
        s = scores.loc[d].dropna(); s = s[s.index.intersection(ok)]
        if len(s) < 20:
            heldL, heldS = set(), set(); continue
        liq_d, vol_d = liq.iloc[i - 1], vol.iloc[i - 1]
        dprop = down_prop.iloc[i - 1] if down_prop is not None else None   # LONG adverse tail = crash (down)
        uprop = up_prop.iloc[i - 1] if up_prop is not None else None       # SHORT adverse tail = squeeze (up)
        if band:                                                        # hysteresis on cross-sectional rank (0 = best)
            r = s.rank(ascending=False, pct=True)
            topL = (set(r.index[r <= en]) | (heldL & set(r.index[r <= ex]))) & set(s.index)
        else:
            topL = set(s.nlargest(max(1, int(len(s) * decile))).index)
        heldL = set(topL); w = _sized(topL, liq_d, vol_d, dprop)
        if ls:                                                          # dollar-neutral short leg — BORROWABLE only
            sh = shortable.iloc[i - 1]; cand = s[s.index.intersection(sh.index[sh.values.astype(bool)])]
            if len(cand) >= 20:
                if band:
                    rs = cand.rank(ascending=True, pct=True)
                    botS = (set(rs.index[rs <= en]) | (heldS & set(rs.index[rs <= ex]))) & set(cand.index)
                else:
                    botS = set(cand.nsmallest(max(1, int(len(cand) * decile))).index)
                heldS = set(botS); w = w.sub(_sized(botS, liq_d, vol_d, uprop), fill_value=0.0)
        rows[d] = w
    W = pd.DataFrame(rows).T.reindex(columns=liq.columns)

    if beta_neut:                                                       # ACTION 4 — strip market beta on the built book
        BETA = rolling_beta(ret, elig, bw=beta_bw)
        W = betaneut(W.fillna(0.0), BETA, gross=beta_gross)
    return W


# backwards-compatible alias while callers migrate off CONSTRUCT.construct (same signature, minus the new risk flags)
def construct(scores, hub, **kw):
    return risk_book(scores, hub, **kw)


if __name__ == "__main__":                                              # A/B the risk actions on a saved sleeve score
    import os, BACKTEST
    from DATAHUB import DataHub
    path = os.path.join(os.path.dirname(__file__), "CURRENT BEST", "out", "mom_score.parquet")
    if not os.path.exists(path):
        print(f"no score panel at {path} — run a sleeve first (mom_layer.save())"); raise SystemExit
    hub = DataHub(start="2000-01-01", min_days=0); S = pd.read_parquet(path)
    pnl = hub.pnl("monthly"); tc = BACKTEST.tiered_transaction_costs(hub.dollar_size("monthly"))
    bf = BACKTEST.tiered_borrow_fees(hub.dollar_size("monthly"))
    print(f"{'risk actions':40}{'netSR':>7}{'ann':>8}{'maxDD':>8}{'turn':>7}")
    grid = [
        ("mdv+band (champion, no new actions)", dict()),
        ("  + tail-veto (long crash)",          dict(tail_veto=True)),
        ("  + beta-neut",                        dict(beta_neut=True)),
        ("  + tail-veto + beta-neut",            dict(tail_veto=True, beta_neut=True)),
        ("L/S + tail-veto (both legs)",          dict(ls=True, tail_veto=True)),
    ]
    for name, kw in grid:
        W = risk_book(S, hub, **kw)
        r = BACKTEST.backtest(W, pnl, freq=12, lag=0, signal_dates=list(W.index), transaction_cost=tc, borrow_fee=bf)
        print(f"  {name:38}{r['sharpe']:>7.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{r['ann_turnover']:>7.1f}")
