#!/usr/bin/env python3
"""
mr_research.py — THE reusable MEAN-REVERSION research harness. ONE script, OVERWRITTEN per experiment.

WORKFLOW (RESEARCH_PROTOCOL Phase R → 4): read MR_research.md + the ⚠️MANDATORY QP articles → edit the EXPERIMENT
block → run → log noteworthy results as a T-entry in MR_research.md → overwrite. NO one-off files.

Encodes the QP framework (Quantitativo): RARITY (QP) + 200d-SMA TREND GATE + EVENT-HOLD, dollar-neutral, on the
HONEST daily engine (signal[d] → trade close[d+1] → earn d+2 = lag=2; per-side bps on daily turnover; borrow on
shorts). Any lag-0 result is a look-ahead mirage. Reversion is a ~0-corr DIVERSIFIER — judge net-SR + corr to MOM/DM,
not standalone stardom (prior honest finding: ~0.2–0.3 net in the liquid universe; treat Quantitativo's 1.1–1.55 as a ceiling).
"""
import warnings, os, sys
warnings.filterwarnings("ignore"); os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, "/Users/enzokreeft/XFUND")
import numpy as np, pandas as pd
from scipy.stats import norm
from DATAHUB import DataHub

def build(tier="liquid"):
    hub = DataHub(start="2000-01-01", min_days=0)
    if hub.px_d is None: raise RuntimeError("MR needs DAILY data (DataHub daily-native mode)")
    return hub, tier

# ── QP rarity signal + trend gate ──────────────────────────────────────────────────────────────────────────
def qp_rarity(hub, k=3, win=252):
    """QP = rarity of the k-day return in the stock's OWN trailing window, 0–100 (low = rare DROP, high = rare POP).
    Rarity via trailing z-score → Gaussian CDF (cheap, vectorised proxy for the empirical-percentile QP indicator)."""
    r = hub.px_d / hub.px_d.shift(k) - 1
    z = (r - r.rolling(win, min_periods=int(win*0.5)).mean()) / (r.rolling(win, min_periods=int(win*0.5)).std() + 1e-9)
    return norm.cdf(z) * 100.0                                    # DataFrame days×names in [0,100]

def qp_signals(hub, tier="liquid", k=3, win=252, lo=15, hi=85, sma=200):
    """Long = rare DROP (QP<lo) in UPTREND (px>200SMA); Short = rare POP (QP>hi) in DOWNTREND (px<200SMA)."""
    QP = qp_rarity(hub, k, win); s = hub.sma_d(sma)
    elig = hub.elig(tier, "daily"); short_ok = hub.elig("shortable" if tier != "relaxed" else "relaxed", "daily")
    up, dn = hub.px_d > s, hub.px_d < s
    long_t  = (QP < lo) & up & elig
    short_t = (QP > hi) & dn & short_ok
    return long_t.fillna(False), short_t.fillna(False)

# ── event-hold → dollar-neutral daily weights → honest backtest ────────────────────────────────────────────
def event_weights(trigger, H=6, lag=2):
    """In-position mask: entered at close[t-1] on trigger[t-2], held H days (fixed-hold approx of the mean-touch exit)."""
    return trigger.shift(lag).rolling(H, min_periods=1).max().fillna(0) > 0

def net(hub, long_t, short_t, tag, H=6, lag=2, cost_bps=5.0, borrow_bps=1.0, short_frac=0.5, gross=2.0):
    Lact, Sact = event_weights(long_t, H, lag), event_weights(short_t, H, lag)
    nl, ns = Lact.sum(1).replace(0, np.nan), Sact.sum(1).replace(0, np.nan)
    wl = Lact.div(nl, 0) * (gross / (1 + short_frac))                                   # long book
    ws = -Sact.div(ns, 0) * (gross / (1 + short_frac)) * short_frac                     # short book (smaller)
    W = wl.add(ws, fill_value=0.0)
    ret = hub.ret_d.reindex_like(W).fillna(0.0)
    pnl = (W * ret).sum(1)                                                              # daily portfolio return
    turn = W.diff().abs().sum(1); cost = turn * cost_bps / 1e4
    bcost = ws.abs().sum(1) * borrow_bps / 1e4 / 252                                    # borrow on short notional
    net = (pnl - cost - bcost).dropna()
    ann = net.mean() * 252; vol = net.std() * np.sqrt(252); sr = ann / (vol + 1e-9)
    cum = (1 + net).cumprod(); dd = (cum / cum.cummax() - 1).min()
    print(f"  {tag:22}{sr:>8.2f}{ann:>9.1%}{dd:>9.1%}{turn.mean()*252:>8.1f}{int(Lact.sum(1).mean()):>6d}L{int(Sact.sum(1).mean()):>5d}S", flush=True)
    return net


# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
# EXPERIMENT — edit ONLY below. T1: honest QP baseline. Then log noteworthy results to MR_research.md.
# ══════════════════════════════════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    hub, tier = build(tier=os.environ.get("TIER", "liquid"))
    print(f"[MR] tier={tier} · honest daily engine (lag=2, {5.0}bp/side)  QP<15 + 200SMA gate + {6}-bar hold", flush=True)
    print(f"\n{'arm':22}{'net SR':>8}{'ann':>9}{'maxDD':>9}{'turn':>8}{'names':>8}", flush=True)
    lt, st = qp_signals(hub, tier)
    net(hub, lt, st, "QP L/S (gated)")                            # T1 baseline
    net(hub, lt, st.iloc[0:0].reindex_like(st).fillna(False), "QP long-only")   # long-only (short off)
    # T2 (ablate gate): compare unconditional (no 200SMA) — swap qp_signals(sma large) etc. in next run.
