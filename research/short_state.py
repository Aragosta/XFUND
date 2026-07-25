#!/usr/bin/env python3
"""
short_state.py — STATE/risk enquiry: can a TREND/DD-conditioned short leg beat long-only end-to-end?

IDEA (plain words): our DM sleeve has a real L/S book, but the short leg was retired because it is the
squeeze+borrow tax ([[short-leg-is-the-tax]], [[target-universe-dependent]]). Hosseinkhani (2026) and our own
short-vol proofs ([[short-vol-thesis-proven]]) both say the short/limits leg is NOT unconditionally dead — it
REVERSES on MARKET TREND: it pays in low-trend/stress and gets squeezed in calm up-trends. So instead of a
constant short, dial the SHORT-leg capital by a trend/drawdown state: full short in low-trend/high-drawdown,
floored/zero in calm up-trends. Test whether that conditioned short beats (a) long-only and (b) constant L/S,
END-TO-END net of tiered cost + borrow, on the SAME cached DM predictions (no retrain — isolates the sizing).

The user's instinct was VIX-level. The paper predicts TREND (not VIX-level) is the sharp conditioner, and that
VIX-level is weaker / can be wrong-signed. So we also test VIX arms (both directions + rising-vol) to let the
data adjudicate directly on our own book.

CONTROL / BAR: long_only is the current champion posture. A short overlay earns its place only if net SR beats
long_only by >= +0.10 without materially worsening maxDD. If nothing clears it, that CONFIRMS the short leg is
dead even conditionally on this book (a real result), and the paper's family-level edge does not survive our
name-level borrow/squeeze engine.

Engine: BACKTEST.py only (same pnl grid, tiered cost+borrow, lag=0, freq=12 — identical to DM.py).
Reuses /tmp/dm_weights.pkl (DM champion L/S walk output). Parameter-light: states are expanding-percentile
(PIT), no hand-picked slopes — matching the state layer's parameter-free dial ethos.
"""
import warnings, os, sys, pickle
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT); os.chdir(_ROOT)
import BACKTEST
from DATAHUB import DataHub

# ── load cached DM L/S predictions (the book whose short leg we re-size) ─────────────────────────────
W = pickle.load(open("/tmp/dm_weights.pkl", "rb"))
W = W.sort_index()
Wl = W.clip(lower=0.0)                                   # long leg  (>=0)
Ws = W.clip(upper=0.0)                                   # short leg (<=0)
print(f"[dm book] {W.shape[0]} months {W.index.min():%Y-%m}..{W.index.max():%Y-%m} · "
      f"avg long gross {Wl.sum(1).mean():.2f} · avg short gross {(-Ws.sum(1)).mean():.2f}", flush=True)

# ── substrate: pnl grid + tiered cost/borrow (identical to DM.py) ────────────────────────────────────
hub = DataHub(start="2000-01-01", min_days=0)
pnl = hub.pnl("monthly"); sm = hub.dollar_size("monthly")
tcd = BACKTEST.tiered_transaction_costs(sm); bfd = BACKTEST.tiered_borrow_fees(sm)
me  = hub.me

# ── PIT market-trend / drawdown / VIX states (as-of each signal month-end) ───────────────────────────
elig = hub.elig("liquid")
mktm = hub.mret.where(elig).mean(axis=1)                 # equal-weight market monthly return
cum  = (1 + mktm.fillna(0)).cumprod()
trend12 = cum / cum.shift(12) - 1
trend24 = cum / cum.shift(24) - 1
dd12    = cum / cum.rolling(12, min_periods=6).max() - 1  # <=0, deeper = more stress
vix     = hub.macro_m["vix"] if (hub.macro_m is not None and "vix" in hub.macro_m) else pd.Series(np.nan, index=me)
vix_chg = vix - vix.shift(3)                              # rising vol = squeeze onset

def exp_pct(s):
    """PIT expanding percentile rank in [0,1] (fraction of history <= current); parameter-free, no look-ahead."""
    s = s.reindex(me)
    return s.expanding(min_periods=24).apply(lambda x: float((x.iloc[-1] >= x.values).mean()), raw=False)

p_trend = exp_pct(trend12)
p_dd    = exp_pct(dd12)
p_vix   = exp_pct(vix)
p_vixch = exp_pct(vix_chg)

# short_scale in [0,1]: HIGH => keep full short; LOW => cut short (squeeze-prone) --------------------
SCALES = {
    "long_only  (short=0)":        pd.Series(0.0, index=me),
    "const L/S  (short=1)":        pd.Series(1.0, index=me),
    "trend_pct  (paper)":          (1.0 - p_trend),                       # low trend -> full short
    "bear_gate  (trend24<0)":      (trend24 < 0).astype(float).reindex(me),
    "dd_pct     (deep DD->short)": (1.0 - p_dd),                          # deep drawdown -> full short
    "vix_HI_cut (naive intuit.)":  (1.0 - p_vix),                         # high VIX -> cut short
    "vix_LO_cut (calm=squeeze)":   p_vix,                                 # low VIX  -> cut short
    "vix_rising_cut":              (1.0 - p_vixch),                       # rising vol -> cut short
}

# ── run each arm through the honest engine ──────────────────────────────────────────────────────────
def run(scale):
    s = scale.reindex(W.index).clip(0, 1)                # align to book dates; missing state -> 0 short (conservative)
    Wp = Wl.add(Ws.mul(s.fillna(0.0), axis=0), fill_value=0.0)
    Wp = Wp.reindex(columns=pnl.columns).sort_index()
    return BACKTEST.backtest(Wp, pnl, freq=12, lag=0, signal_dates=list(Wp.index),
                             transaction_cost=tcd, borrow_fee=bfd)

print(f"\n{'arm':30}{'netSR':>8}{'ann':>8}{'maxDD':>9}{'turn':>7}{'borrow':>8}{'shrt%':>7}", flush=True)
base = None
for nm, sc in SCALES.items():
    r = run(sc)
    if nm.startswith("long_only"): base = r["sharpe"]
    d = "" if base is None else f"   (Δ {r['sharpe']-base:+.2f})"
    shr = f"{sc.reindex(W.index).clip(0,1).mean():.2f}"
    print(f"{nm:30}{r['sharpe']:>8.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>9.1%}"
          f"{r['ann_turnover']:>7.1f}{r['ann_borrow']:>8.2%}{shr:>7}{d}", flush=True)
print("\nnote: Δ vs long_only. short% = avg short-leg scale. paper => trend_pct should top vix_* arms.", flush=True)

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# PART 2 — MODEL TIMING: size the SLEEVE by state (Daniel-Moskowitz managed momentum), not the short leg.
# "when the model performs worse, allocate less." Screen at the RETURN-series level (scale next-month net
# return by a PIT state factor s_{t-1} in [floor,1]); this is exactly the DM(2016) managed-momentum object.
# We test it on the long-only DM book (the deployable posture from Part 1) and on the full L/S book.
# ════════════════════════════════════════════════════════════════════════════════════════════════════
def sr(x):
    x = x.dropna(); return np.sqrt(12) * x.mean() / x.std() if x.std() > 0 else np.nan
def mdd(x):
    e = (1 + x.fillna(0)).cumprod(); return (e / e.cummax() - 1).min()

r_long = run(SCALES["long_only  (short=0)"])["returns"]
r_ls   = run(SCALES["const L/S  (short=1)"])["returns"]

# PIT state factors known at t-1, applied to the return earned at t (shift 1) -------------------------
volp   = mktm.rolling(6, min_periods=3).std()                        # market realized vol (trailing)
panic  = ((trend24 < 0) & (p_vix > 0.66)).astype(float)             # DM(2016) panic: bear + high vol
factors = {
    "static":            pd.Series(1.0, index=me),
    "vol_managed":       (1.0 / (volp / volp.expanding(24).median())).clip(0.4, 1.0),  # inverse-vol (STATE-style)
    "panic_cut(DM2016)": (1.0 - 0.7 * panic),                        # cut to 0.3 in bear+high-vol
    "trend_scale":       (0.4 + 0.6 * p_trend),                      # more in up-trend, less in down (pro-trend)
    "bear_off":          (trend24 >= 0).astype(float).clip(0.3, 1),  # near-off in bear
}
print(f"\n{'sleeve x state-size':30}{'netSR':>8}{'maxDD':>9}   (return-level screen, ignores re-scale turnover)", flush=True)
for book_nm, r in [("DM long-only", r_long), ("DM full L/S", r_ls)]:
    for fn, f in factors.items():
        rr = r * f.reindex(r.index).shift(1).clip(0, 1).fillna(1.0)
        print(f"  {book_nm:16}{fn:14}{sr(rr):>8.2f}{mdd(rr):>9.1%}", flush=True)

# ════════════════════════════════════════════════════════════════════════════════════════════════════
# PART 3 — NAME-AXIS repair: keep the short leg only on SAFE-TO-SHORT names (proven squeeze vetoes:
# liquid/borrowable + low idio-vol + not near 52w-high). This is the CROSS-SECTIONAL lever our own
# research says is where short predictability lives — Part 1 tested the wrong (time) axis.
# ════════════════════════════════════════════════════════════════════════════════════════════════════
rvol6 = hub.mret.rolling(6, min_periods=4).std()                     # idio-vol proxy (high = squeeze-prone)
hi52  = hub.m_px / hub.m_px.rolling(12, min_periods=8).max()         # near 52w-high = squeeze fuel
dvol  = sm                                                           # dollar-volume = borrowability/liquidity
def _al(df): return df.reindex(index=W.index, columns=W.columns)
liquid  = _al(dvol).rank(pct=True, axis=1) >= 0.50                   # easy-to-borrow half
lowvol  = _al(rvol6).rank(pct=True, axis=1) <= 0.60                  # calmer 60%
nothigh = _al(hi52) <= 0.90                                          # >10% below its 52w high
safe    = (liquid & lowvol & nothigh).fillna(False)

Ws_safe = Ws.where(safe, 0.0)
gs = (-Ws.reindex_like(Ws_safe)).sum(1); gn = (-Ws_safe).sum(1)
Ws_safe = Ws_safe.mul((gs / gn.replace(0, np.nan)).fillna(0.0), axis=0)   # renorm to original short gross

def run_book(short_leg, s=None):
    sl = short_leg if s is None else short_leg.mul(s.reindex(W.index).clip(0,1).fillna(0), axis=0)
    Wp = Wl.add(sl, fill_value=0.0).reindex(columns=pnl.columns).sort_index()
    return BACKTEST.backtest(Wp, pnl, freq=12, lag=0, signal_dates=list(Wp.index),
                             transaction_cost=tcd, borrow_fee=bfd)

print(f"\n{'name-axis repair':30}{'netSR':>8}{'ann':>8}{'maxDD':>9}{'borrow':>8}   (vs long_only 0.18)", flush=True)
for nm, book, sc in [
    ("raw short (all names)",   Ws,      None),
    ("safe-short (veto)",       Ws_safe, None),
    ("safe-short + bear_gate",  Ws_safe, (trend24 < 0).astype(float)),
    ("safe-short + trend_pct",  Ws_safe, (1.0 - p_trend)),
]:
    r = run_book(book, sc)
    print(f"  {nm:28}{r['sharpe']:>8.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>9.1%}{r['ann_borrow']:>8.2%}", flush=True)
print(f"  (safe-short keeps {safe.sum(1).mean():.0f}/{(Ws<0).sum(1).mean():.0f} short names on avg)", flush=True)

