#!/usr/bin/env python3
"""ibkr_costs.py — an ACCURATE, EMPIRICAL Interactive Brokers cost model.

WHY THIS REPLACES THE TIERS. `BACKTEST.DEFAULT_COST_TIERS` buckets cost purely by dollar volume
(5/10/25/60/150 bp). That is wrong in BOTH directions and for a structural reason:

  * IBKR commission is charged PER SHARE, not per dollar. So commission in basis points scales with
    1/price, NOT with liquidity. A $200 mega-cap costs 0.0035/200 = 0.18 bp; a $2 stock costs 17.5 bp
    at identical dollar volume. The tiers charge both 5 bp if they trade $1B/mo.
  * The dominant cost for illiquid names is the SPREAD, which we should MEASURE, not assume — and we
    have daily HIGH/LOW, so we can estimate the effective spread per name per month directly.

COMPONENTS (all one-way, as a fraction of traded notional)

 1. COMMISSION — IBKR Pro, US stocks, TIERED schedule:
      $0.0035/share for monthly volume ≤ 300k shares (the tier a fund of our size sits in),
      minimum $0.35 per order, MAXIMUM 1% of trade value.
    → commission_bp = clip(0.0035 / price, 0, 0.01).  The 1% cap binds below ~$0.35/share.

 2. EFFECTIVE SPREAD — Corwin & Schultz (2012) high-low estimator, computed from OUR OHLC:
      β = Σ_{j=0,1} [ln(H_j/L_j)]²          (two consecutive single-day high-low ranges)
      γ = [ln(H_{2day}/L_{2day})]²          (the two-day high-low range)
      α = (√(2β) − √β)/(3 − 2√2) − √(γ/(3 − 2√2))
      S = 2(e^α − 1)/(1 + e^α)              (proportional spread; negatives set to 0)
    We pay HALF the spread one-way. This is the standard low-frequency spread proxy and it is
    measured from data rather than assumed from a bucket.

 3. SEC/regulatory — SEC Section 31 fee is charged on SELLS only, currently ~$27.80 per $1M
    (0.278 bp). Applied as a half-turn average of 0.139 bp. FINRA TAF is ~$0.000166/share (negligible,
    included in the commission term's rounding).

 4. IMPACT — square-root law, opt-in: impact_bp = k·√(participation), participation = trade/ADV.
    OFF by default (k=0) so the model stays a pure cost-to-cross estimate; turn it on to size-test.

BORROW: IBKR general collateral is ~0.25-0.50%/yr; hard-to-borrow ranges from a few % to 100%+. We have
no historical IBKR rate feed, so `borrow_panel` keeps a liquidity-tiered shape but recalibrated to IBKR's
published GC level rather than the old punitive tiers. FLAGGED as the least accurate part of this model —
the honest fix is to record IBKR's SLB rates going forward (see the HTB data notes in MOM_research.md).

Usage:
    from research.ibkr_costs import ibkr_cost_panel, borrow_panel
    tc = ibkr_cost_panel(hub, grid="monthly")     # one-way fraction, dates × names
    bf = borrow_panel(hub, grid="monthly")        # annual borrow fee fraction
Run directly for a calibration report vs the old tiers and vs Frazzini-Israel-Moskowitz (6.2 bp median).
"""
import warnings, sys, os
warnings.filterwarnings("ignore"); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

IBKR_PER_SHARE = 0.0035        # IBKR Pro tiered, monthly volume ≤ 300k shares
IBKR_MAX_PCT   = 0.01          # commission capped at 1% of trade value
SEC_FEE_HALF   = 0.0000139     # SEC §31 fee 0.278 bp on sells → 0.139 bp per half-turn
TICK           = 0.01          # US equities trade on a $0.01 tick above $1.00
K_IMPACT       = 0.10          # square-root impact: trading 100% of ADV costs ~10% (Almgren/Kyle calibration)
_SQRT2 = np.sqrt(2.0); _K = 3.0 - 2.0 * _SQRT2


def corwin_schultz_spread(high: pd.DataFrame, low: pd.DataFrame,
                          volume: pd.DataFrame | None = None) -> pd.DataFrame:
    """Corwin-Schultz (2012) proportional effective spread from daily HIGH/LOW. Returns a DAILY panel.

    TWO CORRECTIONS, both aimed at the estimator's known failure mode on illiquid names:
      (a) NON-TRADING DAYS. If a stock does not trade, high == low, the range is 0 and CS collapses toward a
          ZERO spread — i.e. it reports the least tradable names as the CHEAPEST. Those days are set to NaN
          (via `volume`, and via the degenerate high==low test) so they never enter the monthly median.
      (b) NEGATIVE ESTIMATES. CS routinely yields α<0 → S<0. Clipping at 0 (the paper's convention) biases
          the estimate DOWN. We clip at the TICK FLOOR instead, applied by the caller.
    """
    H, L = high.where(high > 0), low.where(low > 0)
    if volume is not None:
        nz = volume.reindex_like(H) > 0
        H, L = H.where(nz), L.where(nz)
    deg = (H <= L)                                                       # no intraday range = did not trade
    H, L = H.where(~deg), L.where(~deg)
    hl = np.log(H / L)
    b = hl ** 2 + (hl ** 2).shift(1)                                     # β: two single-day ranges
    H2 = pd.concat([H, H.shift(1)]).groupby(level=0).max()
    L2 = pd.concat([L, L.shift(1)]).groupby(level=0).min()
    g = np.log(H2 / L2) ** 2                                             # γ: the two-day range
    a = (np.sqrt(2.0 * b) - np.sqrt(b)) / _K - np.sqrt(g / _K)
    S = 2.0 * (np.exp(a) - 1.0) / (1.0 + np.exp(a))
    return S.clip(lower=0.0)                                             # negative estimates → 0 (CS convention)


def ibkr_cost_panel(hub, grid: str = "monthly", impact_k: float = 0.0,
                    participation: float = 0.0) -> pd.DataFrame:
    """ONE-WAY cost fraction per name: IBKR commission + half the MEASURED spread + SEC fee [+ impact].

    NaN is preserved wherever the name has no price (a survivorship-free panel is mostly empty cells).
    Gaps INSIDE a name's life are filled by that month's cross-sectional median so the panel is safe to
    feed to BACKTEST, but dead cells stay NaN and never enter calibration statistics.
    """
    px_d = hub.px_d
    daily = pd.read_parquet("data/daily.parquet")
    hi = daily["high"].reindex(index=px_d.index, columns=px_d.columns)
    lo = daily["low"].reindex(index=px_d.index, columns=px_d.columns)
    vol = daily["volume"].reindex(index=px_d.index, columns=px_d.columns)
    spr_d = corwin_schultz_spread(hi, lo, volume=vol)
    if grid == "monthly":
        spr = _to_grid(spr_d.resample("ME").median(), hub.me, px_d.columns)
        price = _to_grid(px_d.resample("ME").last(), hub.me, px_d.columns)
    else:
        spr, price = spr_d, px_d
    alive = price.notna()
    comm = (IBKR_PER_SHARE / price.where(price > 0)).clip(upper=IBKR_MAX_PCT)
    spr = spr.where(alive)
    spr = spr.apply(lambda r: r.fillna(r.median()), axis=1)          # fill gaps WITHIN the cross-section
    # TICK FLOOR — a US equity cannot have a spread below one $0.01 tick, so the HALF-spread cannot be below
    # $0.005/price. This is a PHYSICAL bound, and it is exactly where Corwin-Schultz is weakest: a $2 stock
    # is floored at 25 bp, a $200 stock at 0.25 bp. It removes the "microcaps are cheap" artifact directly.
    half = np.maximum(0.5 * spr, (TICK / 2.0) / price.where(price > 0))
    cost = (comm + half + SEC_FEE_HALF).where(alive)
    if impact_k > 0 and participation > 0:                           # square-root impact, size-dependent
        cost = cost + impact_k * np.sqrt(np.clip(participation, 0, None))
    return cost.replace([np.inf, -np.inf], np.nan).clip(lower=0.0001, upper=0.50)


def cost_at_aum(hub, W, aum, base=None, k=K_IMPACT):
    """THE cost model, fully specified: commission + max(measured half-spread, tick floor) + impact(AUM).

    Once AUM is given there is no free judgement left — impact is the only size-dependent term and it uses
    the standard square-root law on participation = |Δw|·AUM / ADV.
    """
    base = base if base is not None else ibkr_cost_panel(hub)
    adv = hub.dollar_size("monthly").rolling(3, min_periods=1).mean()
    dW = W.diff().abs().fillna(W.abs())
    part = (dW * aum).div(adv.reindex_like(W).replace(0, np.nan))
    return (base.reindex_like(W).fillna(base.stack().median())
            + k * np.sqrt(part.clip(lower=0).fillna(0.0))).clip(0.0001, 0.50)


def _to_grid(df, me, cols):
    """Align a calendar-month-end panel onto DataHub's trading-month-end grid (period match)."""
    d = df.copy(); d.index = pd.DatetimeIndex(d.index).to_period("M")
    d = d.reindex(pd.PeriodIndex(me, freq="M")); d.index = me
    return d.reindex(columns=cols)


# IBKR-calibrated borrow: GC 0.25-0.50%/yr; HTB scales up with illiquidity. LEAST accurate component —
# no historical IBKR SLB feed available, so the SHAPE is tiered but the LEVEL is IBKR's published GC.
IBKR_BORROW_TIERS = (
    (1e9, 0.0030),   # ≥ $1B/mo   → 0.30%/yr  (general collateral)
    (1e8, 0.0050),   # ≥ $100M    → 0.50%/yr
    (1e7, 0.0150),   # ≥ $10M     → 1.5%/yr
    (1e6, 0.0600),   # ≥ $1M      → 6%/yr
    (0.0, 0.2000),   # < $1M      → 20%/yr    (often simply unborrowable — the elig filter removes those)
)


def borrow_panel(hub, grid: str = "monthly") -> pd.DataFrame:
    import BACKTEST
    return BACKTEST.tiered_borrow_fees(hub.dollar_size(grid), tiers=IBKR_BORROW_TIERS)


if __name__ == "__main__":
    from DATAHUB import DataHub
    import BACKTEST
    hub = DataHub(start="1990-01-01", min_days=0)
    new = ibkr_cost_panel(hub); old = BACKTEST.tiered_transaction_costs(hub.dollar_size("monthly"))
    mdv = hub.dollar_size("monthly").rolling(3, min_periods=1).mean()
    print("\n[ibkr_costs] ONE-WAY cost, basis points — MEASURED (IBKR commission + Corwin-Schultz spread)")
    print(f"  {'universe':30}{'n obs':>10}{'OLD tiers':>12}{'IBKR model':>12}{'ratio':>8}")
    for tag, m in (("all names", None), ("mdv>$1M/mo", mdv > 1e6), ("mdv>$5M/mo", mdv > 5e6),
                   ("mdv>$100M/mo", mdv > 1e8), ("mdv>$1B/mo", mdv > 1e9)):
        o = old.where(m) if m is not None else old
        n_ = new.where(m) if m is not None else new
        os_, ns_ = o.stack(), n_.stack()
        if not len(ns_): continue
        print(f"  {tag:30}{len(ns_):>10,}{os_.median()*1e4:>12.1f}{ns_.median()*1e4:>12.1f}"
              f"{ns_.median()/max(os_.median(),1e-9):>8.2f}")
    print(f"\n  Frazzini-Israel-Moskowitz (2015) measured median institutional cost = 6.2 bp")
    print(f"  IBKR model median, mdv>$100M/mo = {new.where(mdv>1e8).stack().median()*1e4:.1f} bp")
    print("\n  commission share of total cost, by liquidity:")
    px = _to_grid(hub.px_d.resample("ME").last(), hub.me, hub.px_d.columns)
    comm = (IBKR_PER_SHARE / px.where(px > 0)).clip(upper=IBKR_MAX_PCT)
    for tag, m in (("mdv>$1B/mo", mdv > 1e9), ("mdv>$100M/mo", mdv > 1e8), ("mdv>$5M/mo", mdv > 5e6)):
        print(f"    {tag:16}{(comm.where(m).stack()/new.where(m).stack()).median():>6.1%}"
              f"   median price ${px.where(m).stack().median():>7.2f}")
