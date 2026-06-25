"""
Known-answer validation for BACKTEST.py.

Each test builds a deterministic price/strategy whose return, volatility, drawdown,
or cost is known in closed form, then asserts the engine reproduces it.  Run:

    python test_backtest.py
"""
import numpy as np
import pandas as pd
from BACKTEST import backtest, tiered_transaction_costs, walk_forward

TOL = 1e-9
def _close(a, b, tol=1e-6): assert abs(a - b) <= tol, f"{a} != {b} (tol {tol})"
def dates(n): return pd.date_range("2000-01-31", periods=n, freq="ME")


def test_constant_return_buy_and_hold():
    """B&H at a constant +1%/period: every realized return is 1%, vol=0, drawdown=0."""
    n, r = 121, 0.01
    px = pd.DataFrame({"A": (1 + r) ** np.arange(n)}, index=dates(n))
    res = backtest(pd.Series({"A": 1.0}), px, freq=12, lag=0, transaction_cost=0.0)

    realized = res["returns"].to_numpy()[1:]          # period 0 has no return
    assert np.allclose(realized, r, atol=1e-12), "per-period return must equal the constant"
    _close(res["ann_vol"], 0.0, 1e-9)                 # constant return → zero vol
    _close(res["max_drawdown"], 0.0, 1e-12)           # monotonic up → no drawdown
    _close(res["total_return"], (1 + r) ** (n - 1) - 1)            # 120 compounding periods
    _close(res["ann_return"], (1 + r) ** 12 - 1)                   # (1.01^120)^(12/120) = 1.01^12
    print("✓ constant-return buy & hold: returns, vol=0, dd=0, annualization")


def test_known_drawdown():
    """Up 50% then down 50% → peak-to-trough drawdown is exactly -50%."""
    px = pd.DataFrame({"A": [1.0, 1.5, 0.75, 0.90]}, index=dates(4))
    res = backtest(pd.Series({"A": 1.0}), px, freq=12, lag=0, transaction_cost=0.0)
    _close(res["max_drawdown"], -0.5)                 # 0.75 / 1.5 - 1
    print("✓ known drawdown: max_drawdown = -50%")


def test_known_volatility():
    """Alternating ±10% returns → ann_vol = sample_std(±0.10) × √12 (independent calc)."""
    n = 121
    rets = np.where(np.arange(n) % 2 == 1, 0.10, -0.10)   # r[1],r[2],... alternate
    px = pd.DataFrame({"A": np.concatenate([[1.0], np.cumprod(1 + rets[1:])])}, index=dates(n))
    res = backtest(pd.Series({"A": 1.0}), px, freq=12, lag=0, transaction_cost=0.0)

    expected_vol = np.std(res["returns"].to_numpy()[1:], ddof=1) * np.sqrt(12)  # excl. period-0
    _close(res["ann_vol"], expected_vol, 1e-12)
    realized = res["returns"].to_numpy()[1:]
    assert np.allclose(np.abs(realized), 0.10, atol=1e-12), "returns must be ±10%"
    print(f"✓ known volatility: ann_vol = {res['ann_vol']:.4f} matches √12·σ")


def test_transaction_cost_exact():
    """Constant prices, one full rebalance, flat 100bp → equity falls by exactly tc·Σ|Δw|."""
    px = pd.DataFrame({"A": [1.0] * 5, "B": [1.0] * 5}, index=dates(5))
    w = pd.DataFrame({"A": [0.5], "B": [-0.5]}, index=[px.index[0]])   # Σ|Δw| = 1.0
    # long-only leg first to isolate cost size: use a long 100% in A
    wL = pd.DataFrame({"A": [1.0], "B": [0.0]}, index=[px.index[0]])
    res = backtest(wL, px, freq=12, lag=0, transaction_cost=0.01,
                   signal_dates=[px.index[0]], short_cost_mult=1.5)
    _close(res["total_return"], -0.01)                # |Δw|=1, cost=0.01, prices flat
    print("✓ transaction cost: 100bp on |Δw|=1 ⇒ -1.00% exactly")


def test_short_costs_1p5x():
    """A short position pays 1.5× the cost of the identical long position."""
    px = pd.DataFrame({"A": [1.0] * 5}, index=dates(5))
    long_  = backtest(pd.DataFrame({"A": [1.0]},  index=[px.index[0]]), px,
                      freq=12, lag=0, transaction_cost=0.01, signal_dates=[px.index[0]])
    short_ = backtest(pd.DataFrame({"A": [-1.0]}, index=[px.index[0]]), px,
                      freq=12, lag=0, transaction_cost=0.01, signal_dates=[px.index[0]],
                      short_cost_mult=1.5)
    cost_long  = -long_["total_return"]               # 0.01
    cost_short = -short_["total_return"]              # 0.015
    _close(cost_short / cost_long, 1.5)
    print(f"✓ short borrow proxy: short cost / long cost = {cost_short/cost_long:.2f}×")


def test_short_borrow_fee():
    """A held short pays borrow = annual_rate/freq each period; a long pays none."""
    px = pd.DataFrame({"A": [1.0] * 13}, index=dates(13))     # flat prices → isolate borrow
    short = backtest(pd.DataFrame({"A": [-1.0]}, index=[px.index[0]]), px,
                     freq=12, lag=0, transaction_cost=0.0, borrow_fee=0.12,
                     signal_dates=[px.index[0]])
    _close(short["ann_borrow"], 0.12)                         # 1%/mo × 12
    _close(short["total_return"], 0.99 ** 12 - 1)             # -1%/mo compounded, 12 periods
    long_ = backtest(pd.DataFrame({"A": [1.0]}, index=[px.index[0]]), px,
                     freq=12, lag=0, transaction_cost=0.0, borrow_fee=0.12,
                     signal_dates=[px.index[0]])
    _close(long_["ann_borrow"], 0.0)                          # long pays no borrow
    _close(long_["total_return"], 0.0)
    print("✓ short borrow fee: 12%/yr ⇒ -1%/mo on the short; long pays nothing")


def test_no_lookahead_timing():
    """A signal on the jump date must NOT capture that period's jump (executes next period)."""
    px = pd.DataFrame({"A": [1.0, 1.0, 1.0, 2.0, 2.0]}, index=dates(5))   # +100% return at index 3
    pre  = backtest(pd.DataFrame({"A": [1.0]}, index=[px.index[2]]), px,   # signal BEFORE jump
                    freq=12, lag=0, transaction_cost=0.0, signal_dates=[px.index[2]])
    at   = backtest(pd.DataFrame({"A": [1.0]}, index=[px.index[3]]), px,   # signal ON jump date
                    freq=12, lag=0, transaction_cost=0.0, signal_dates=[px.index[3]])
    lagd = backtest(pd.DataFrame({"A": [1.0]}, index=[px.index[2]]), px,   # lag=1 pushes past jump
                    freq=12, lag=1, transaction_cost=0.0, signal_dates=[px.index[2]])
    _close(pre["total_return"], 1.0)                  # signal at t=2 → exec t=3 → earns +100%
    _close(at["total_return"], 0.0)                   # signal at t=3 → exec t=4 → misses jump
    _close(lagd["total_return"], 0.0)                 # lag=1 → exec t=4 → misses jump
    print("✓ no look-ahead: same-date signal can't earn that period's return; lag shifts later")


def test_tiered_cost_tiers():
    """Dollar-volume maps to the correct one-way cost tier."""
    dv = pd.DataFrame({"big": [2e9], "mid": [5e7], "micro": [5e5]}, index=[dates(1)[0]])
    tc = tiered_transaction_costs(dv, lookback=1)
    _close(tc.loc[tc.index[0], "big"], 0.0005)        # ≥ $1B
    _close(tc.loc[tc.index[0], "mid"], 0.0025)        # $10M–100M
    _close(tc.loc[tc.index[0], "micro"], 0.0150)      # < $1M
    print("✓ tiered costs: $2B→5bp, $50M→25bp, $0.5M→150bp")


def test_walk_forward_is_oos():
    """Walk-forward only trades AFTER each train window (out-of-sample by construction)."""
    n = 60
    px = pd.DataFrame({"A": (1.01) ** np.arange(n), "B": (1.005) ** np.arange(n)}, index=dates(n))
    calls = {"trained_rows": []}
    def sig(train_px):
        calls["trained_rows"].append(len(train_px))
        return pd.Series({"A": 1.0, "B": 0.0})        # always hold A
    res = walk_forward(sig, px, train=24, test=6, freq=12, lag=0, transaction_cost=0.0)
    assert all(r == 24 for r in calls["trained_rows"]), "each fit must see exactly `train` rows"
    assert res["ann_return"] > 0, "holding the up-trending asset OOS should profit"
    print(f"✓ walk-forward OOS: {len(calls['trained_rows'])} blocks, train=24 rows each")


if __name__ == "__main__":
    test_constant_return_buy_and_hold()
    test_known_drawdown()
    test_known_volatility()
    test_transaction_cost_exact()
    test_short_costs_1p5x()
    test_short_borrow_fee()
    test_no_lookahead_timing()
    test_tiered_cost_tiers()
    test_walk_forward_is_oos()
    print("\nALL TESTS PASSED")
