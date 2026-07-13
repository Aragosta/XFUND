#!/usr/bin/env python3
"""capacity_poc.py — simple ADV-participation capacity test on the champion book.

At AUM A, a name's target position $ = |weight| * A. Cap it at PART * ADV_$ (ADV = trailing 63-day mean daily
$-volume). Names whose target exceeds the cap get shrunk to the cap (the un-deployable capital -> cash, so gross
shrinks). Backtest the CAPPED book net at each AUM to see how SR / ann / deployed-gross degrade with size —
the honest capacity curve, vs the (capacity-blind) target book."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST

PART = float(__import__("os").environ.get("PART", 1.0))                     # max position = PART days of ADV
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
mret = px.reindex(me).pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
adv = (px*vb).rolling(63, min_periods=40).mean().reindex(me, method="ffill")  # ADV_$ (daily), at month-ends

W = pickle.load(open("/tmp/meta_weights.pkl","rb")); W.index = pd.DatetimeIndex(pd.to_datetime(W.index))
W = W[~W.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)

def capped(A):
    cap = (PART*adv/A)                                                      # max |weight| that fits ADV at this AUM
    Wc = np.sign(W)*np.minimum(W.abs(), cap.reindex_like(W).fillna(np.inf))
    dep = Wc.abs().sum(1) / W.abs().sum(1).replace(0, np.nan)               # deployed fraction of target gross
    return Wc, float(dep[dep.index >= "2013-01-01"].mean())
def rep(name, Wc, dep):
    r = BACKTEST.backtest(Wc.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in Wc.index if Wc.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)
    rr = pd.Series(r["returns"]); rr.index = pd.DatetimeIndex(rr.index); rr = rr[rr.index >= "2011-01-01"]
    print(f"  {name:16}{dep:>9.0%}{r['sharpe']:>7.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{skew(rr):>+7.2f}", flush=True)

print("="*64); print(f"ADV-CAPACITY curve — champion book, cap = {PART:.1f} x daily ADV (net, 2011+)")
print(f"  {'AUM':16}{'deployed':>9}{'SR':>7}{'ann':>8}{'maxDD':>8}{'skew':>7}")
rep("target (no cap)", W, 1.0)
for A in (1e8, 5e8, 2e9, 1e10):
    Wc, dep = capped(A); rep(f"${A/1e9:.1f}B", Wc, dep)