#!/usr/bin/env python3
"""FULLTEST.py — end-to-end: MOM -> DM -> ERC -> META -> DynTrad. Loads the META final book
(/tmp/meta_weights.pkl, produced by META.py) and overlays the Garleanu-Pedersen DynTrad execution layer
(partial adjustment -> cuts turnover/capacity). Re-neutralizes AFTER DynTrad (stale positions carry stale
beta). Net via BACKTEST.py (tiered)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
import BACKTEST, BETANEUT
from execution_layer import DynTrad

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
Wmeta = pickle.load(open("/tmp/meta_weights.pkl","rb")).reindex(index=me, columns=px.columns).fillna(0.0)

def rep(name, W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)
    rr = pd.Series(r["returns"]); rr.index = pd.DatetimeIndex(rr.index); rr = rr[rr.index >= "2011-01-01"]
    print(f"  {name:30}{r['sharpe']:>6.2f}{r['sortino_ratio']:>8.2f}{r['ann_return']:>8.1%}{r['ann_vol']:>7.1%}{r['max_drawdown']:>8.1%}{skew(rr):>+7.2f}{r['ann_turnover']:>7.1f}", flush=True)

print("="*98); print("FULL STACK — MOM -> DM(+TS) -> ERC -> META(odds) -> DynTrad  (net, 2011+)")
print(f"  {'stage':30}{'SR':>6}{'Sortino':>8}{'ann':>8}{'vol':>7}{'maxDD':>8}{'skew':>7}{'turn':>7}")
rep("META final book", Wmeta)

# --- CALIBRATE DynTrad to our book (a) phi = book ACF1, (b) lambda from measured cost/vol ---
r0 = BACKTEST.backtest(Wmeta.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in Wmeta.index if Wmeta.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)
book_ret = pd.Series(r0["returns"])                                              # META book monthly returns (net)
dW = Wmeta.diff().abs(); num = (tc.reindex_like(Wmeta).ffill()*dW).sum(1); den = dW.sum(1).replace(0, np.nan)
c_oneway = float((num/den).dropna().mean())                                      # turnover-weighted one-way cost fraction
dt = DynTrad.from_book(Wmeta, book_ret, avg_cost_oneway=c_oneway, risk_aversion=1.0, gross_exposure=2.0)
print(f"  [calibrated] phi={float(dt.phi[0]):.2f}  one-way cost={c_oneway*1e4:.0f}bp  book vol={book_ret.std()*100:.1f}%/mo  ->  lambda={dt.lam:.2f}  delta={dt.trading_fraction:.2f}", flush=True)
Wdyn = dt.run(Wmeta.fillna(0.0)).reindex(index=me, columns=px.columns).fillna(0.0)
rep(f"+ DynTrad CALIBRATED (d={dt.trading_fraction:.2f})", Wdyn)
rep(f"+ calibrated + re-BETANEUT", BETANEUT.betaneut(Wdyn, BETA))
# paper-default reference (lambda=1.5) for comparison
dtp = DynTrad(signal_decay=np.array([0.5]), risk_aversion=1.0, cost_multiplier=1.5, input_type="weights", gross_exposure=2.0)
rep(f"+ DynTrad paper-default (d={dtp.trading_fraction:.2f})", dtp.run(Wmeta.fillna(0.0)).reindex(index=me, columns=px.columns).fillna(0.0))
