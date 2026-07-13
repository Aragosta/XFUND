#!/usr/bin/env python3
"""lowvol_sleeve.py — the DEFENSIVE / low-risk sleeve (the diversifier), through BACKTEST.py from the start.

Rationale (see analysis): vol is ~perfectly predictable cross-sectionally (IC 0.93) vs returns (0.01), so a
risk-based factor is the one well-posed ML problem; the low-risk premium is real (high-vol underperforms) and
its capacity is COMPLEMENTARY to momentum (large, low-beta names). Value is as a low/negative-rho diversifier.

Tests each risk signal as a monthly dollar-neutral L/S book (LONG low-risk / SHORT high-risk) and reports
SR + turnover + CORRELATION to the DM/MOM momentum streams. Then a composite (mean rank of the signals) and
a beta-neutral BAB variant. One engine only (BACKTEST.py, tiered costs, lag=0)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle, os
import BACKTEST

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"])
px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
dvd = px*vb; dv = dvd.rolling(63, min_periods=40).mean()
dret = px.pct_change(fill_method=None).where(lambda z: z.abs() < 1.0)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0)
synth = (1 + mret.fillna(0.0)).cumprod()
mdv = dvd.resample("ME").sum().reindex(me, method="ffill")
elig = ((m_px > 5) & (dv.reindex(me, method="ffill") > 5e6) &
        (px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill") > 0.9)).fillna(False)
tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)

mkt = dret.mean(axis=1)                                                     # equal-weight market proxy
def at_me(d): return d.reindex(me, method="ffill")
# --- risk signals (all: LOW value = defensive = expected outperformer) ---
vol63  = at_me(dret.rolling(63, min_periods=40).std())
covm   = dret.mul(mkt, axis=0).rolling(126, min_periods=80).mean() - dret.rolling(126,min_periods=80).mean().mul(mkt.rolling(126,min_periods=80).mean(),axis=0)
beta   = at_me(covm.div((mkt**2).rolling(126,min_periods=80).mean() - mkt.rolling(126,min_periods=80).mean()**2, axis=0))
resid  = dret.sub(beta.reindex(dret.index, method="ffill").mul(mkt, axis=0))
idio   = at_me(resid.rolling(63, min_periods=40).std())
maxret = at_me(dret.rolling(21, min_periods=15).max())                      # lottery / MAX effect (Bali et al)
dnvol  = at_me(dret.where(dret < 0).rolling(63, min_periods=30).std())      # downside semi-deviation
SIG = {"vol": vol63, "beta": beta, "idio-vol": idio, "MAX": maxret, "downside-vol": dnvol}

def wmat(M, neutralize_beta=False):
    """monthly dollar-neutral decile L/S: LONG low-M, SHORT high-M."""
    W = pd.DataFrame(0.0, index=me, columns=px.columns)
    for d in me:
        s = M.loc[d].where(elig.loc[d]); s = s.dropna()
        if len(s) < 100: continue
        n = max(1, int(len(s)*0.10))
        w = pd.Series(0.0, index=s.index)
        w[s.nsmallest(n).index] = 0.5/n; w[s.nlargest(n).index] = -0.5/n     # long LOW risk
        if neutralize_beta:
            b = beta.loc[d].reindex(w.index).fillna(1.0).values; wv = w.values
            wv = wv - b*(wv@b)/(b@b+1e-9); g=np.abs(wv).sum()
            if g>0: wv = wv*(2.0/g); w = pd.Series(wv, index=w.index)
        W.loc[d, w.index] = w.values
    return W

# momentum streams for correlation
def toM(s): s=pd.Series(s).dropna(); s.index=pd.DatetimeIndex(s.index).to_period("M"); return s[~s.index.duplicated()]
MOM = toM(pickle.load(open("/tmp/mom_champ.pkl","rb"))["n1"]).shift(1) if os.path.exists("/tmp/mom_champ.pkl") else None
DM  = toM(pickle.load(open("/tmp/dm_streams.pkl","rb"))[1]) if os.path.exists("/tmp/dm_streams.pkl") else None

def evalbook(W, tag):
    sig = [d for d in W.index if W.loc[d].abs().sum() > 1e-9]
    r = BACKTEST.backtest(W, synth, freq=12, lag=0, signal_dates=sig, transaction_cost=tc, borrow_fee=bf)
    rr = toM(r["returns"])
    cM = pd.DataFrame({"a":rr,"b":MOM}).dropna().corr().iloc[0,1] if MOM is not None else np.nan
    cD = pd.DataFrame({"a":rr,"b":DM}).dropna().corr().iloc[0,1] if DM is not None else np.nan
    print(f"  {tag:20}{r['sharpe']:>6.2f}{r['ann_return']:>7.1%}{r['max_drawdown']:>7.1%}{r['ann_turnover']:>6.1f}"
          f"   corrMOM {cM:+.2f}  corrDM {cD:+.2f}", flush=True)
    return rr

print("="*84); print("LOW-VOL / DEFENSIVE sleeve — signal scan (monthly L/S, long low-risk, tiered net)")
print("="*84)
print(f"  {'signal':20}{'SR':>6}{'ann':>7}{'maxDD':>7}{'turn':>6}   correlation to momentum")
for nm, M in SIG.items(): evalbook(wmat(M), nm)
# composite = mean cross-sectional rank of the signals (all low=good)
comp = sum(M.rank(axis=1, pct=True) for M in SIG.values())
evalbook(wmat(comp), "COMPOSITE")
Wbab = wmat(beta, neutralize_beta=True)
bab_ret = evalbook(Wbab, "BAB (beta-neutral)")
pickle.dump(Wbab, open("/tmp/bab_weights.pkl", "wb"))                       # for the 3-sleeve portfolio test
pickle.dump(bab_ret, open("/tmp/bab_stream.pkl", "wb"))
print("[done] saved BAB weights + stream")
