#!/usr/bin/env python3
"""RIGOROUS factor-momentum (upgrade of the crude PCA sketch).

Replaces statistical eigenvectors with a real factor risk model: each stock's trailing return
is decomposed by causal rolling regression on FF5 (Mkt,SMB,HML,RMW,CMA) + 9 industry factors
(SPDR sector ETFs, market-relative). common = factor-explained, specific = residual.
Then: form momentum on common / common-ex-market / specific, backtest through the cost engine,
and CERTIFY the common-momentum sleeve with an alpha regression on FF5+MOM+STR (HAC t-stats)."""
import warnings; warnings.filterwarnings("ignore")
import io, zipfile, requests
import numpy as np, pandas as pd
from numpy.linalg import lstsq
import deep_momentum_xgb as d
import BACKTEST
import statsmodels.api as sm

Q, L, MIN_T = 0.05, 36, 132          # decile tails, 36m estimation window, start month index
pd.set_option("display.width", 200)

# ── data ──
print("[load] monthly universe ...", flush=True)
prices_monthly, rets_monthly, size_monthly = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
T = len(rets_monthly); dates = rets_monthly.index
eligible, shortable = d.compute_eligibility(prices_monthly, size_monthly, min_dollar_vol_abs=5e6)
pnl_prices = d._build_pnl_prices(prices_monthly)
tcost = BACKTEST.tiered_transaction_costs(size_monthly); bfee = BACKTEST.tiered_borrow_fees(size_monthly)
ym = dates.to_period("M")

def ff(url):
    raw = zipfile.ZipFile(io.BytesIO(requests.get(url, timeout=60).content))
    raw = raw.read(raw.namelist()[0]).decode("latin-1").splitlines()
    idx, vals = [], []
    for ln in raw:
        p = [x.strip() for x in ln.split(",")]
        if len(p) >= 2 and len(p[0]) == 6 and p[0].isdigit():
            idx.append(pd.Period(p[0][:4] + "-" + p[0][4:], "M")); vals.append([float(x) for x in p[1:]])
    return pd.DataFrame(vals, index=idx)

print("[ff] downloading factors ...", flush=True)
f5 = ff("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip")
f5.columns = ["MktRF", "SMB", "HML", "RMW", "CMA", "RF"]
mom = ff("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Momentum_Factor_CSV.zip"); mom.columns = ["MOM"]
strv = ff("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_ST_Reversal_Factor_CSV.zip"); strv.columns = ["STR"]
FF = pd.concat([f5, mom, strv], axis=1) / 100.0
FF = FF.reindex(ym).reset_index(drop=True); FF.index = dates           # align to our month-ends

print("[etf] downloading 9 sector ETFs ...", flush=True)
SECT = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]
spx = {}
for tk in SECT:
    try:
        s = d.fetch_tiingo_monthly(tk, "1999-01-01", dates[-1].strftime("%Y-%m-%d")).iloc[:, 0]
        sm = s.resample("ME").last().pct_change()
        sm.index = sm.index.to_period("M")                 # align by period, not exact timestamp
        spx[tk] = sm
    except Exception as e:
        print(f"  [warn] {tk}: {e}")
sect = pd.DataFrame(spx).reindex(ym); sect.index = dates    # period-align then assign our dates
mkt_tot = (FF["MktRF"] + FF["RF"])
ind = sect.sub(mkt_tot, axis=0)                                        # industry = sector minus market

# factor design columns used IN THE DECOMPOSITION (no momentum/STR — those are what we form)
DEC = pd.concat([FF[["MktRF", "SMB", "HML", "RMW", "CMA"]], ind], axis=1)
DEC.columns = ["MktRF", "SMB", "HML", "RMW", "CMA"] + [f"IND_{s}" for s in SECT]
nfac = DEC.shape[1]
print(f"[decomp] {nfac} factors, {DEC.dropna().shape[0]} usable months", flush=True)

# ── causal rolling decomposition + signal construction ──
sig = {k: {} for k in ["total", "common", "common_xmkt", "specific"]}
mom_pos = slice(L - 12, L - 1)                                          # months t-12..t-2 within the L-window
for t in range(max(MIN_T, L + 1), T):
    Fwin = DEC.iloc[t - L:t]
    if Fwin.isna().any().any():
        continue
    e = eligible.iloc[t - 1]
    Rwin = rets_monthly.iloc[t - L:t]
    cols = Rwin.columns[Rwin.notna().all().values & e.reindex(Rwin.columns).fillna(False).values]
    if len(cols) < 50:
        continue
    Fd = np.column_stack([np.ones(L), Fwin.to_numpy()])                # design w/ intercept (L x nfac+1)
    R = Rwin[cols].to_numpy()                                          # (L x n)
    beta, *_ = lstsq(Fd, R, rcond=None)                               # (nfac+1 x n)
    sumF = Fd[mom_pos].sum(0)                                          # cumulative factor exposure over mom window
    common      = sumF[1:] @ beta[1:]                                  # all factors (ex intercept)
    common_xmkt = sumF[2:] @ beta[2:]                                  # ex market (drop MktRF col)
    total       = R[mom_pos].sum(0)                                    # raw 12-1 momentum
    specific    = total - common
    sd = dates[t - 1]; sh = shortable.iloc[t - 1]; shset = set(sh.index[sh.values]); nq = max(1, int(len(cols) * Q))
    for nm, vals in [("total", total), ("common", common), ("common_xmkt", common_xmkt), ("specific", specific)]:
        s = pd.Series(vals, index=cols)
        w = pd.Series(0.0, index=cols); w[s.nlargest(nq).index] = 1.0 / nq
        ssub = s.loc[[c for c in s.index if c in shset]]; w[ssub.nsmallest(nq).index] = -1.0 / nq
        sig[nm][sd] = w

W = {k: pd.DataFrame(v).T.fillna(0.0).sort_index() for k, v in sig.items()}
print(f"[signals] {len(W['common'])} months  ({W['common'].index[0].date()}–{W['common'].index[-1].date()})", flush=True)

def run(weights):
    first = weights.index[0]; px = pnl_prices.loc[first:]
    s = [x for x in weights.index if x in px.index]
    return BACKTEST.backtest(weights.reindex(columns=px.columns).fillna(0.0), px,
                             freq=12, lag=0, transaction_cost=tcost, borrow_fee=bfee, signal_dates=s)

res = {k: run(W[k]) for k in W}
def perf(x):
    x = x.dropna(); ann = (1 + x).prod() ** (12 / len(x)) - 1
    vol = x.std() * np.sqrt(12); dn = x[x < 0].std() * np.sqrt(12)
    eq = (1 + x).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return ann, (x.mean() * 12) / vol, (x.mean() * 12) / dn, dd

print("\n" + "=" * 80 + "\n=== RIGOROUS FACTOR-MOMENTUM (real-factor decomposition, q=0.05) ===")
print(f"{'sleeve':20}{'ann':>8}{'sharpe':>8}{'sortino':>8}{'maxDD':>9}")
for nm, lab in [("total", "Total 12-1"), ("common", "Common (FF5+ind)"),
                ("common_xmkt", "Common ex-market"), ("specific", "Stock-specific")]:
    a, sh, so, dd = perf(pd.Series(res[nm]["returns"]))
    print(f"{lab:20}{a:>8.1%}{sh:>8.2f}{so:>8.2f}{dd:>9.1%}")

# ── ALPHA CERTIFICATION: regress common-momentum L/S on FF5+MOM+STR (HAC) ──
print("\n=== ALPHA REGRESSION — Common-ex-market momentum on FF5+MOM+STR (Newey-West) ===")
y = pd.Series(res["common_xmkt"]["returns"]).dropna()
Xf = FF.loc[y.index, ["MktRF", "SMB", "HML", "RMW", "CMA", "MOM", "STR"]]
keep = Xf.dropna().index; y = y.loc[keep]; Xf = Xf.loc[keep]
m = sm.OLS(y.values, sm.add_constant(Xf.values)).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
names = ["alpha", "MktRF", "SMB", "HML", "RMW", "CMA", "MOM", "STR"]
print(f"{'term':8}{'coef':>10}{'t-stat':>9}")
for nm, c, tval in zip(names, m.params, m.tvalues):
    star = "***" if abs(tval) > 2.6 else ("**" if abs(tval) > 1.96 else "")
    sc = c * 12 if nm == "alpha" else c
    print(f"{nm:8}{sc:>10.4f}{tval:>9.2f} {star}" + ("   (annualized alpha)" if nm == "alpha" else ""))
print(f"\nR² = {m.rsquared:.3f}   n = {len(y)} months")
print("\n[done]", flush=True)
