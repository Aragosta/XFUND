#!/usr/bin/env python3
"""PCA momentum decomposition (De Boer-Gao-Montminy 2025 on our universe).

Decompose each stock's trailing 12-1m return into common (factor/industry/style) vs
stock-specific via asymptotic PCA (Connor-Korajczyk, N>>T), split common into beta (PC1)
vs the rest (PC2..K = style/industry proxy), and backtest each momentum sleeve plus the
specific short-term-reversal hedge. Tests: does common(ex-beta) momentum persist & avoid
crashes, while stock-specific momentum carries the crash and reverses short-term?"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from numpy.linalg import eigh, lstsq
import deep_momentum_xgb as d
import BACKTEST

K, Q, WIN, START_T = 5, 0.05, 12, 36
pd.set_option("display.width", 200)

print("[load] tiingo ...", flush=True)
prices_monthly, rets_monthly, size_monthly = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
T = len(rets_monthly); dates = rets_monthly.index
eligible, shortable = d.compute_eligibility(prices_monthly, size_monthly, min_dollar_vol_abs=5e6)
pnl_prices = d._build_pnl_prices(prices_monthly)
tcost = BACKTEST.tiered_transaction_costs(size_monthly)
bfee  = BACKTEST.tiered_borrow_fees(size_monthly)
print(f"[data] {T} months, {rets_monthly.shape[1]} tickers, {dates[0].date()}–{dates[-1].date()}", flush=True)

names = ["total", "common", "common_xbeta", "beta", "specific", "specific_rev"]
sigs = {nm: {} for nm in names}
univ = []
for t in range(START_T, T):                       # hold month t, signal date t-1
    win = rets_monthly.iloc[t - WIN:t]            # rows = months t-12 .. t-1  (12, N)
    e = eligible.iloc[t - 1]
    ok = win.notna().all().values & e.reindex(win.columns).fillna(False).values
    cols = win.columns[ok]
    if len(cols) < 50:
        continue
    Rw = win[cols].to_numpy()                     # (12, n)
    n = Rw.shape[1]; univ.append(n)
    Omega = Rw @ Rw.T / n                          # (12,12) asymptotic-PCA
    _, V = eigh(Omega)
    F = V[:, ::-1][:, :K]                          # (12,K) top-K factor returns
    B, *_ = lstsq(F, Rw, rcond=None)              # (K,n) loadings
    common = F @ B                                 # (12,n)
    resid = Rw - common
    pc1 = np.outer(F[:, 0], B[0, :])               # market/beta component
    common_xbeta = common - pc1                    # PC2..K = style/industry proxy
    m = slice(0, WIN - 1)                          # t-12..t-2 (skip last month)
    rev = WIN - 1                                  # last month = t-1
    raw = {
        "total":        Rw[m].sum(0),
        "common":       common[m].sum(0),
        "common_xbeta": common_xbeta[m].sum(0),
        "beta":         pc1[m].sum(0),
        "specific":     resid[m].sum(0),
        "specific_rev": -resid[rev],               # reverse last-month specific return
    }
    sd = dates[t - 1]
    sh = shortable.iloc[t - 1]; shset = set(sh.index[sh.values])
    nq = max(1, int(n * Q))
    for nm, vals in raw.items():
        s = pd.Series(vals, index=cols)
        w = pd.Series(0.0, index=cols); w[s.nlargest(nq).index] = 1.0 / nq
        ssub = s.loc[[c for c in s.index if c in shset]]
        w[ssub.nsmallest(nq).index] = -1.0 / nq
        sigs[nm][sd] = w

W = {nm: pd.DataFrame(r).T.fillna(0.0).sort_index() for nm, r in sigs.items()}
print(f"[univ] avg names/month={np.mean(univ):.0f}  signal months={len(W['total'])}  "
      f"({W['total'].index[0].date()}–{W['total'].index[-1].date()})", flush=True)

def run(weights):
    first = weights.index[0]; px = pnl_prices.loc[first:]
    s = [x for x in weights.index if x in px.index]
    return BACKTEST.backtest(weights.reindex(columns=px.columns).fillna(0.0), px,
                             freq=12, lag=0, transaction_cost=tcost, borrow_fee=bfee, signal_dates=s)

res = {nm: run(W[nm]) for nm in names}
rr = pd.DataFrame({nm: pd.Series(res[nm]["returns"]) for nm in names}).dropna()
corr = rr.corr()["total"]

def perf(r):
    r = r.dropna()
    if len(r) < 6: return (np.nan,) * 4
    ann = (1 + r).prod() ** (12 / len(r)) - 1
    vol = r.std() * np.sqrt(12); dn = r[r < 0].std() * np.sqrt(12)
    eq = (1 + r).cumprod(); dd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol, (r.mean() * 12) / dn, dd

label = {"total": "Total 12-1 (≈bench)", "common": "Common (all PCs)",
         "common_xbeta": "Common ex-beta (style/ind)", "beta": "Beta (PC1) momentum",
         "specific": "Stock-specific mom", "specific_rev": "Specific 1m REVERSAL"}

def table(title, sl):
    print("\n" + "=" * 92 + f"\n=== {title} ===")
    print(f"{'sleeve':28}{'ann':>8}{'sharpe':>8}{'sortino':>8}{'maxDD':>9}{'corr→tot':>9}")
    for nm in names:
        r = pd.Series(res[nm]["returns"])
        if sl is not None: r = r.loc[sl:]
        a, sh, so, dd = perf(r)
        print(f"{label[nm]:28}{a:>8.1%}{sh:>8.2f}{so:>8.2f}{dd:>9.1%}{corr[nm]:>9.2f}")

table("FULL SAMPLE (incl. 2008-09 momentum crash)", None)
table("2011-07 onward (comparable to DM runs)", "2011-07-01")

# crash check: worst 6 months of total momentum, what each sleeve did
worst = pd.Series(res["total"]["returns"]).sort_values().head(6)
print("\n=== 6 WORST MONTHS OF TOTAL MOMENTUM — what each sleeve returned ===")
hdr = "month".ljust(12) + "".join(f"{nm[:10]:>12}" for nm in names)
print(hdr)
for dt in worst.index:
    row = pd.Timestamp(dt).date().__str__().ljust(12)
    for nm in names:
        v = pd.Series(res[nm]["returns"]).get(dt, np.nan)
        row += f"{v:>12.1%}"
    print(row)
print("\n[done]", flush=True)
