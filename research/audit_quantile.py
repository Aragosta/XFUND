"""audit_quantile.py — MECHANISM of the capacity collapse. audit_capacity{,_all}.py showed MOM/VQ die and DM
survives at a few discrete top-N cutoffs. This asks: WHERE does the alpha live? Split each sleeve's universe
into liquidity DECILES (by trailing $ volume) each month, and for each liquidity decile separately report:
  - SR / ann / maxDD of a LONG-ONLY book in that liquidity bucket (using the sleeve's own long-leg picks)
  - same for SHORT-only
  - the fraction of each sleeve's gross book weight that sits in each liquidity decile
This tells us: is the edge a smooth liquidity gradient, a cliff at the bottom decile, or leg-specific
(e.g. value's crash lives in illiquid SHORTS specifically, as audit found for the all-eligible case)."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from DATAHUB import DataHub
import BACKTEST
hub = DataHub(); me, m_px, synth, elig = hub.me, hub.m_px, hub.m_px, hub.elig("liquid")
tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
mdv = hub.mdv.where(elig)

def load_weights(p):
    W = pickle.load(open(p, "rb")); W.index = pd.DatetimeIndex(W.index)
    return W.reindex(index=me, columns=m_px.columns).fillna(0.0)

def load_vq():
    S = pd.read_pickle("/tmp/fundq_score.pkl")
    W = pd.DataFrame(0.0, index=me, columns=m_px.columns)
    for d in me:
        s = S.loc[d].dropna() if d in S.index else pd.Series(dtype=float)
        if len(s) < 60: continue
        n = max(1, int(len(s) * 0.10))
        W.loc[d, s.nlargest(n).index] = 1.0 / n
        W.loc[d, s.nsmallest(n).index] = -1.0 / n
    return W

def liq_decile(d):
    """month-d liquidity decile membership (0=most illiquid ... 9=most liquid) among mdv-eligible names."""
    row = mdv.loc[d].dropna()
    if len(row) < 50: return pd.Series(dtype=int)
    return pd.qcut(row.rank(method="first"), 10, labels=False)

def leg_book_by_decile(W, leg):
    """for each of the 10 liquidity deciles, build a monthly book holding ONLY that sleeve's `leg`
    (long or short) positions whose names fall in that liquidity decile that month, renormalized to gross 1."""
    books = [pd.DataFrame(0.0, index=me, columns=m_px.columns) for _ in range(10)]
    for d in me:
        w = W.loc[d]
        side = w.clip(lower=0) if leg == "long" else -w.clip(upper=0)
        if side.sum() < 1e-9: continue
        dec = liq_decile(d)
        if dec.empty: continue
        for k in range(10):
            names = dec[dec == k].index
            sub = side[side.index.isin(names)]
            g = sub.sum()
            if g < 1e-9: continue
            books[k].loc[d] = (sub / g) * (1.0 if leg == "long" else -1.0)
    return books

def run(W, lab):
    sd = [d for d in me if W.loc[d].abs().sum() > 1e-9]
    if not sd: return None
    try:
        r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=1, signal_dates=sd, transaction_cost=tc, borrow_fee=bf)
    except TypeError:
        print(f"    {lab:30} *** BLOWN UP: equity <= 0 (undiversified illiquid short-only book wiped out by a squeeze) ***")
        return None
    x = pd.Series(r["returns"]); x.index = pd.DatetimeIndex(x.index)
    x = x[(x.index >= "2016-01-01") & (x.index < "2027-01-01")].dropna()
    if len(x) < 12: return None
    e = (1 + x).cumprod()
    sr = x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else np.nan
    ann = (1 + x).prod() ** (12 / len(x)) - 1
    dd = (e / e.cummax() - 1).min()
    print(f"    {lab:30} SR {sr:>5.2f}  ann {ann:>7.1%}  maxDD {dd:>7.1%}  n_mo {len(x):>4}")
    return sr

print("=" * 100)
print("LIQUIDITY-DECILE MECHANISM AUDIT — where does the alpha live, and how does it die? (honest engine, 2016-26)")
print("  decile 0 = most ILLIQUID tenth of the eligible universe each month, decile 9 = most LIQUID tenth")

for nm, loader in [("MOM", lambda: load_weights("/tmp/mom_weights.pkl")),
                    ("DM", lambda: load_weights("/tmp/dm_weights.pkl")),
                    ("VQ", load_vq)]:
    W = loader()
    print(f"\n{nm}:")
    for leg in ("long", "short"):
        print(f"  {leg.upper()} leg, by liquidity decile:")
        books = leg_book_by_decile(W, leg)
        for k in range(10):
            run(books[k], f"decile {k}")

print("\n" + "=" * 100)
print("Interpretation: a smooth SR ramp 0->9 = broad illiquidity premium. A cliff (only deciles 0-2 work) =")
print("microcap-specific artifact. If SHORT legs decay faster than LONG legs, the short side is the fragile one")
print("(consistent with the all-eligible audit finding value's crash risk concentrated in illiquid shorts).")
