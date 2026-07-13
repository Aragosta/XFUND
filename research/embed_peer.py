#!/usr/bin/env python3
"""embed_peer.py — 13F co-holding PEER MOMENTUM. The first NON-price-STRUCTURE signal.

13F embeddings (PPMI+SVD on institutional co-holding, 64-dim, quarterly PIT) define WHO moves together —
a stock's economic neighbours in smart-money portfolios. Peer lead-lag (Ali-Hirshleifer 'connected firms'):
neighbours' recent return predicts own next-month return. The LINKAGE is non-price (13F), so the signal is
orthogonal to own-stock momentum. Signal_i = cosine-sim-weighted mean of top-K peers' past return.
PIT: use the latest 13F snapshot filed >=60d ago. L/S decile, beta-neut, net via BACKTEST. Corr to MOM+DM alpha."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from scipy.stats import skew
import BACKTEST, BETANEUT

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)

EMB = pickle.load(open("data/13f/emb_vectors_focused.pkl","rb"))
snaps = sorted(EMB.keys())                                                    # quarter-end embedding dates (PIT)
def pit_snapshot(d):                                                          # latest 13F filed >=60d before d
    avail = [s for s in snaps if s <= d - pd.Timedelta(days=60)]
    return avail[-1] if avail else None

K = 50                                                                        # top-K peers
def peer_signal(d, lookback=1):
    s = pit_snapshot(d)
    if s is None: return None
    E = EMB[s]; cols = [c for c in E.index if c in px.columns]                # embedding names in our universe
    live = elig.loc[d]; cols = [c for c in cols if bool(live.get(c, False))]
    if len(cols) < 60: return None
    V = E.loc[cols].values.astype(float); V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-12)
    S = V @ V.T; np.fill_diagonal(S, -np.inf)                                 # cosine sim, mask self
    idx = np.argpartition(-S, K, axis=1)[:, :K]                              # top-K neighbours per stock
    W = np.zeros_like(S); rows = np.arange(len(cols))[:, None]
    W[rows, idx] = np.clip(S[rows, idx], 0, None)                            # keep only positive sims
    W = W / (W.sum(axis=1, keepdims=True) + 1e-12)
    # peers' PAST return over `lookback` months (known at d)
    past = (m_px.loc[d] / m_px.loc[me[me.get_loc(d) - lookback]] - 1.0)
    r = past.reindex(cols).values; ok = np.isfinite(r); r = np.where(ok, r, np.nan)
    sig = np.nansum(W * np.where(np.isfinite(r)[None, :], r[None, :], 0.0), axis=1)
    return pd.Series(sig, index=cols)

def build(lookback):
    W = pd.DataFrame(0.0, index=me, columns=px.columns)
    for d in me:
        if me.get_loc(d) < 1: continue
        sg = peer_signal(d, lookback)
        if sg is None: continue
        sg = sg.dropna()
        if len(sg) < 40: continue
        n = max(1, int(len(sg) * 0.10))
        W.loc[d, sg.nlargest(n).index] = 1.0/n; W.loc[d, sg.nsmallest(n).index] = -1.0/n
    return BETANEUT.betaneut(W, BETA)

def stream(W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)
    s = pd.Series(r["returns"]); s.index = pd.DatetimeIndex(s.index); return s[s.index >= "2013-09-01"].dropna()
def show(name, x):
    x = x.dropna(); e = (1+x).cumprod()
    print(f"  {name:22} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}  skew {skew(x):>+5.2f}")
    print("    " + " ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+5.0%}" for y in range(2014, 2027)))

print("="*100); print("13F PEER-MOMENTUM SLEEVE (co-holding embeddings, beta-neut) — net, 2013Q3+")
al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
ar = stream(al); show("ALPHA (MOM+DM meta)", ar)
for lb in (1, 3, 6, 12):
    pr = stream(build(lb))
    show(f"PEER-MOM lb={lb}", pr)
    c = pd.DataFrame({"p":pr,"a":ar}).dropna(); print(f"      corr(PEER, ALPHA) = {c.corr().iloc[0,1]:+.2f}  ({len(c)} months)")
