#!/usr/bin/env python3
"""embed_char.py — do the 13F embedding COORDINATES themselves predict returns (a non-price characteristic)?
Peer-momentum failed (signal=price). The truly non-price object is the 64-dim vector. Test: pooled ridge of
next-month cross-sectional return on the 64 embedding dims, trained on history < t (leak-free), predict t.
Report IC + L/S decile SR. If IC~0 -> embeddings carry no DIRECT alpha; their value is a RISK MODEL (Door 2)."""
import warnings; warnings.filterwarnings("ignore")
import pickle, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
import BACKTEST, BETANEUT

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
EMB = pickle.load(open("data/13f/emb_vectors_focused.pkl","rb")); snaps = sorted(EMB.keys())
def pit(d):
    a = [s for s in snaps if s <= d - pd.Timedelta(days=60)]; return a[-1] if a else None

fwd = mret.shift(-1)                                                          # next-month return (target)
# assemble a long panel: (date, ticker) -> 64 dims + fwd, z-scored fwd per month
rows = []
for d in me:
    s = pit(d)
    if s is None: continue
    E = EMB[s]; cols = [c for c in E.index if c in px.columns and bool(elig.loc[d].get(c, False))]
    if len(cols) < 60: continue
    X = E.loc[cols].values.astype(float); y = fwd.loc[d, cols].values.astype(float)
    ok = np.isfinite(y);
    if ok.sum() < 40: continue
    yz = (y - np.nanmean(y[ok])) / (np.nanstd(y[ok]) + 1e-9)                  # cross-sectional z of fwd ret
    for i in np.where(ok)[0]:
        rows.append((d, cols[i], *X[i], yz[i]))
P = pd.DataFrame(rows, columns=["d","tic"]+[f"e{i}" for i in range(64)]+["y"])
EF = [f"e{i}" for i in range(64)]
print(f"[panel] {len(P)} obs, {P['d'].nunique()} months")

# leak-free walk-forward: train pooled on months strictly before t, predict month t
W = pd.DataFrame(0.0, index=me, columns=px.columns); ics = []
mons = sorted(P["d"].unique())
for t_ in mons:
    tr = P[P["d"] < t_]
    if tr["d"].nunique() < 12: continue
    r = Ridge(alpha=50.0).fit(tr[EF].values, tr["y"].values)
    cur = P[P["d"] == t_]; pred = r.predict(cur[EF].values)
    ics.append(np.corrcoef(pred, cur["y"].values)[0,1])
    sg = pd.Series(pred, index=cur["tic"].values)
    n = max(1, int(len(sg)*0.10))
    W.loc[t_, sg.nlargest(n).index] = 1.0/n; W.loc[t_, sg.nsmallest(n).index] = -1.0/n
ics = np.array(ics); print(f"[embed-characteristic] mean IC {np.nanmean(ics):+.4f}  t-stat {np.nanmean(ics)/ (np.nanstd(ics)/np.sqrt(len(ics))):+.2f}  ({len(ics)} months)")
Wn = BETANEUT.betaneut(W, BETA)
r = BACKTEST.backtest(Wn.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in Wn.index if Wn.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)
x = pd.Series(r["returns"]); x.index = pd.DatetimeIndex(x.index); x = x[x.index>="2014-06-01"].dropna()
e=(1+x).cumprod(); print(f"  EMBED-CHAR L/S    ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  maxDD {(e/e.cummax()-1).min():>6.1%}")
al = pickle.load(open("/tmp/meta_weights.pkl","rb")); al.index = pd.DatetimeIndex(pd.to_datetime(al.index)); al = al[~al.index.duplicated()].reindex(index=me, columns=px.columns).fillna(0.0)
ar = BACKTEST.backtest(al, synth, freq=12, lag=0, signal_dates=[d for d in al.index if al.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)["returns"]
ar = pd.Series(ar); ar.index = pd.DatetimeIndex(ar.index)
c = pd.DataFrame({"e":x,"a":ar}).dropna(); print(f"  corr(EMBED-CHAR, ALPHA) = {c.corr().iloc[0,1]:+.2f}")
