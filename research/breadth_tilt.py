#!/usr/bin/env python3
"""breadth_tilt.py — adjudicate the user's claim WITH BREADTH: with MANY models, does tilting toward the
trailing-6mo winners beat static? (Momentum Sharpe ~ IC*sqrt(N); my 2-3 sleeve tests were breadth-starved.)
Build ~30 momentum/reversal parameterizations -> monthly gross streams -> compare STATIC (equal/ERC) vs
6mo-return TILT across all of them. Leak-free. Settles whether strategy-momentum works once N is large."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd

px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2004-01-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z.abs()<0.8)
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252,min_periods=200).mean().reindex(me,method="ffill")
elig = (m_px>5)&(cov>0.9)&(mdv>5e6); fwd = mret.shift(-1)

def factor_stream(lb, skip, q, rev=False):
    sig = m_px.shift(skip)/m_px.shift(lb) - 1.0                                  # past return lb..skip
    S = sig.where(elig); R = S.rank(axis=1, pct=True)
    hi = (R>=1-q).astype(float); lo = (R<=q).astype(float)
    long_,short = (lo,hi) if rev else (hi,lo)                                    # mom: long winners; rev: long losers
    L = long_.div(long_.sum(1).replace(0,np.nan),axis=0); Sh = short.div(short.sum(1).replace(0,np.nan),axis=0)
    W = (L - Sh).fillna(0.0)
    return (W.shift(1) * mret).sum(axis=1)                                       # gross monthly, lag 1

# ~30 parameterizations (momentum family + a few reversals)
specs = []
for lb in (3,6,9,12):
    for skip in (0,1):
        for q in (0.1,0.2): specs.append((lb,skip,q,False))
specs += [(1,0,0.1,True),(1,0,0.2,True),(2,0,0.1,True),(2,1,0.1,False),(4,1,0.1,False),(18,1,0.1,False)]
STR = pd.DataFrame({f"m{lb}_{skip}_{int(q*100)}{'r' if rev else ''}": factor_stream(lb,skip,q,rev) for lb,skip,q,rev in specs})
STR = STR.loc["2005-01-01":].dropna(how="all"); n = STR.shape[1]
print(f"[breadth] {n} model streams, {len(STR)} months, avg pairwise corr {STR.corr().values[np.triu_indices(n,1)].mean():.2f}")

def perf(c,w0="2005-06-01",w1="2027-01-01"):
    x=c[(c.index>=w0)&(c.index<w1)].dropna(); e=(1+x).cumprod()
    return x.mean()/x.std()*np.sqrt(12), (e/e.cummax()-1).min()
def run(mode, win=6, lam=None, warm=24):
    W=pd.DataFrame(np.nan,index=STR.index,columns=STR.columns)
    for i in range(warm,len(STR)):
        H=STR.iloc[:i]
        if mode=="equal": w=np.ones(n)/n
        elif mode=="invvol": iv=1/(H.iloc[-36:].std().values+1e-9); w=iv/iv.sum()
        elif mode=="tilt":                                                      # tilt toward trailing-`win` winners
            cr=(1+H.iloc[-win:]).prod().values-1; e=np.exp(lam*(cr-cr.mean())); w=e/e.sum()
        W.iloc[i]=w
    c=(STR*W).sum(axis=1); return c[W.notna().all(axis=1)]

print("\nSTATIC vs TILT across the model ensemble (gross, 2005-2026):")
for m in ("equal","invvol"): s,d=perf(run(m)); print(f"  static {m:8}            SR {s:>5.2f}  maxDD {d:>6.1%}")
print("  --- tilt toward trailing-6mo winners, strength sweep ---")
for lam in (2,5,10,20,40):
    s,d=perf(run("tilt",win=6,lam=lam)); print(f"  6mo-tilt  lam={lam:>2}          SR {s:>5.2f}  maxDD {d:>6.1%}")
print("  --- tilt lookback sweep (lam=10) ---")
for win in (3,6,12,18):
    s,d=perf(run("tilt",win=win,lam=10)); print(f"  tilt win={win:>2}mo  (lam10)     SR {s:>5.2f}  maxDD {d:>6.1%}")
