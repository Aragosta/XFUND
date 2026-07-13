#!/usr/bin/env python3
"""RISK.py — DOOR 2: a real multi-factor risk model for CONSTRUCTION (not more alpha).

Today's neutralization strips ONE factor (equal-weight market beta). But a momentum book loads on many common
risks (sectors, size, crowding) whose co-movement is FORECASTABLE (2nd moment, R^2~55%) even though direction
isn't. This builds a leak-free STATISTICAL factor model (PCA on the trailing return panel -> K orthonormal
factor directions = the common-risk axes) and neutralizes the book flat against ALL K, w' = w - B(B'w), renorm
gross 2.0. Also a 13F-AUGMENTED variant: append co-holding embedding directions as extra factors (co-holding =
shared institutional-flow risk; embeddings had zero alpha but real covariance -> exactly a risk factor).
Test on the SAME raw MOM+DM book -> does better construction raise net SR / cut maxDD, no new alpha?"""
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

# ---- raw dollar-neutral alpha book = MOM + DM combined at name level ----
MOM = pickle.load(open("/tmp/mom_weights.pkl","rb")); DM = pickle.load(open("/tmp/dm_weights.pkl","rb"))
MOM.index = pd.DatetimeIndex(MOM.index); DM.index = pd.DatetimeIndex(DM.index)
def dnorm(W):                                                                 # dollar-neutral, gross 2.0
    W = W.reindex(index=me, columns=px.columns).fillna(0.0)
    return W.div(W.abs().sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0) * 2.0
RAW = (dnorm(MOM) + dnorm(DM)); RAW = RAW.div(RAW.abs().sum(axis=1).replace(0,np.nan), axis=0).fillna(0.0) * 2.0

EMB = pickle.load(open("data/13f/emb_vectors_focused.pkl","rb")); snaps = sorted(EMB.keys())
def pit(d):
    a = [s for s in snaps if s <= d - pd.Timedelta(days=60)]; return a[-1] if a else None

WIN = 36                                                                      # trailing months for factor est
def factor_neut(W, K, win=WIN, use_emb=False):
    out = pd.DataFrame(0.0, index=W.index, columns=W.columns)
    Rm = mret.values; idx = list(me)
    for d in W.index:
        w = W.loc[d]
        if w.abs().sum() < 1e-9: continue
        i = idx.index(d)
        if i < win: out.loc[d] = w; continue
        hist = mret.iloc[i-win+1:i+1]                                         # trailing win months, through d (leak-free vs fwd)
        good = hist.columns[(hist.notna().sum(axis=0) >= win-2) & (elig.loc[d].reindex(hist.columns).fillna(False))]
        good = [c for c in good if abs(w.get(c,0.0)) > 0 or True]             # full eligible set for a clean factor space
        R = hist[good].fillna(0.0).values.astype(float); R = R - R.mean(axis=0, keepdims=True)
        # statistical factors = top-K right singular vectors (orthonormal cross-sectional loadings)
        U,S,Vt = np.linalg.svd(R, full_matrices=False); B = Vt[:K].T                      # (Ngood x K)
        if use_emb:                                                           # append 13F co-holding directions
            s = pit(d)
            if s is not None:
                E = EMB[s]; common = [c for c in good if c in E.index]
                if len(common) >= 30:
                    Bemb = np.zeros((len(good), min(20, E.shape[1])))
                    ev = E.loc[common].values.astype(float); ev = ev - ev.mean(0)
                    Ue,Se,Vte = np.linalg.svd(ev, full_matrices=False)        # top embedding directions in stock-space
                    pos = {c:j for j,c in enumerate(good)}
                    for j,c in enumerate(common): Bemb[pos[c], :] = Ue[j, :Bemb.shape[1]]
                    B = np.column_stack([B, Bemb])
        B,_ = np.linalg.qr(B)                                                 # orthonormalize combined factor basis
        wv = w.reindex(good).fillna(0.0).values
        wv = wv - B @ (B.T @ wv)                                              # project flat against all factors
        neu = w.copy()*0.0; neu.loc[good] = wv
        g = neu.abs().sum(); out.loc[d] = neu*(2.0/g) if g>0 else neu
    return out

def stream(W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)
    s = pd.Series(r["returns"]); s.index = pd.DatetimeIndex(s.index); return s.dropna()
def show(name, x):
    x = x[(x.index>="2011-06-01")&(x.index<"2023-01-01")].dropna(); e=(1+x).cumprod()      # honest window, cap fantasy tail
    print(f"  {name:26} ann {(1+x).prod()**(12/len(x))-1:>6.1%}  SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  vol {x.std()*np.sqrt(12):>5.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}  skew {skew(x):>+5.2f}")

print("="*104); print("DOOR 2 — MULTI-FACTOR RISK MODEL vs single-market-beta neutralization (raw MOM+DM book) — net, 2011-2022")
show("raw dollar-neutral", stream(RAW))
show("baseline: 1-factor market beta", stream(BETANEUT.betaneut(RAW, BETA)))
for K in (5, 10, 20):
    show(f"stat factors K={K}", stream(factor_neut(RAW, K)))
show("stat K=10 + 13F emb factors", stream(factor_neut(RAW, 10, use_emb=True)))
