#!/usr/bin/env python3
"""FUND.py — the FUNDAMENTAL / VALUE ML sleeve. Same rank-space XGBoost engine as MOM, pointed at fundamental
characteristics (value+quality+investment+accruals+safety+growth) with a SECTOR-NEUTRAL rank target — the design
our target tests settled: sector-neutralizing kills ~70% of value's artifact, rank/multi-horizon denoises, value
is slow (6/9/12mo). Not a standalone star (liquid-US value winter) — judged as a DIVERSIFIER (-0.40 to momentum):
does it lift the MOM/DM ensemble? All data from DataHub; PIT-lagged fundamentals; sector-neutral decile L/S, beta-neut."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import norm
from xgboost import XGBRegressor
from DATAHUB import DataHub
import BACKTEST, BETANEUT

hub = DataHub(); me, m_px, mret, synth, elig = hub.me, hub.m_px, hub.mret, hub.synth, hub.elig("liquid")
sec = hub.sector; tc = BACKTEST.tiered_transaction_costs(hub.mdv); bf = BACKTEST.tiered_borrow_fees(hub.mdv)
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
f = hub.fund; mcap = hub.mcap()
def d12(x): return x / x.shift(12) - 1                                          # ~annual growth on PIT panel

# ---- fundamental feature panel ----
feat = {
 "bm": f("book")/mcap, "ep": f("ni")/mcap, "sp": f("rev")/mcap, "cfp": f("ocf")/mcap, "dp": f("dividends")/mcap,
 "gpa": (f("gross_profit")/f("assets")).where(f("assets")>0), "roe": (f("ni")/f("book")).where(f("book")>0),
 "roa": f("ni")/f("assets"), "ocfa": f("ocf")/f("assets"), "gmar": f("gross_profit")/f("rev"),
 "ag": d12(f("assets")), "capexa": f("capex")/f("assets"), "accr": (f("ni")-f("ocf"))/f("assets"),
 "drec": f("receivables").diff(12)/f("assets"), "dinv": f("inventory").diff(12)/f("assets"),
 "lev": f("debt_lt")/f("assets"), "curr": f("assets_cur")/f("liab_cur"), "casha": f("cash")/f("assets"),
 "revg": d12(f("rev")), "nig": d12(f("ni")), "size": np.log(mcap),
}
def zc(df): z=df.sub(df.mean(1),0).div(df.std(1)+1e-9,0); return z.clip(-3,3)
F = {k: zc(v.replace([np.inf,-np.inf],np.nan)) for k,v in feat.items()}; COLS = list(F)
cover = pd.concat([v.notna() for v in feat.values()], axis=0).groupby(level=0).sum()  # feature coverage
print(f"[fund] {len(COLS)} features, coverage ~{int((mcap.notna()&elig).sum(1).mean())} names/mo", flush=True)

def grank(a): r=pd.Series(a).rank(method="average"); return norm.ppf((r-0.5)/max(len(r),2))
HZ = (6, 9, 12); EMB = max(HZ)
def sec_rank(fwd, d):                                                           # sector-neutral rank of fwd return at d
    r = fwd.loc[d].where(elig.loc[d]); r = r - r.groupby(sec).transform("mean"); return r
# pool: features (rank) + sector-neutral rank targets per month
pool = {}
for i, d in enumerate(me):
    live = elig.loc[d] & mcap.loc[d].notna()
    idx = live[live].index
    if len(idx) < 100 or i+EMB >= len(me): continue
    X = np.column_stack([F[c].loc[d].reindex(idx).fillna(0.0).values for c in COLS])
    Y = {h: grank(sec_rank(mret.shift(-h+0), me[i]).reindex(idx).values) if False else
            grank((mret.iloc[i+h].where(elig.iloc[i+h]) - mret.iloc[i+h].where(elig.iloc[i+h]).groupby(sec).transform("mean")).reindex(idx).values)
         for h in HZ}
    pool[d] = dict(X=X.astype(np.float32), idx=idx, Y=Y)

REG = dict(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
           tree_method="hist", multi_strategy="multi_output_tree", verbosity=0)
dates = [d for d in me if d in pool]; W = pd.DataFrame(0.0, index=me, columns=m_px.columns); mdl=None
SCORE = pd.DataFrame(np.nan, index=me, columns=m_px.columns)                    # full sector-neutral score (for decile analysis)
for j, d in enumerate(dates):
    if j < 60: continue                                                        # need ~5y history
    if (mdl is None) or (j % 3 == 0):                                          # retrain quarterly
        tr = [dates[k] for k in range(j) if dates[k] <= me[max(0, me.get_loc(d)-EMB)]]  # embargo = max horizon
        if len(tr) < 48: continue
        Xtr = np.vstack([pool[t]["X"] for t in tr]); Ytr = np.column_stack([np.concatenate([pool[t]["Y"][h] for t in tr]) for h in HZ])
        ok = np.isfinite(Ytr).all(1); mdl = XGBRegressor(**REG); mdl.fit(Xtr[ok], Ytr[ok])
    p = mdl.predict(pool[d]["X"]).mean(1)                                       # mean predicted rank across horizons
    s = pd.Series(p, index=pool[d]["idx"]); s = s - s.groupby(sec.reindex(s.index)).transform("mean")  # sector-neutral score
    SCORE.loc[d, s.index] = s.values
    n = max(1, int(len(s)*0.10)); W.loc[d, s.nlargest(n).index] = 1.0/n; W.loc[d, s.nsmallest(n).index] = -1.0/n
W = BETANEUT.betaneut(W, BETA)
r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum()>1e-9], transaction_cost=tc, borrow_fee=bf)
fr = pd.Series(r["returns"]); fr.index = pd.DatetimeIndex(fr.index)
def show(nm,x):
    x=x[(x.index>="2013-01-01")&(x.index<"2027-01-01")].dropna(); e=(1+x).cumprod()
    print(f"  {nm:20} SR {x.mean()/x.std()*np.sqrt(12):>5.2f}  ann {(1+x).prod()**(12/len(x))-1:>6.1%}  maxDD {(e/e.cummax()-1).min():>6.1%}")
    print("    "+" ".join(f"{y}:{(1+x[[d.year==y for d in x.index]]).prod()-1:>+4.0%}" for y in range(2013,2027)))
print("="*80); print("FUNDAMENTAL/VALUE ML (sector-neutral rank target, XGBoost) — net")
show("FUND ML", fr)
pickle.dump(r["returns"], open("/tmp/fund_returns.pkl","wb")); SCORE.to_pickle("/tmp/fund_score.pkl")

# ---- DECILE OUTPUT ANALYSIS: which deciles pay, IC, year by year ----
fwd1 = mret.shift(-1); rows=[]; ics={}
for d in me:
    s = SCORE.loc[d].dropna()
    if len(s) < 100 or not np.isfinite(fwd1.loc[d]).any(): continue
    dec = pd.qcut(s.rank(method="first"), 10, labels=False); fr = fwd1.loc[d]; mkt = fr.reindex(s.index).mean()
    ics.setdefault(d.year, []).append(pd.Series(s).corr(fr.reindex(s.index), method="spearman"))
    for k in range(10):
        nm = s.index[dec.values==k]; v = fr.reindex(nm).mean()
        if np.isfinite(v): rows.append((d.year, k, v - mkt))
DEC = pd.DataFrame(rows, columns=["yr","dec","ex"])
prof = DEC.groupby("dec")["ex"].mean()*12
print("\nDECILE MAP — mean fwd excess-over-universe return by score decile (D0=short .. D9=long), annualized:")
print("  " + "  ".join(f"D{k}:{prof[k]*100:>+5.1f}" for k in range(10)))
print(f"  IC(score,fwd) all = {np.nanmean([np.nanmean(v) for v in ics.values()]):+.3f}   D9-D0 spread {(prof[9]-prof[0])*100:+.1f}%/yr")
print("\nYEAR-BY-YEAR: D0(short) / D9(long) excess + IC:")
py = DEC.pivot_table(index="yr", columns="dec", values="ex").mul(12)
for y in [y for y in sorted(py.index) if 2013<=y<=2026]:
    print(f"  {y}  D0 {py.loc[y,0]*100:>+5.0f}%  D9 {py.loc[y,9]*100:>+5.0f}%  spread {(py.loc[y,9]-py.loc[y,0])*100:>+5.0f}%  IC {np.nanmean(ics.get(y,[np.nan])):>+.3f}")
# ensemble diversification vs MOM/DM
import os
for nm,pth,key in [("MOM","/tmp/mom_champ.pkl","n1"),("DM","/tmp/dm_returns.pkl",None)]:
    if os.path.exists(pth):
        o=pickle.load(open(pth,"rb")); s=pd.Series(o[key] if key else o); s.index=pd.DatetimeIndex(s.index)
        c=pd.DataFrame({"f":fr,"o":s}).dropna(); print(f"  corr(FUND, {nm}) = {c.corr().iloc[0,1]:+.2f}")
