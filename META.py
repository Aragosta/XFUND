#!/usr/bin/env python3
"""META.py — the meta-model layer (de Prado meta-labeling): a NAME-level confidence sizer on top of MOM+DM.

Pipeline (each layer distinct, none replaces another):
    MOM, DM decile books  --BETANEUT--> beta-neut  --ERC--> combined book  --META--> name-sized book
The primary sleeves (MOM, DM) decide the SIDE; ERC allocates CAPITAL across them; META sizes each NAME by its
predicted confidence. NO stop-loss execution (so the training TARGET is also no-stop: month-end profitability),
NO swept-lambda tilt — sizing is the parameter-free calibrated ODDS p/(1-p) (Kelly for a calibrated prob).

Three parts:
  FEATURES : position sign, vol, 12-1 mom, reversal, size, market vol, market drawdown, dispersion,
             AND THE SLEEVE OUTPUTS (MOM weight, DM weight, their agreement).
  TARGET   : binary — did the position (in the sleeve's direction) end the month PROFITABLE (no stop).
  MODEL    : XGBoost classifier + isotonic calibration, walk-forward (quarterly retrain, 2-mo embargo, seed-bag).
Net via BACKTEST.py (tiered). Compares ERC baseline vs + META (calibrated-odds sizing)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, pickle
from scipy.stats import skew
from xgboost import XGBClassifier
from sklearn.isotonic import IsotonicRegression
import BACKTEST, BETANEUT, ERC

# ---------- DATA / GRID ----------
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
t = px.columns.str.match(r"^Z[A-Z]ZZT$") | px.columns.isin(["ZXYZ","ZTEST","ZVV","ZBZX","ZBZZT"]); px = px.loc[:, ~t]
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex(columns=px.columns).reindex(index=px.index)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values); me = me[me >= pd.Timestamp("2010-12-01")]
m_px = px.reindex(me); mret = m_px.pct_change(fill_method=None).where(lambda z: z < 1.0); synth = (1 + mret.fillna(0.0)).cumprod()
mdv = (px*vb).resample("ME").sum().reindex(me, method="ffill"); cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
elig = (m_px > 5) & (cov > 0.9) & (mdv > 5e6); tc = BACKTEST.tiered_transaction_costs(mdv); bf = BACKTEST.tiered_borrow_fees(mdv); meP = pd.PeriodIndex(me, freq="M")
volm = (px.pct_change(fill_method=None).ewm(span=63, min_periods=20).std()*np.sqrt(21)).reindex(me, method="ffill")
BETA = BETANEUT.rolling_beta(mret, elig, bw=60)
def to_grid(W):
    W = W.copy(); W.index = pd.PeriodIndex(pd.DatetimeIndex(W.index), freq="M"); W = W[~W.index.duplicated()].reindex(meP); W.index = me
    return W.reindex(columns=px.columns).astype(float)

# ---------- SLEEVES: load + beta-neutralize ----------
Wmom = to_grid(pickle.load(open("/tmp/mom_weights.pkl","rb"))); Wdm = to_grid(pickle.load(open("/tmp/dm_weights.pkl","rb")))
Wmom_bn = BETANEUT.betaneut(Wmom, BETA); Wdm_bn = BETANEUT.betaneut(Wdm, BETA)

# ---------- ERC combine (capital, sleeve-level) ----------
def netstream(W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)["returns"]
    s = pd.Series(r); s.index = pd.PeriodIndex(s.index, freq="M"); return s
st = pd.DataFrame({"MOM": netstream(Wmom_bn), "DM": netstream(Wdm_bn)}).dropna()
Aw = ERC.expanding_alloc(st, method="erc", win=36); Aw.index = st.index.to_timestamp("M").to_period("M")
WERC = {d: (Aw.loc[d.to_period("M"),"MOM"]*Wmom_bn.loc[d].fillna(0) + Aw.loc[d.to_period("M"),"DM"]*Wdm_bn.loc[d].fillna(0)) for d in me if d.to_period("M") in Aw.index}

# ---------- META FEATURES + TARGET (per position) ----------
mom12 = (m_px.shift(1)/m_px.shift(12)-1).where(lambda z: z.abs() < 5); size = np.log(mdv)
mkt = mret.where(elig).mean(axis=1); mkt_rv = mkt.rolling(6, min_periods=3).std()*np.sqrt(12)
eq = (1+mkt.fillna(0)).cumprod(); mkt_dd = eq/eq.cummax()-1; disp = mret.where(elig).std(axis=1)
Wc = (Wmom.fillna(0)*0.5 + Wdm.fillna(0)*0.5)                               # union of decile picks = the positions to judge
rows = []
for k in range(len(me)-1):
    w = Wc.iloc[k].values; idx = np.where(np.abs(w) > 1e-9)[0]
    if len(idx) == 0: continue
    sgn = np.sign(w[idx]); fwd = mret.iloc[k+1].values[idx]
    label = (sgn*fwd > 0).astype(int)                                      # TARGET: position profitable at month-end (no stop)
    mw = Wmom.iloc[k].values[idx]; dw = Wdm.iloc[k].values[idx]
    agree = ((np.sign(mw) == np.sign(dw)) & (mw != 0) & (dw != 0)).astype(float)
    F = np.column_stack([(sgn > 0).astype(float), volm.iloc[k].values[idx], mom12.iloc[k].values[idx], mret.iloc[k].values[idx],
                         size.iloc[k].values[idx], np.full(len(idx), mkt_rv.iloc[k]), np.full(len(idx), mkt_dd.iloc[k]),
                         np.full(len(idx), disp.iloc[k]), mw, dw, agree])   # FEATURES (+ sleeve outputs)
    for j, ii in enumerate(idx):
        if np.isfinite(fwd[j]): rows.append((k, ii, label[j], *F[j]))
FN = ["f_long","f_vol","f_mom","f_ret1","f_size","f_mrv","f_mdd","f_disp","f_momw","f_dmw","f_agree"]
D = pd.DataFrame(rows, columns=["k","col","y"]+FN)
print(f"[META] {len(D)} positions, profitable {D['y'].mean():.1%}", flush=True)

# ---------- META MODEL: XGB classifier + isotonic calibration, walk-forward ----------
ks = sorted(D["k"].unique()); fpk = next(k for k in ks if me[k].year >= 2013); p = pd.Series(np.nan, index=D.index); mdl = iso = None
for k in ks:
    if k < fpk: continue
    if mdl is None or me[k].month in (1,4,7,10):
        tr = D[D["k"] <= k-2]; X = tr[FN].values; ok = np.isfinite(X).all(1)  # 2-month embargo (purges 1-mo labels)
        if ok.sum() > 500:
            mdl = [XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                   eval_metric="logloss", verbosity=0, random_state=s).fit(X[ok], tr["y"].values[ok]) for s in range(2)]
            raw = np.mean([m.predict_proba(np.nan_to_num(X[ok]))[:,1] for m in mdl], axis=0)
            iso = IsotonicRegression(out_of_bounds="clip").fit(raw, tr["y"].values[ok])   # calibration
    if mdl is None: continue
    mk = (D["k"] == k).values
    raw = np.mean([m.predict_proba(np.nan_to_num(D.loc[mk, FN].values))[:,1] for m in mdl], axis=0)
    p.iloc[np.where(mk)[0]] = iso.predict(raw)
D["p"] = p; oos = D[D["p"].notna()]
print(f"[META] OOS edge: hi-p {oos[oos['p']>oos['p'].median()]['y'].mean():.1%} vs lo-p {oos[oos['p']<=oos['p'].median()]['y'].mean():.1%}", flush=True)
pmat = pd.DataFrame(np.nan, index=me, columns=px.columns)
for _, r in oos.iterrows(): pmat.iat[int(r["k"]), int(r["col"])] = r["p"]

# ---------- SIZING: calibrated ODDS p/(1-p), name-level, re-neutralize ----------
def book(sized):
    W = pd.DataFrame(0.0, index=me, columns=px.columns)
    for d, werc in WERC.items():
        mult = pd.Series(1.0, index=px.columns)
        if sized:
            pm = pmat.loc[d]; isd = pm.notna()
            if isd.any(): mult[isd] = (pm[isd]/(1-pm[isd])).clip(0, 10)     # calibrated odds
        w = werc*mult
        W.loc[d] = BETANEUT.neutralize(w[w != 0], BETA.loc[d]).reindex(px.columns).fillna(0.0) if (w != 0).any() else 0.0
    return W
def rep(name, W):
    r = BACKTEST.backtest(W.fillna(0.0), synth, freq=12, lag=0, signal_dates=[d for d in W.index if W.loc[d].abs().sum() > 1e-9], transaction_cost=tc, borrow_fee=bf)
    rr = pd.Series(r["returns"]); rr.index = pd.DatetimeIndex(rr.index); rr = rr[rr.index >= "2011-01-01"]
    print(f"  {name:32}{r['sharpe']:>6.2f}{r['ann_return']:>8.1%}{r['max_drawdown']:>8.1%}{skew(rr):>+7.2f}{r['ann_turnover']:>7.1f}", flush=True)
print("="*84); print("MOM+DM (beta-neut) ERC + META overlay (calibrated odds) — NET, 2011+")
print(f"  {'variant':32}{'SR':>6}{'ann':>8}{'maxDD':>8}{'skew':>7}{'turn':>7}")
Wfinal = book(True)
rep("ERC baseline (no meta)", book(False)); rep("+ META (calibrated odds)", Wfinal)
pickle.dump(Wfinal, open("/tmp/meta_weights.pkl","wb")); print("[done] final book -> /tmp/meta_weights.pkl", flush=True)
