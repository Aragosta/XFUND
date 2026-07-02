#!/usr/bin/env python3
"""CANONICAL GO-TO ENSEMBLE — the single authoritative backtest of the two proven momentum sleeves.

  DM  sleeve : MONTHLY, Han champion (classifier + RET reclass; nMOM/MMOM/size/FFD features), tight
               universe from the 14,744-name survivorship-free panel. Already survivorship-robust.
  MOM sleeve : DAILY, multi-output trend regressor (Poh features: raw+vol-norm returns + Baz MACD;
               rank by mean[t+1,t+2,t+3]), on a CLEANED broad survivorship-robust universe.

CORRECT DATA HANDLING for the broad MOM universe (the fix the survivorship test exposed):
  - returns via pct_change(fill_method=None)  → no forward-fill across gaps
  - mask |monthly return| >= 100% as data errors (unadjusted splits / penny glitches)
  - price > $5 (penny filter) + $-vol > $5M (liquidity) + top-1000 by $-vol (tradeable, survivorship-robust)
COSTS: tiered_transaction_costs + tiered_borrow_fees on both. Reports GROSS and NET; combines = equal-
weight (primary), risk-parity, rolling-Sharpe. DM streams cached (deterministic champion).
"""
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import numpy as np, pandas as pd
from xgboost import XGBRegressor
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

# ─── config ───
N_SEEDS   = 3
MOM_TOPN  = 1000          # broad MOM universe: top-N by $-volume each month
MOM_MINPX = 5.0           # penny-stock filter
MOM_MINDV = 5e6           # liquidity floor ($ / month)
RET_CAP   = 1.0           # mask UPSIDE glitches (r >= +100%); keep real losses incl -100%
DELIST_RET = -0.30        # realistic delisting return (Han fallback) imputed when a name stops trading
BASEP = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)

def perf(r):
    r = pd.Series(r).dropna(); ann = (1 + r).prod() ** (12 / len(r)) - 1; vol = r.std() * np.sqrt(12)
    eq = (1 + r).cumprod(); mdd = (eq / eq.cummax() - 1).min()
    return ann, (r.mean() * 12) / vol if vol > 0 else np.nan, mdd
def costed(gross, W, tc, bf):
    net, prev = {}, None
    for t in sorted(gross.index):
        w = W[t]
        if prev is not None:
            dw = w.subtract(prev, fill_value=0).abs()
            c = float((dw * tc.loc[t].reindex(dw.index).fillna(0.015)).sum())
            b = float((w.clip(upper=0).abs() * bf.loc[t].reindex(w.index).fillna(0.05)).sum()) / 12.0
        else: c = b = 0.0
        net[t] = gross[t] - c - b; prev = w
    return pd.Series(net)

# ═══════════ DM sleeve (cached deterministic champion) ═══════════
DMC = "/tmp/dm_streams.pkl"
if os.path.exists(DMC):
    print("[DM] load cached ...", flush=True); dm_g, dm_n = pickle.load(open(DMC, "rb"))
else:
    print("[DM] compute champion ...", flush=True)
    pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
    elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6)
    pnl = d._build_pnl_prices(pm); rp = pnl.pct_change()
    tcd = BACKTEST.tiered_transaction_costs(sm); bfd = BACKTEST.tiered_borrow_fees(sm)
    ff = d._ffd_from_training_window(pm, d.MIN_TRAIN_YRS * 12 + max(MOM_WINDOWS) + 1)
    for m in list(ff): ff[m] = ff[m].reindex(rm.index)
    W = d._generate_all_dm_weights(rm, sm, pm, q=d.TOP_Q, min_train_months=d.MIN_TRAIN_YRS * 12,
            max_train_months=120, n_seeds=3, use_ffd=True, ffd_scores=ff, portfolio="ls",
            eligible=elig, shortable=short)["ret"]
    dg, dW = {}, {}
    for t in W.index:
        if t not in pnl.index: continue
        p = pnl.index.get_loc(t)
        if p + 1 >= len(pnl.index): break
        dg[t] = float((W.loc[t] * rp.iloc[p + 1]).sum(skipna=True)); dW[t] = W.loc[t]
    dm_g = pd.Series(dg); dm_n = costed(dm_g, dW, tcd.reindex(W.index), bfd.reindex(W.index))
    pickle.dump((dm_g, dm_n), open(DMC, "wb"))
print(f"[DM] net SR {perf(dm_n)[1]:.2f}", flush=True)

# ═══════════ MOM sleeve (cleaned broad survivorship-robust universe) ═══════════
print("[MOM] load broad + CLEAN ...", flush=True)
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
m_px = px.reindex(me)
mret = m_px.pct_change(fill_method=None)                       # no forward-fill across gaps
mret = mret.where(mret < RET_CAP)                             # mask UPSIDE glitches only (keep real losses)
_V = m_px.notna().values; _last = len(me) - 1                 # realistic delisting: name stops mid-sample
for _j in range(_V.shape[1]):                                 #   -> impute -30% delist return next month
    _w = np.where(_V[:, _j])[0]
    if len(_w) and _w[-1] < _last - 1 and _w[-1] + 1 < len(me): mret.iat[_w[-1] + 1, _j] = DELIST_RET
vol_d = px.pct_change(fill_method=None).ewm(span=63, min_periods=20).std()
T = len(me)
dvm = (px * vb).resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mom_dvol = dvm.reindex(pd.PeriodIndex(me, freq="M")); mom_dvol.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
topN = mom_dvol.rank(axis=1, ascending=False) <= MOM_TOPN
eligm = (m_px > MOM_MINPX) & (cov > 0.9) & (mom_dvol > MOM_MINDV) & topN
tc_mom = BACKTEST.tiered_transaction_costs(mom_dvol); bf_mom = BACKTEST.tiered_borrow_fees(mom_dvol)
print(f"[MOM] avg eligible/mo (2011+): {eligm.loc['2011':].sum(1).mean():.0f}", flush=True)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1).where(lambda z: z.abs() < 5)
    pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); POH_COLS = list(pf.keys())
f2011 = int(np.argmax(me.year >= 2011)); ffd = d._ffd_from_training_window(m_px, f2011)   # FFD helps broad
for _m, _v in ffd.items():
    _z = _v.reindex(me); pf[f"ffd{_m}"] = _z.sub(_z.mean(1), 0).div(_z.std(1) + 1e-9, 0)   # cross-sec z
PF = list(pf.keys())
print("[MOM] pool ...", flush=True)
pool = {}
for k in range(13, T - 3):
    dt = me[k]; el = eligm.loc[dt].fillna(False); idx0 = el.index[el.values]
    P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in PF}); pn = mret.iloc[k + 1].reindex(idx0)
    ok = P[POH_COLS].notna().all(axis=1) & mret.iloc[k+1].reindex(idx0).notna() & mret.iloc[k+2].reindex(idx0).notna() & mret.iloc[k+3].reindex(idx0).notna()
    idx = P.index[ok.values]                                     # require Poh valid; FFD neutral-filled
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    pool[k] = dict(P=P.loc[idx].fillna(0.0), Y=Y, pnl=pn.reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
print("[MOM] walk-forward ...", flush=True)
mom_g, mom_W = {}, {}; ms = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month in (1, 4, 7, 10) or ms is None:                 # quarterly retrain (fresher >> annual)
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt]   # expanding window (momentum is stationary)
        if len(tr) >= 36:
            X = pd.concat([pool[t]["P"] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
            ms = [XGBRegressor(**BASEP, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
    if ms is None: continue
    sc = np.mean([m.predict(pool[k]["P"].values).mean(1) for m in ms], axis=0)
    s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
    iv = (1 / (vol_d.loc[dt].reindex(s.index) * np.sqrt(21))).clip(upper=50)
    w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
    w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
    mom_g[dt] = float((w * pool[k]["pnl"]).sum(skipna=True)); mom_W[dt] = w
mom_g = pd.Series(mom_g); mom_n = costed(mom_g, mom_W, tc_mom, bf_mom)
print(f"[MOM] net SR {perf(mom_n)[1]:.2f}", flush=True)

# ═══════════ align + combine (net) ═══════════
def align(a, b):
    ap = a.copy(); ap.index = a.index.to_period("M"); bp = b.copy(); bp.index = b.index.to_period("M")
    return pd.DataFrame({"DM": ap, "MOM": bp}).dropna()
Ng = align(dm_n, mom_n); Gg = align(dm_g, mom_g)
def combine(A, wf, warm=24):
    out = {}
    for i in range(len(A)):
        w = np.array([0.5, 0.5]) if i < warm else wf(A.iloc[:i]); out[A.index[i]] = float(w @ A.iloc[i].values)
    return pd.Series(out)
def w_rp(h): iv = 1 / (h.std().values + 1e-9); return iv / iv.sum()
def w_sr(h):
    sr = (h.mean() * 12 / (h.std() * np.sqrt(12) + 1e-9)).clip(lower=0).values
    return (sr / sr.sum()) if sr.sum() > 0 else np.array([0.5, 0.5])

print("\n" + "=" * 66)
print(f"=== CANONICAL ENSEMBLE (cleaned broad MOM, {len(Ng)} months) ===")
print(f"{'strategy':28}{'gSR':>7}{'gDD':>8}  |  {'netSR':>7}{'netAnn':>8}{'netDD':>8}")
rows = [("DM (reversal)", Gg["DM"], Ng["DM"]), ("MOM (trend, clean broad)", Gg["MOM"], Ng["MOM"]),
        ("Combine equal-weight", Gg.mean(1), Ng.mean(1)),
        ("Combine risk-parity", combine(Gg, w_rp), combine(Ng, w_rp)),
        ("Combine rolling-Sharpe", combine(Gg, w_sr), combine(Ng, w_sr))]
for nm, g, nn in rows:
    _, gs, gd = perf(g); na, ns, nd = perf(nn)
    print(f"{nm:28}{gs:>7.2f}{gd:>8.1%}  |  {ns:>7.2f}{na:>8.1%}{nd:>8.1%}")
print(f"\ncorr(DM,MOM) net = {Ng.corr().iloc[0,1]:.2f}")
print("[done]", flush=True)
