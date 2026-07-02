#!/usr/bin/env python3
"""SURVIVORSHIP STRESS TEST: re-run the MOM leg on the broad 10,026-name survivorship-robust universe
(top-1000 by $-vol each month — includes names that were liquid then delisted, unlike the 750 'exists-
today' checkpoint). DM stays on its champion tight universe (already survivorship-robust). Net combine.
If MOM's Sharpe holds here vs the 750 version (0.95 net), survivorship wasn't inflating it."""
import warnings; warnings.filterwarnings("ignore")
import os, pickle
import numpy as np, pandas as pd
from xgboost import XGBRegressor
import deep_momentum_xgb as d
from features import make_features, MOM_WINDOWS
import BACKTEST

N_SEEDS = 1
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

# ═══ DM sleeve (cached — deterministic champion) ═══
DMC = "/tmp/dm_streams.pkl"
if os.path.exists(DMC):
    print("[DM] load cached streams ...", flush=True); dm_g, dm_n = pickle.load(open(DMC, "rb"))
else:
    print("[DM] compute champion ...", flush=True)
    pm, rm, sm = d.load_broad_universe_tiingo(skip_download=True, verbose=False)
    elig, short = d.compute_eligibility(pm, sm, min_dollar_vol_pct=0.30, min_dollar_vol_abs=1e6)
    pnl = d._build_pnl_prices(pm); rpnl = pnl.pct_change()
    tc_dm = BACKTEST.tiered_transaction_costs(sm); bf_dm = BACKTEST.tiered_borrow_fees(sm)
    fp0 = d.MIN_TRAIN_YRS * 12 + max(MOM_WINDOWS) + 1
    ffd = d._ffd_from_training_window(pm, fp0)
    for m in list(ffd): ffd[m] = ffd[m].reindex(rm.index)
    dm_w = d._generate_all_dm_weights(rm, sm, pm, q=d.TOP_Q, min_train_months=d.MIN_TRAIN_YRS * 12,
            max_train_months=120, n_seeds=3, use_ffd=True, ffd_scores=ffd, portfolio="ls",
            eligible=elig, shortable=short)["ret"]
    dg, dW = {}, {}
    for t in dm_w.index:
        if t not in pnl.index: continue
        p = pnl.index.get_loc(t)
        if p + 1 >= len(pnl.index): break
        dg[t] = float((dm_w.loc[t] * rpnl.iloc[p + 1]).sum(skipna=True)); dW[t] = dm_w.loc[t]
    dm_g = pd.Series(dg); dm_n = costed(dm_g, dW, tc_dm.reindex(dm_w.index), bf_dm.reindex(dm_w.index))
    pickle.dump((dm_g, dm_n), open(DMC, "wb"))
print(f"[DM] {len(dm_g)} months  net SR {perf(dm_n)[1]:.2f}", flush=True)

# ═══ MOM sleeve on BROAD survivorship-robust universe ═══
print("[MOM] broad universe ...", flush=True)
px = pd.read_parquet("tiingo_daily_close.parquet").sort_index()
vb = pd.read_parquet("tiingo_daily_volume.parquet").reindex_like(px)
rets_d = px.pct_change(); vol_d = rets_d.ewm(span=63, min_periods=20).std()
me = pd.DatetimeIndex(px.index.to_series().resample("ME").last().dropna().values)
mret = px.reindex(me).pct_change(); T = len(me)
dvm = (px * vb).resample("ME").sum(); dvm.index = dvm.index.to_period("M")
mom_dvol = dvm.reindex(pd.PeriodIndex(me, freq="M")); mom_dvol.index = me
cov = px.notna().rolling(252, min_periods=200).mean().reindex(me, method="ffill")
top1000 = mom_dvol.rank(axis=1, ascending=False) <= 1000                    # liquid, survivorship-robust
eligm = (px.reindex(me) > 1.0) & (cov > 0.9) & (mom_dvol > 1e6) & top1000
tc_mom = BACKTEST.tiered_transaction_costs(mom_dvol); bf_mom = BACKTEST.tiered_borrow_fees(mom_dvol)
print(f"[MOM] avg eligible/mo (2011+): {eligm.loc['2011':].sum(1).mean():.0f}", flush=True)
def hl(s): return np.log(0.5) / np.log(1 - 1 / s)
def at_me(f): return f.reindex(me, method="ffill")
def zc(s): return (s - s.mean()) / (s.std() + 1e-9)
pf = {}
for m, dd in [(3, 63), (6, 126), (12, 252)]:
    pf[f"ret{m}"] = at_me(px / px.shift(dd) - 1); pf[f"nret{m}"] = at_me((px / px.shift(dd) - 1) / (vol_d * np.sqrt(dd)))
comp = 0.0
for kk, (S, L) in enumerate([(8, 24), (16, 48), (32, 96)], 1):
    q = (px.ewm(halflife=hl(S)).mean() - px.ewm(halflife=hl(L)).mean()) / px.rolling(63, min_periods=20).std()
    y = q / q.rolling(252, min_periods=60).std(); pf[f"y{kk}"] = at_me(y); comp = comp + y * np.exp(-y**2/4)/0.89
pf["macd_comp"] = at_me(comp); PF = list(pf.keys())
print("[MOM] pool ...", flush=True)
pool = {}
for k in range(13, T - 3):
    dt = me[k]; el = eligm.loc[dt].fillna(False)
    idx0 = el.index[el.values]
    P = pd.DataFrame({c: pf[c].loc[dt].reindex(idx0) for c in PF}); pn = mret.iloc[k + 1].reindex(idx0)
    ok = P.notna().all(axis=1) & mret.iloc[k+1].reindex(idx0).notna() & mret.iloc[k+2].reindex(idx0).notna() & mret.iloc[k+3].reindex(idx0).notna()
    idx = P.index[ok.values]
    if len(idx) < 50: continue
    Y = np.column_stack([zc(mret.iloc[k+h].reindex(idx)).values for h in (1, 2, 3)])
    pool[k] = dict(P=P.loc[idx], Y=Y, pnl=pn.reindex(idx), dt=dt)
keys = sorted(pool); fp = next(i for i, k in enumerate(keys) if me[k].year >= 2011)
print("[MOM] walk-forward ...", flush=True)
BASEP = dict(n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)
mom_g, mom_W = {}, {}; ms = None
for i in range(fp, len(keys)):
    k = keys[i]; dt = pool[k]["dt"]
    if dt.month == 1 or ms is None:
        tr = [keys[j] for j in range(i) if pool[keys[j]]["dt"] < dt][-120:]
        if len(tr) >= 36:
            X = pd.concat([pool[t]["P"] for t in tr]).values; Y = np.vstack([pool[t]["Y"] for t in tr])
            ms = [XGBRegressor(**BASEP, multi_strategy="multi_output_tree", random_state=s).fit(X, Y) for s in range(N_SEEDS)]
            print(f"  [{dt.year}] train_rows={len(X)}", flush=True)
    if ms is None: continue
    sc = np.mean([m.predict(pool[k]["P"].values).mean(1) for m in ms], axis=0)
    s = pd.Series(sc, index=pool[k]["pnl"].index); n = max(1, int(len(s) * 0.10))
    iv = (1 / (vol_d.loc[dt].reindex(s.index) * np.sqrt(21))).clip(upper=50)
    w = pd.Series(0.0, index=s.index); lo, sh = s.nlargest(n).index, s.nsmallest(n).index
    w[lo] = iv[lo] / iv[lo].sum(); w[sh] = -iv[sh] / iv[sh].sum()
    mom_g[dt] = float((w * pool[k]["pnl"]).sum()); mom_W[dt] = w
mom_g = pd.Series(mom_g); mom_n = costed(mom_g, mom_W, tc_mom, bf_mom)

# ═══ align + combine (net) ═══
def align(a, b):
    ap = a.copy(); ap.index = a.index.to_period("M"); bp = b.copy(); bp.index = b.index.to_period("M")
    return pd.DataFrame({"DM": ap, "MOM": bp}).dropna()
Ng = align(dm_n, mom_n)
print("\n" + "=" * 62)
print(f"=== ENSEMBLE on BROAD survivorship-robust MOM ({len(Ng)} months, NET) ===")
print(f"{'strategy':28}{'annNET':>9}{'SRnet':>8}{'DDnet':>9}")
for nm, r in [("DM (reversal, tight)", Ng["DM"]), ("MOM (trend, BROAD ~1000)", Ng["MOM"]),
              ("Combine equal-weight", Ng.mean(1))]:
    a, s, m = perf(r); print(f"{nm:28}{a:>9.1%}{s:>8.2f}{m:>9.1%}")
print(f"\ncorr(DM,MOM) net = {Ng.corr().iloc[0,1]:.2f}   (vs 750-universe: MOM 0.95, combine 1.46)")
print("[done]", flush=True)
